"""
Crypto Quick Trade UI
Enhanced quick trade interface with ticker management, scanner integration,
investment controls, risk management, AI validation, and automated execution
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
from loguru import logger
import json
import os
from clients.kraken_client import OrderType, OrderSide



def display_quick_trade_header():
    """Display header for quick trade section"""
    st.markdown("### ⚡ Enhanced Quick Crypto Trade")
    st.write("Manage tickers, integrate scanners, customize investments, and execute trades with advanced risk management")


def display_strategy_quick_selector():
    """Display quick strategy selector buttons"""
    st.markdown("**🎯 Quick Strategy Selection:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚡ Scalping Mode", use_container_width=True, type="primary"):
            st.session_state.crypto_quick_strategy = "SCALP"
            st.session_state.crypto_quick_timeframe = "5m"
            st.session_state.crypto_quick_stop_pct = 1.0
            st.session_state.crypto_quick_target_pct = 2.5
            st.rerun()
    
    with col2:
        if st.button("🚀 Momentum Mode", use_container_width=True):
            st.session_state.crypto_quick_strategy = "MOMENTUM"
            st.session_state.crypto_quick_timeframe = "15m"
            st.session_state.crypto_quick_stop_pct = 2.0
            st.session_state.crypto_quick_target_pct = 5.0
            st.rerun()
    
    with col3:
        if st.button("📊 Swing Mode", use_container_width=True):
            st.session_state.crypto_quick_strategy = "SWING"
            st.session_state.crypto_quick_timeframe = "1h"
            st.session_state.crypto_quick_stop_pct = 3.0
            st.session_state.crypto_quick_target_pct = 8.0
            st.rerun()


def display_trade_setup(kraken_client, crypto_config):
    """Display trade setup form"""
    
    # Get current strategy settings
    strategy = st.session_state.get('crypto_quick_strategy', 'SCALP')
    timeframe = st.session_state.get('crypto_quick_timeframe', '5m')
    stop_pct = st.session_state.get('crypto_quick_stop_pct', 1.0)
    target_pct = st.session_state.get('crypto_quick_target_pct', 2.5)
    
    st.markdown(f"**Current Mode:** {strategy} | Timeframe: {timeframe}")
    
    st.divider()
    
    # Trade parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📍 Entry Setup**")
        
        # Symbol selection
        trade_pair = st.selectbox(
            "Trading Pair",
            crypto_config.CRYPTO_WATCHLIST if hasattr(crypto_config, 'CRYPTO_WATCHLIST') 
            else ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD'],
            key="quick_trade_pair",
            help="Select crypto pair to trade"
        )
        
        # Check if there's a pre-selected symbol from scanner
        if 'crypto_quick_trade_pair' in st.session_state and st.session_state.crypto_quick_trade_pair:
            trade_pair = st.session_state.crypto_quick_trade_pair
            st.info(f"📌 Pre-selected from scanner: {trade_pair}")
        
        # Side selection
        trade_side = st.radio(
            "Direction",
            ['BUY', 'SELL'],
            horizontal=True,
            key="quick_trade_side"
        )
        
        # Position size - get custom capital if set
        available_capital = st.session_state.get('crypto_quick_capital', crypto_config.TOTAL_CAPITAL)
        max_position_pct = st.session_state.get('crypto_quick_max_position_pct', crypto_config.MAX_POSITION_SIZE_PCT)
        
        # Dynamic min/max based on available capital
        min_position = max(10.0, available_capital * 0.01)  # Min 1% of capital or $10
        max_position = min(10000.0, available_capital)
        suggested_position = min(available_capital * (max_position_pct / 100), available_capital * 0.5)
        
        position_size_usd = st.number_input(
            "Position Size (USD)",
            min_value=min_position,
            max_value=max_position,
            value=min(suggested_position, max_position),
            step=10.0 if available_capital < 100 else 50.0,
            key="crypto_position_size",
            help=f"Dollar amount to invest (1-{max_position_pct}% of ${available_capital:,.2f} capital)"
        )
    
    with col2:
        st.markdown("**🎯 Risk Management**")
        
        # Stop loss
        custom_stop = st.checkbox("Custom Stop Loss", value=False, key="custom_stop")
        
        if custom_stop:
            stop_loss_pct = st.slider(
                "Stop Loss %",
                min_value=0.5,
                max_value=10.0,
                value=stop_pct,
                step=0.5,
                key="quick_stop_loss"
            )
        else:
            stop_loss_pct = stop_pct
            st.text(f"Auto Stop Loss: {stop_loss_pct}% (based on {strategy} strategy)")
        
        # Take profit
        custom_target = st.checkbox("Custom Take Profit", value=False, key="custom_target")
        
        if custom_target:
            take_profit_pct = st.slider(
                "Take Profit %",
                min_value=1.0,
                max_value=20.0,
                value=target_pct,
                step=0.5,
                key="quick_take_profit"
            )
        else:
            take_profit_pct = target_pct
            st.text(f"Auto Take Profit: {take_profit_pct}% (based on {strategy} strategy)")
        
        # Calculate R:R
        risk_reward = take_profit_pct / stop_loss_pct
        
        rr_color = "🟢" if risk_reward >= 2 else "🟡" if risk_reward >= 1.5 else "🔴"
        st.metric("Risk:Reward Ratio", f"{rr_color} {risk_reward:.2f}:1")
    
    st.divider()
    
    # Fetch current price and display trade preview
    try:
        ticker = kraken_client.get_ticker_data(trade_pair)
        
        if ticker:
            current_price = ticker['last_price']
            bid_price = ticker['bid']
            ask_price = ticker['ask']
            
            # Display current market data
            st.markdown("**💰 Current Market Data:**")
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            
            with mcol1:
                st.metric("Last Price", f"${current_price:,.2f}")
            
            with mcol2:
                st.metric("Bid", f"${bid_price:,.2f}")
            
            with mcol3:
                st.metric("Ask", f"${ask_price:,.2f}")
            
            with mcol4:
                spread = ((ask_price - bid_price) / bid_price) * 100
                st.metric("Spread", f"{spread:.3f}%")
            
            st.divider()
            
            # Calculate trade details
            entry_price = ask_price if trade_side == 'BUY' else bid_price
            crypto_amount = position_size_usd / entry_price
            
            if trade_side == 'BUY':
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                take_profit_price = entry_price * (1 + take_profit_pct / 100)
            else:
                stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
                take_profit_price = entry_price * (1 - take_profit_pct / 100)
            
            risk_amount = position_size_usd * (stop_loss_pct / 100)
            reward_amount = position_size_usd * (take_profit_pct / 100)
            
            # Display trade preview
            st.markdown("**📋 Trade Preview:**")
            
            tcol1, tcol2 = st.columns(2)
            
            with tcol1:
                st.markdown("**Position Details:**")
                st.text(f"Entry Price: ${entry_price:,.2f}")
                st.text(f"Crypto Amount: {crypto_amount:.6f} {trade_pair.split('/')[0]}")
                st.text(f"Position Value: ${position_size_usd:,.2f}")
            
            with tcol2:
                st.markdown("**Exit Prices:**")
                st.text(f"Stop Loss: ${stop_loss_price:,.2f}")
                st.text(f"Take Profit: ${take_profit_price:,.2f}")
                st.text(f"R:R Ratio: {risk_reward:.2f}:1")
            
            st.divider()
            
            # Risk/Reward breakdown
            st.markdown("**💹 Expected Outcomes:**")
            
            ocol1, ocol2, ocol3 = st.columns(3)
            
            with ocol1:
                st.markdown("**If Take Profit Hit:**")
                st.success(f"**Profit:** +${reward_amount:,.2f}")
                profit_pct_of_account = (reward_amount / crypto_config.TOTAL_CAPITAL) * 100
                st.text(f"({profit_pct_of_account:.2f}% of account)")
            
            with ocol2:
                st.markdown("**If Stop Loss Hit:**")
                st.error(f"**Loss:** -${risk_amount:,.2f}")
                loss_pct_of_account = (risk_amount / crypto_config.TOTAL_CAPITAL) * 100
                st.text(f"({loss_pct_of_account:.2f}% of account)")
            
            with ocol3:
                st.markdown("**Break-Even:**")
                if trade_side == 'BUY':
                    be_price = entry_price * (1 + spread/100)
                else:
                    be_price = entry_price * (1 - spread/100)
                st.info(f"**Price:** ${be_price:,.2f}")
                st.text(f"(after spread)")
            
            st.divider()
            
            # Safety checks
            st.markdown("**⚠️ Safety Checks:**")
            
            checks = []
            warnings = []
            
            # Check 1: Position size vs account
            position_pct = (position_size_usd / crypto_config.TOTAL_CAPITAL) * 100
            if position_pct <= crypto_config.MAX_POSITION_SIZE_PCT:
                checks.append(f"✅ Position size OK ({position_pct:.1f}% of account)")
            else:
                warnings.append(f"⚠️ Position size exceeds limit ({position_pct:.1f}% > {crypto_config.MAX_POSITION_SIZE_PCT}%)")
            
            # Check 2: Risk per trade
            risk_pct_account = (risk_amount / crypto_config.TOTAL_CAPITAL) * 100
            max_risk = crypto_config.RISK_PER_TRADE_PCT * 100
            if risk_pct_account <= max_risk:
                checks.append(f"✅ Risk per trade OK ({risk_pct_account:.2f}% of account)")
            else:
                warnings.append(f"⚠️ Risk exceeds limit ({risk_pct_account:.2f}% > {max_risk}%)")
            
            # Check 3: Risk/Reward ratio
            if risk_reward >= 2.0:
                checks.append(f"✅ Excellent R:R ratio ({risk_reward:.2f}:1)")
            elif risk_reward >= 1.5:
                checks.append(f"🟡 Acceptable R:R ratio ({risk_reward:.2f}:1)")
            else:
                warnings.append(f"⚠️ Poor R:R ratio ({risk_reward:.2f}:1) - Consider 2:1 minimum")
            
            # Check 4: Spread
            if spread < 0.1:
                checks.append(f"✅ Tight spread ({spread:.3f}%)")
            else:
                warnings.append(f"⚠️ Wide spread ({spread:.3f}%) - may impact profit")
            
            # Display checks
            for check in checks:
                st.success(check)
            
            for warning in warnings:
                st.warning(warning)
            
            st.divider()
            
            # Execute button
            can_trade = len(warnings) == 0
            
            if can_trade:
                st.success("✅ All safety checks passed - Ready to trade!")
            else:
                st.error("❌ Safety checks failed - Review warnings above")
            
            # Initialize AI reviewer if not already done
            if 'ai_trade_reviewer' not in st.session_state:
                try:
                    from services.ai_crypto_trade_reviewer import AICryptoTradeReviewer
                    from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                    
                    llm_analyzer = LLMStrategyAnalyzer()
                    st.session_state.ai_trade_reviewer = AICryptoTradeReviewer(llm_analyzer=llm_analyzer)
                    logger.info("✅ AI Trade Reviewer initialized")
                except Exception as e:
                    logger.warning(f"AI Trade Reviewer init failed: {e}")
                    st.session_state.ai_trade_reviewer = None
            
            # AI Review Button
            col_ai1, col_ai2 = st.columns(2)
            
            with col_ai1:
                if st.button("🤖 Get AI Review", use_container_width=True):
                    if st.session_state.ai_trade_reviewer:
                        with st.spinner("AI analyzing trade..."):
                            # Get capital info
                            total_capital = st.session_state.get('crypto_quick_capital', crypto_config.TOTAL_CAPITAL)
                            actual_balance = st.session_state.get('crypto_actual_balance', 0)
                            
                            approved, confidence, reasoning, recommendations = st.session_state.ai_trade_reviewer.pre_trade_review(
                                pair=trade_pair,
                                side=trade_side,
                                entry_price=entry_price,
                                position_size_usd=position_size_usd,
                                stop_loss_price=stop_loss_price,
                                take_profit_price=take_profit_price,
                                strategy=strategy,
                                market_data=ticker,
                                total_capital=total_capital,
                                actual_balance=actual_balance
                            )
                            
                            st.session_state.ai_review_result = {
                                'approved': approved,
                                'confidence': confidence,
                                'reasoning': reasoning,
                                'recommendations': recommendations,
                                'timestamp': datetime.now(),
                                'capital_available': st.session_state.get('crypto_quick_capital', crypto_config.TOTAL_CAPITAL),
                                'actual_balance': st.session_state.get('crypto_actual_balance', 0)
                            }
                            st.rerun()
                    else:
                        st.warning("AI Trade Reviewer not available")
            
            with col_ai2:
                if st.button("🚀 Execute Trade", type="primary", disabled=not can_trade, use_container_width=True):
                    st.session_state.show_execute_confirmation = True
                    st.rerun()
            
            # Display AI Review Results
            if 'ai_review_result' in st.session_state:
                review = st.session_state.ai_review_result
                
                st.divider()
                st.markdown("### 🤖 AI Trade Review")
                
                if review['approved']:
                    st.success(f"✅ **APPROVED** - Confidence: {review['confidence']:.1f}%")
                else:
                    st.error(f"❌ **REJECTED** - Confidence: {review['confidence']:.1f}%")
                
                st.markdown(f"**Analysis:** {review['reasoning'][:200]}...")
                
                with st.expander("📋 Detailed AI Recommendations", expanded=False):
                    recs = review['recommendations']
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        st.markdown(f"**Entry Timing:** {recs.get('entry_timing', 'N/A')}")
                        st.markdown(f"**Position Size:** {recs.get('position_size', 'N/A')}")
                        st.markdown(f"**Market Conditions:** {recs.get('market_conditions', 'N/A')}")
                    
                    with col_r2:
                        st.markdown("**Top Risks:**")
                        for risk in recs.get('risks', [])[:3]:
                            st.markdown(f"- {risk}")
                        
                        st.markdown("**Recommendations:**")
                        for rec in recs.get('recommendations', [])[:3]:
                            st.markdown(f"- {rec}")
            
            # Execute confirmation modal
            if st.session_state.get('show_execute_confirmation'):
                st.divider()
                st.warning("⚠️ **LIVE TRADING CONFIRMATION** - This will execute a REAL trade with REAL money on Kraken!")
                
                # Show AI approval status
                if 'ai_review_result' in st.session_state:
                    review = st.session_state.ai_review_result
                    capital_info = f" | Available: ${review.get('capital_available', 0):,.2f}"
                    if review.get('actual_balance', 0) > 0:
                        capital_info += f" | Kraken Balance: ${review['actual_balance']:,.2f}"
                    
                    if review['approved']:
                        st.info(f"✅ AI approved this trade ({review['confidence']:.1f}% confidence){capital_info}")
                    else:
                        st.error(f"⚠️ AI rejected this trade ({review['confidence']:.1f}% confidence) - Proceed with caution!{capital_info}")
                else:
                    # Show current capital info
                    capital = st.session_state.get('crypto_quick_capital', crypto_config.TOTAL_CAPITAL)
                    actual_bal = st.session_state.get('crypto_actual_balance', 0)
                    info_text = f"💡 Tip: Get AI review before executing | Investment: ${capital:,.2f}"
                    if actual_bal > 0:
                        info_text += f" | Kraken Balance: ${actual_bal:,.2f}"
                    st.warning(info_text)
                
                # Order type selection
                use_limit = st.checkbox(
                    "✅ Use Limit Order (Recommended - Safer than Market Order)",
                    value=True,
                    key="use_limit_order",
                    help="Limit orders execute at your specified price or better. Market orders execute immediately at current price."
                )
                
                if use_limit:
                    limit_price = st.number_input(
                        f"Limit Price (Current: ${entry_price:,.2f})",
                        min_value=entry_price * 0.95,
                        max_value=entry_price * 1.05,
                        value=entry_price,
                        step=entry_price * 0.001,
                        key="limit_price_input",
                        help="Your order will only execute at this price or better"
                    )
                else:
                    limit_price = None
                
                # Final confirmation
                confirm = st.checkbox(
                    f"I confirm I want to {trade_side} {crypto_amount:.6f} {trade_pair.split('/')[0]} at ${entry_price:,.2f}",
                    key="execute_confirm"
                )
                
                col_ex1, col_ex2 = st.columns(2)
                
                with col_ex1:
                    if st.button("✅ Confirm & Execute", disabled=not confirm, use_container_width=True, type="primary"):
                        # Execute the trade
                        with st.spinner("Executing trade on Kraken..."):
                            try:
                                # Determine order type
                                order_type = OrderType.LIMIT if use_limit else OrderType.MARKET
                                order_side = OrderSide.BUY if trade_side == 'BUY' else OrderSide.SELL
                                
                                # Place order
                                order = kraken_client.place_order(
                                    pair=trade_pair,
                                    side=order_side,
                                    order_type=order_type,
                                    volume=crypto_amount,
                                    price=limit_price if use_limit else None,
                                    stop_loss=stop_loss_price,
                                    take_profit=take_profit_price,
                                    validate=False
                                )
                                
                                if order:
                                    st.success(f"✅ Trade executed successfully!")
                                    st.success(f"Order ID: {order.order_id}")
                                    
                                    # Start AI monitoring
                                    if st.session_state.ai_trade_reviewer:
                                        st.session_state.ai_trade_reviewer.start_trade_monitoring(
                                            trade_id=order.order_id,
                                            pair=trade_pair,
                                            side=trade_side,
                                            entry_price=entry_price,
                                            current_price=entry_price,
                                            volume=crypto_amount,
                                            stop_loss=stop_loss_price,
                                            take_profit=take_profit_price,
                                            strategy=strategy
                                        )
                                        st.info("📊 AI monitoring activated for this trade")
                                    
                                    # Save to session history
                                    if 'trade_history' not in st.session_state:
                                        st.session_state.trade_history = []
                                    
                                    st.session_state.trade_history.append({
                                        'timestamp': datetime.now(),
                                        'order_id': order.order_id,
                                        'pair': trade_pair,
                                        'side': trade_side,
                                        'amount': crypto_amount,
                                        'entry_price': entry_price,
                                        'stop_loss': stop_loss_price,
                                        'take_profit': take_profit_price,
                                        'order_type': order_type.value,
                                        'status': 'active'
                                    })
                                    
                                    # Display trade details
                                    st.code(f"""
✅ TRADE EXECUTED
Order ID: {order.order_id}
Pair: {trade_pair}
Side: {trade_side}
Type: {order_type.value.upper()}
Amount: {crypto_amount:.6f}
Entry: ${entry_price:,.2f}
Stop Loss: ${stop_loss_price:,.2f}
Take Profit: ${take_profit_price:,.2f}
Position Size: ${position_size_usd:,.2f}
                                    """.strip())
                                    
                                    # Clear execution flag
                                    st.session_state.show_execute_confirmation = False
                                else:
                                    st.error("❌ Trade execution failed. Please check logs.")
                                
                            except Exception as e:
                                st.error(f"❌ Execution error: {e}")
                                logger.error(f"Trade execution error: {e}", exc_info=True)
                
                with col_ex2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.show_execute_confirmation = False
                        st.rerun()
                
                st.divider()
                
                # Trade summary
                st.markdown("📋 **Trade Summary:**")
                st.code(f"""
Pair: {trade_pair}
Side: {trade_side}
Order Type: {'LIMIT' if use_limit else 'MARKET'}
{'Limit Price: $' + f'{limit_price:,.2f}' if use_limit else 'Market Price: $' + f'{entry_price:,.2f}'}
Amount: {crypto_amount:.6f} {trade_pair.split('/')[0]}
Stop Loss: ${stop_loss_price:,.2f} (-{stop_loss_pct}%)
Take Profit: ${take_profit_price:,.2f} (+{take_profit_pct}%)
Position Size: ${position_size_usd:,.2f}
Risk: ${risk_amount:,.2f} ({stop_loss_pct}%)
Reward: ${reward_amount:,.2f} ({take_profit_pct}%)
R:R Ratio: {risk_reward:.2f}:1
                """.strip())
            
            return {
                'pair': trade_pair,
                'side': trade_side,
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'position_size_usd': position_size_usd,
                'crypto_amount': crypto_amount,
                'risk_amount': risk_amount,
                'reward_amount': reward_amount,
                'risk_reward': risk_reward
            }
        
        else:
            st.error(f"Could not fetch price data for {trade_pair}")
            return None
    
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        logger.error(f"Quick trade error: {e}", exc_info=True)
        return None


def manage_custom_tickers():
    """Manage custom ticker list for quick trading"""
    st.markdown("#### 📋 Ticker Management")
    
    # Initialize custom tickers in session state
    if 'crypto_custom_tickers' not in st.session_state:
        st.session_state.crypto_custom_tickers = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Add ticker input
        new_ticker = st.text_input(
            "Add Ticker (e.g., BTC/USD, ETH/USD, SOL/USD)",
            key="add_crypto_ticker_input",
            placeholder="Enter crypto pair..."
        )
    
    with col2:
        st.markdown("")  # Spacer
        st.markdown("")  # Spacer
        if st.button("➕ Add Ticker", key="add_crypto_ticker_btn", use_container_width=True):
            if new_ticker and new_ticker not in st.session_state.crypto_custom_tickers:
                st.session_state.crypto_custom_tickers.append(new_ticker.upper())
                st.success(f"✅ Added {new_ticker.upper()}")
                # Don't rerun - just update state
            elif new_ticker in st.session_state.crypto_custom_tickers:
                st.warning("Ticker already in list")
    
    # Display current tickers
    if st.session_state.crypto_custom_tickers:
        st.markdown(f"**🎯 Active Tickers ({len(st.session_state.crypto_custom_tickers)}):**")
        
        # Create removable ticker chips
        cols = st.columns(5)
        for idx, ticker in enumerate(st.session_state.crypto_custom_tickers):
            with cols[idx % 5]:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"`{ticker}`")
                with col_b:
                    if st.button("✖", key=f"remove_ticker_{idx}"):
                        st.session_state.crypto_custom_tickers.remove(ticker)
                        st.success(f"Removed {ticker}")
                        # Let Streamlit auto-refresh on next interaction
        
        # Clear all button
        if st.button("🗑️ Clear All Tickers", key="clear_all_tickers"):
            st.session_state.crypto_custom_tickers = []
            st.success("Cleared all tickers")
    else:
        st.info("📄 No custom tickers added yet")


def integrate_scanners(kraken_client, crypto_config):
    """Integrate with existing scanners to pull tickers"""
    st.markdown("#### 🔍 Quick Import from Scanners")
    st.info("💡 Tip: These import from major crypto pairs - add manually for specific coins")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⭐ Import Major Pairs", key="import_major", use_container_width=True):
            try:
                # Import major crypto pairs
                major_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD', 
                              'AVAX/USD', 'DOT/USD', 'MATIC/USD', 'LINK/USD', 'ATOM/USD']
                
                if 'crypto_custom_tickers' not in st.session_state:
                    st.session_state.crypto_custom_tickers = []
                
                added = 0
                for ticker in major_pairs:
                    if ticker not in st.session_state.crypto_custom_tickers:
                        st.session_state.crypto_custom_tickers.append(ticker)
                        added += 1
                
                if added > 0:
                    st.success(f"✅ Added {added} major crypto pairs")
                else:
                    st.info("All major pairs already added")
            except Exception as e:
                st.error(f"Error importing major pairs: {e}")
    
    with col2:
        if st.button("💰 Import Altcoins", key="import_altcoins", use_container_width=True):
            try:
                # Import popular altcoins
                altcoins = ['UNI/USD', 'AAVE/USD', 'DOGE/USD', 'SHIB/USD', 'LTC/USD',
                           'BCH/USD', 'ALGO/USD', 'VET/USD', 'SAND/USD', 'MANA/USD']
                
                if 'crypto_custom_tickers' not in st.session_state:
                    st.session_state.crypto_custom_tickers = []
                
                added = 0
                for ticker in altcoins:
                    if ticker not in st.session_state.crypto_custom_tickers:
                        st.session_state.crypto_custom_tickers.append(ticker)
                        added += 1
                
                if added > 0:
                    st.success(f"✅ Added {added} altcoin pairs")
                else:
                    st.info("All altcoins already added")
            except Exception as e:
                st.error(f"Error importing altcoins: {e}")
    
    st.markdown("---")
    
    # Advanced: Import from watchlist
    with st.expander("🔍 Advanced: Import from Saved Watchlist", expanded=False):
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            if st.button("⭐ My Crypto Watchlist", key="import_watchlist", use_container_width=True):
                try:
                    # Get watchlist from crypto_watchlist_manager
                    if 'crypto_watchlist_manager' in st.session_state:
                        wl_manager = st.session_state.crypto_watchlist_manager
                        watchlist = wl_manager.get_watchlist()
                        
                        if 'crypto_custom_tickers' not in st.session_state:
                            st.session_state.crypto_custom_tickers = []
                        
                        added = 0
                        # Import top 10 from watchlist
                        for item in watchlist[:10]:
                            ticker = item.get('pair', item.get('symbol', ''))
                            if ticker and ticker not in st.session_state.crypto_custom_tickers:
                                st.session_state.crypto_custom_tickers.append(ticker)
                                added += 1
                        
                        if added > 0:
                            st.success(f"✅ Imported {added} from watchlist")
                        else:
                            st.info("All watchlist items already added")
                    else:
                        st.warning("⚠️ Watchlist manager not available - add cryptos to watchlist first")
                except Exception as e:
                    st.error(f"Error importing watchlist: {e}")
        
        with col_w2:
            if st.button("📋 Penny Scanner Results", key="import_penny", use_container_width=True):
                try:
                    # Try to scan penny cryptos directly
                    from services.penny_crypto_scanner import PennyCryptoScanner
                    
                    with st.spinner("Scanning for penny cryptos..."):
                        scanner = PennyCryptoScanner(kraken_client, crypto_config)
                        results = scanner.scan_penny_cryptos(max_price=1.0, top_n=10, min_runner_score=60)
                        
                        if 'crypto_custom_tickers' not in st.session_state:
                            st.session_state.crypto_custom_tickers = []
                        
                        added = 0
                        for result in results[:5]:
                            ticker = result.get('pair', '')
                            if ticker and ticker not in st.session_state.crypto_custom_tickers:
                                st.session_state.crypto_custom_tickers.append(ticker)
                                added += 1
                        
                        if added > 0:
                            st.success(f"✅ Found and added {added} penny cryptos")
                        else:
                            st.info("No new penny cryptos found")
                except Exception as e:
                    st.error(f"Error scanning penny cryptos: {e}")
                    logger.error(f"Penny scan error: {e}", exc_info=True)
    
    st.markdown("---")


def configure_investment_settings(crypto_config, kraken_client=None):
    """Configure investment amount and risk settings"""
    st.markdown("#### 💰 Investment & Risk Configuration")
    
    # Get actual Kraken account balance if available
    actual_balance = 0.0
    if kraken_client:
        try:
            total_usd = kraken_client.get_total_balance_usd()
            actual_balance = total_usd
            st.success(f"✅ **Kraken Account Balance:** ${actual_balance:,.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch Kraken balance: {e}")
            st.warning("⚠️ Could not fetch live balance from Kraken")
    
    # Get default from config or session state
    if 'crypto_quick_capital' in st.session_state:
        default_capital = st.session_state.crypto_quick_capital
    else:
        default_capital = getattr(crypto_config, 'TOTAL_CAPITAL', 1000.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Investment Settings**")
        
        # Investment amount (not total balance - this is what they want to allocate)
        # Dynamic min_value based on actual balance (for small accounts)
        min_investment = 10.0  # Absolute minimum $10
        if actual_balance > 0 and actual_balance < 100:
            # If balance is less than $100, allow investing as little as $10
            max_investment = max(actual_balance, 100.0)
            step_size = 10.0
            recommended_min = min(50.0, actual_balance * 0.5)  # Recommend 50% of balance or $50
        else:
            max_investment = max(100000.0, actual_balance) if actual_balance > 0 else 100000.0
            step_size = 50.0
            recommended_min = 100.0
        
        # Ensure default value is within valid range
        if actual_balance > 0:
            safe_default = max(min_investment, min(default_capital, actual_balance))
        else:
            safe_default = max(min_investment, default_capital)
        
        custom_capital = st.number_input(
            "Amount to Invest ($)",
            min_value=min_investment,
            max_value=max_investment,
            value=safe_default,
            step=step_size,
            key="crypto_invest_capital",
            help=f"How much capital to allocate for trading (Available: ${actual_balance:,.2f})" if actual_balance > 0 else "How much capital to allocate for trading"
        )
        
        # Validate against actual balance
        if actual_balance > 0 and custom_capital > actual_balance:
            st.error(f"⚠️ Investment amount (${custom_capital:,.2f}) exceeds your Kraken balance (${actual_balance:,.2f})")
            custom_capital = actual_balance
        
        # Warning for very small amounts
        if custom_capital < 50:
            st.warning(f"⚠️ Very small investment amount (${custom_capital:.2f}). Consider depositing more for better trading opportunities.")
        
        # Max position size percentage
        max_position_pct = st.slider(
            "Max Position Size (% of capital)",
            min_value=1.0,
            max_value=20.0,
            value=getattr(crypto_config, 'MAX_POSITION_SIZE_PCT', 5.0),
            step=0.5,
            key="crypto_max_position_pct",
            help="Maximum % of capital to risk per trade"
        )
        
        # Calculate max position in dollars
        max_position_usd = custom_capital * (max_position_pct / 100)
        st.metric("Max Position Size", f"${max_position_usd:,.2f}")
    
    with col2:
        st.markdown("**⚠️ Risk Management**")
        
        # Risk per trade
        risk_per_trade_pct = st.slider(
            "Risk Per Trade (% of capital)",
            min_value=0.5,
            max_value=5.0,
            value=getattr(crypto_config, 'RISK_PER_TRADE_PCT', 2.0),
            step=0.25,
            key="crypto_risk_per_trade",
            help="Maximum % of capital at risk per trade"
        )
        
        # Default stop loss
        default_stop_loss = st.slider(
            "Default Stop Loss (%)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="crypto_default_stop",
            help="Default stop loss percentage"
        )
        
        # Calculate max loss in dollars
        max_loss_usd = custom_capital * (risk_per_trade_pct / 100)
        st.metric("Max Loss Per Trade", f"${max_loss_usd:,.2f}", delta=f"-{risk_per_trade_pct}%", delta_color="inverse")
    
    # Save to session state (without causing rerun)
    st.session_state.crypto_quick_capital = custom_capital
    st.session_state.crypto_quick_max_position_pct = max_position_pct
    st.session_state.crypto_quick_risk_pct = risk_per_trade_pct
    st.session_state.crypto_quick_default_stop = default_stop_loss
    st.session_state.crypto_actual_balance = actual_balance
    
    st.divider()
    
    # Safety guidelines
    with st.expander("🛡️ Risk Management Guidelines", expanded=False):
        st.markdown(f"""
        ### Recommended Settings for Beginners:
        
        **Investment Amount: ${custom_capital:,.2f}**
        {f"**Kraken Balance: ${actual_balance:,.2f}**" if actual_balance > 0 else ""}
        - **Start with:** $100-500 for learning
        - **Position size:** ${max_position_usd:,.2f} ({max_position_pct}% max)
        - **Risk per trade:** ${max_loss_usd:,.2f} ({risk_per_trade_pct}% max)
        
        ### Risk Management Rules:
        1. **Never risk more than 2% per trade**
        2. **Use stop losses on EVERY trade**
        3. **Maintain 2:1 minimum reward:risk ratio**
        4. **Start with smallest position sizes ($20-50)**
        5. **Increase size only after 10+ profitable trades**
        
        ### Position Sizing Formula:
        ```
        Position Size = (Capital * Risk%) / Stop Loss%
        Example: ($1000 * 2%) / 2% = $1000 max position
        But limited to {max_position_pct}% = ${max_position_usd:,.2f}
        ```
        
        ### Crypto-Specific Risks:
        - **Volatility:** Crypto moves 10x faster than stocks
        - **24/7 Trading:** Markets never close, gaps can happen
        - **Liquidity:** Some pairs have wide spreads
        - **No Paper Trading:** Every trade uses real money
        - **Security:** Store funds on exchange only for active trading
        """)


def render_quick_trade_tab(kraken_client, crypto_config):
    """Main function to render enhanced quick trade tab"""
    display_quick_trade_header()
    
    # Safety warning
    st.warning("""
    ⚠️ **LIVE TRADING WARNING**
    
    - Kraken has **NO PAPER TRADING MODE**
    - All trades use **REAL MONEY**
    - Start with **small positions** ($20-50 for learning)
    - Always use **stop losses**
    - Never risk more than **2% per trade**
    """)
    
    if hasattr(crypto_config, 'TEST_MODE') and crypto_config.TEST_MODE:
        st.info("✅ **TEST MODE ENABLED**: Extra safety checks active")
    else:
        st.error("⚠️ **TEST MODE DISABLED**: Enable in config for extra safety")
    
    st.divider()
    
    # Create tabs for different sections
    setup_tab, tickers_tab, trade_tab, monitor_tab = st.tabs([
        "⚙️ Setup & Config",
        "📊 Ticker Management", 
        "⚡ Execute Trade",
        "📈 AI Trade Monitor"
    ])
    
    with setup_tab:
        st.markdown("### ⚙️ Configuration & Risk Settings")
        
        # Investment & Risk Configuration (pass kraken_client to fetch balance)
        configure_investment_settings(crypto_config, kraken_client)
        
        # Quick strategy selector
        st.markdown("---")
        display_strategy_quick_selector()
        
        # Strategy tips
        with st.expander("💡 Strategy Tips & Guidelines", expanded=False):
            st.markdown("""
            ### ⚡ Scalping Mode (5m-15m)
            - **Target:** 1-3% quick profits
            - **Stop:** Tight 0.5-1.5%
            - **Best for:** High liquidity pairs (BTC, ETH)
            - **Time commitment:** Active monitoring required
            
            ### 🚀 Momentum Mode (15m-1h)
            - **Target:** 3-7% on strong moves
            - **Stop:** Medium 1.5-3%
            - **Best for:** Trending markets
            - **Time commitment:** Check every 30-60 minutes
            
            ### 📊 Swing Mode (1h-4h)
            - **Target:** 5-12% over 1-3 days
            - **Stop:** Wider 3-5%
            - **Best for:** Established trends
            - **Time commitment:** Check 2-3 times daily
            
            ### 🎯 General Tips
            - **Always use stop losses** - no exceptions!
            - **Start small** - test with $20-50 positions
            - **Follow R:R ratio** - minimum 2:1 (reward:risk)
            - **Tight spreads** - check bid/ask before entering
            - **Volume matters** - higher volume = better fills
            - **Avoid weekends** - lower liquidity, wider spreads
            - **Set alerts** - don't stare at charts all day
            """)
    
    with tickers_tab:
        st.markdown("### 📊 Ticker Management & Scanner Integration")
        
        # Scanner integration buttons
        integrate_scanners(kraken_client, crypto_config)
        
        # Manual ticker management
        manage_custom_tickers()
        
        # Display ticker selection for quick access
        st.markdown("---")
        if st.session_state.get('crypto_custom_tickers'):
            st.markdown("**🎯 Quick Select Ticker for Trade:**")
            selected_ticker = st.selectbox(
                "Select from your tickers",
                options=st.session_state.crypto_custom_tickers,
                key="quick_select_ticker"
            )
            
            if st.button("➡️ Use This Ticker in Trade Tab", use_container_width=True):
                st.session_state.crypto_quick_trade_pair = selected_ticker
                st.success(f"Selected {selected_ticker} - Go to 'Execute Trade' tab")
        else:
            st.info("Add tickers manually or import from scanners above")
    
    with trade_tab:
        st.markdown("### ⚡ Execute Trade")
        
        # Use custom ticker list if available
        if st.session_state.get('crypto_custom_tickers'):
            # Override crypto_config watchlist temporarily
            original_watchlist = getattr(crypto_config, 'CRYPTO_WATCHLIST', [])
            crypto_config.CRYPTO_WATCHLIST = st.session_state.crypto_custom_tickers
        
        # Trade setup
        display_trade_setup(kraken_client, crypto_config)
        
        # Restore original watchlist
        if st.session_state.get('crypto_custom_tickers'):
            crypto_config.CRYPTO_WATCHLIST = original_watchlist
    
    with monitor_tab:
        st.markdown("### 📈 AI Trade Monitoring & Management")
        
        # Initialize AI reviewer if needed
        if 'ai_trade_reviewer' not in st.session_state or st.session_state.ai_trade_reviewer is None:
            st.info("AI Trade Monitor not initialized. Execute a trade to start monitoring.")
        else:
            reviewer = st.session_state.ai_trade_reviewer
            
            # Display active monitors
            if not reviewer.active_monitors:
                st.info("📊 No active trades being monitored")
                
                # Show trade history
                if 'trade_history' in st.session_state and st.session_state.trade_history:
                    st.markdown("---")
                    st.markdown("### 📜 Trade History")
                    
                    df = pd.DataFrame(st.session_state.trade_history)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp', ascending=False)
                    
                    st.dataframe(df, use_container_width=True)
            else:
                st.success(f"📊 Monitoring {len(reviewer.active_monitors)} active trade(s)")
                
                # Refresh button
                if st.button("🔄 Refresh All Trades", use_container_width=True):
                    st.rerun()
                
                st.divider()
                
                # Display each monitored trade
                for trade_id, monitor_data in reviewer.active_monitors.items():
                    pair = monitor_data['pair']
                    
                    with st.expander(f"📊 {pair} - Order {trade_id[:8]}...", expanded=True):
                        # Get current price
                        try:
                            ticker = kraken_client.get_ticker_data(pair)
                            current_price = ticker['last_price'] if ticker else monitor_data['entry_price']
                        except:
                            current_price = monitor_data['entry_price']
                        
                        # Calculate P&L
                        side = monitor_data['side']
                        entry_price = monitor_data['entry_price']
                        
                        if side == 'BUY':
                            pnl_pct = (current_price - entry_price) / entry_price * 100
                            pnl_usd = (current_price - entry_price) * monitor_data['volume']
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price * 100
                            pnl_usd = (entry_price - current_price) * monitor_data['volume']
                        
                        # Display trade info
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        with col_m1:
                            st.metric("Entry Price", f"${entry_price:,.2f}")
                        
                        with col_m2:
                            st.metric("Current Price", f"${current_price:,.2f}")
                        
                        with col_m3:
                            pnl_color = "normal" if pnl_pct >= 0 else "inverse"
                            st.metric("P&L %", f"{pnl_pct:+.2f}%", delta_color=pnl_color)
                        
                        with col_m4:
                            st.metric("P&L $", f"${pnl_usd:+,.2f}", delta_color=pnl_color)
                        
                        # Stop Loss and Take Profit levels
                        col_sl, col_tp = st.columns(2)
                        
                        with col_sl:
                            st.markdown(f"🛑 **Stop Loss:** ${monitor_data['stop_loss']:,.2f}")
                            sl_distance = abs(current_price - monitor_data['stop_loss']) / current_price * 100
                            st.caption(f"Distance: {sl_distance:.2f}%")
                        
                        with col_tp:
                            st.markdown(f"🎯 **Take Profit:** ${monitor_data['take_profit']:,.2f}")
                            tp_distance = abs(monitor_data['take_profit'] - current_price) / current_price * 100
                            st.caption(f"Distance: {tp_distance:.2f}%")
                        
                        # Get AI recommendation
                        col_ai_check, col_actions = st.columns([2, 1])
                        
                        with col_ai_check:
                            if st.button(f"🤖 Get AI Recommendation", key=f"ai_check_{trade_id}"):
                                with st.spinner("AI analyzing trade..."):
                                    ai_recommendation = reviewer.check_trade_status(
                                        trade_id=trade_id,
                                        current_price=current_price,
                                        market_data=ticker if 'ticker' in locals() else None
                                    )
                                    
                                    st.session_state[f'ai_rec_{trade_id}'] = ai_recommendation
                                    st.rerun()
                        
                        # Display AI recommendation if available
                        if f'ai_rec_{trade_id}' in st.session_state:
                            rec = st.session_state[f'ai_rec_{trade_id}']
                            
                            st.divider()
                            
                            # Action badge
                            action = rec['action']
                            if action == 'HOLD':
                                st.info(f"💼 **AI Recommendation:** {action}")
                            elif action in ['TAKE_PARTIAL', 'CLOSE_NOW']:
                                st.warning(f"⚠️ **AI Recommendation:** {action}")
                            elif action in ['ADD_POSITION', 'TIGHTEN_STOP']:
                                st.success(f"✅ **AI Recommendation:** {action}")
                            else:
                                st.info(f"📊 **AI Recommendation:** {action}")
                            
                            st.markdown(f"**Confidence:** {rec['confidence']:.1f}%")
                            st.markdown(f"**Reasoning:** {rec['reasoning'][:200]}...")
                            
                            # Action parameters
                            if rec['parameters']:
                                st.markdown("**Parameters:**")
                                for key, value in rec['parameters'].items():
                                    st.markdown(f"- {key}: {value}")
                            
                            # Action buttons based on recommendation
                            if action == 'TAKE_PARTIAL':
                                partial_pct = rec['parameters'].get('partial_pct', 50)
                                if st.button(f"💰 Take {partial_pct}% Profit", key=f"partial_{trade_id}"):
                                    st.info(f"Partial profit taking would be executed here")
                            
                            elif action == 'CLOSE_NOW':
                                if st.button(f"🚪 Close Position Now", key=f"close_{trade_id}", type="primary"):
                                    st.warning(f"Position closure would be executed here")
                            
                            elif action == 'TIGHTEN_STOP':
                                new_stop = rec['parameters'].get('new_stop')
                                if new_stop and st.button(f"🛡️ Move Stop to ${new_stop:,.2f}", key=f"tighten_{trade_id}"):
                                    st.info(f"Stop loss adjustment would be executed here")
                        
                        # Manual actions
                        with col_actions:
                            if st.button("🚪 Close", key=f"manual_close_{trade_id}"):
                                st.session_state[f'confirm_close_{trade_id}'] = True
                                st.rerun()
                        
                        # Close confirmation
                        if st.session_state.get(f'confirm_close_{trade_id}'):
                            st.warning("⚠️ Confirm manual close?")
                            col_c1, col_c2 = st.columns(2)
                            
                            with col_c1:
                                if st.button("✅ Yes, Close", key=f"yes_close_{trade_id}"):
                                    # Stop monitoring
                                    reviewer.stop_monitoring(trade_id)
                                    st.success("Trade closed and monitoring stopped")
                                    del st.session_state[f'confirm_close_{trade_id}']
                                    st.rerun()
                            
                            with col_c2:
                                if st.button("❌ Cancel", key=f"no_close_{trade_id}"):
                                    del st.session_state[f'confirm_close_{trade_id}']
                                    st.rerun()


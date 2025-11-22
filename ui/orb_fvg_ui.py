"""
15-Minute ORB + FVG Strategy UI Component

User interface for the Reddit-inspired ORB+FVG trading strategy.
Displays signals, charts, and trade parameters.
Integrated with Discord alerts and trade journaling.
"""

import streamlit as st
from loguru import logger
from typing import List, Optional
from datetime import datetime

from services.orb_fvg_strategy import ORBFVGStrategy, ORBFVGSignal
from services.orb_fvg_alerts import create_orb_fvg_alert_manager


def render_orb_fvg_scanner():
    """Render the ORB+FVG strategy scanner UI"""
    
    st.markdown("### 📊 15-Min ORB + FVG Strategy Scanner")
    st.markdown("""
    **Strategy Overview** (Based on r/tradingmillionaires $2K payout):
    - ✅ **Opening Range**: First 15 minutes (9:30-9:45 AM)
    - ✅ **Fair Value Gaps**: Price inefficiencies for entry confirmation
    - ✅ **Risk/Reward**: 1.5-2R targets with tight stops
    - ✅ **Trading Window**: 9:45 AM - 11:00 AM (best results)
    - ✅ **Once Per Day**: Maximum one trade per ticker per day
    
    **Why This Works:**
    - High win rate (60-70%) with proper execution
    - Clear entry/exit rules (no guessing)
    - PDT-safe (one trade per day per ticker)
    - Morning momentum capture
    """)
    
    st.divider()
    
    # Configuration section
    with st.expander("⚙️ Strategy Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            orb_minutes = st.number_input(
                "ORB Period (minutes)",
                min_value=5,
                max_value=30,
                value=15,
                help="Opening range period (default: 15 min)"
            )
        
        with col2:
            target_rr = st.number_input(
                "Target R:R Ratio",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.5,
                help="Risk/Reward ratio (default: 2.0)"
            )
        
        with col3:
            min_volume = st.number_input(
                "Min Volume Ratio",
                min_value=1.0,
                max_value=5.0,
                value=1.5,
                step=0.5,
                help="Minimum volume vs average (default: 1.5x)"
            )
    
    # Ticker input section
    st.subheader("🎯 Scan for Signals")
    
    col_input1, col_input2 = st.columns([3, 1])
    
    with col_input1:
        # Use multiselect with 'Select All' button (per user preference)
        if 'watchlist_tickers' in st.session_state and st.session_state.watchlist_tickers:
            default_tickers = st.session_state.watchlist_tickers[:10]  # First 10
        else:
            default_tickers = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']
        
        # Get available tickers for multiselect
        if 'watchlist_tickers' in st.session_state and st.session_state.watchlist_tickers:
            available_tickers = st.session_state.watchlist_tickers
        else:
            available_tickers = [
                'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'GOOGL', 
                'AMZN', 'META', 'PLTR', 'SOFI', 'RIVN', 'MARA', 'RIOT'
            ]
        
        selected_tickers = st.multiselect(
            "Select Tickers to Scan",
            options=available_tickers,
            default=default_tickers,
            help="Choose tickers from your watchlist or popular stocks"
        )
        
        # Select All / Clear All buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("✅ Select All", key="orb_select_all"):
                st.session_state.orb_selected_tickers = available_tickers
                st.rerun()
        with col_btn2:
            if st.button("❌ Clear All", key="orb_clear_all"):
                st.session_state.orb_selected_tickers = []
                st.rerun()
    
    with col_input2:
        st.write("")
        st.write("")
        scan_button = st.button("🔍 Scan for Setups", type="primary", use_container_width=True)
    
    # Scan button action
    if scan_button:
        if not selected_tickers:
            st.error("⚠️ Please select at least one ticker to scan")
            return
        
        with st.status(f"🔍 Scanning {len(selected_tickers)} tickers for ORB+FVG setups...", expanded=True) as status:
            try:
                st.write("Initializing ORB+FVG strategy...")
                
                # Initialize strategy
                strategy = ORBFVGStrategy(
                    orb_minutes=orb_minutes,
                    min_volume_ratio=min_volume,
                    target_rr=target_rr
                )
                
                st.write(f"Scanning {len(selected_tickers)} tickers...")
                
                # Scan all tickers
                signals = strategy.scan_multiple_tickers(selected_tickers)
                
                status.update(
                    label=f"✅ Scan complete! Found {len(signals)} signal(s)",
                    state="complete"
                )
                
                if signals:
                    st.success(f"🎯 Found {len(signals)} ORB+FVG trading opportunity(s)!")
                    
                    # Initialize alert manager for Discord + journaling
                    try:
                        alert_manager = create_orb_fvg_alert_manager()
                        
                        # Send Discord alerts for all signals
                        for signal in signals:
                            try:
                                alert_manager.send_signal_alert(signal)
                                logger.info(f"📢 Discord alert sent for {signal.symbol}")
                            except Exception as e:
                                logger.error(f"Error sending alert for {signal.symbol}: {e}")
                        
                        st.info(f"📢 Discord alerts sent for all {len(signals)} signal(s)!")
                        
                    except Exception as e:
                        logger.error(f"Error with alert manager: {e}")
                        st.warning("⚠️ Alerts may not have been sent. Check logs.")
                    
                    # Display each signal
                    for idx, signal in enumerate(signals, 1):
                        display_signal(signal, idx)
                else:
                    st.info("📭 No ORB+FVG setups found at this time")
                    st.markdown("""
                    **Why no signals?**
                    - ⏰ Market may not be in optimal trading window (9:45-11:00 AM)
                    - 📊 No breakouts above/below opening range yet
                    - 📉 Volume requirements not met
                    - ✅ Already traded these tickers today (one per day limit)
                    
                    **Tips:**
                    - Run this scan between 9:45 AM - 11:00 AM ET for best results
                    - Make sure you're scanning liquid stocks with good volume
                    - Check back every 15-30 minutes for new setups
                    """)
            
            except Exception as e:
                status.update(label="❌ Scan failed", state="error")
                st.error(f"Error: {e}")
                logger.error("ORB+FVG scan error: {}", str(e), exc_info=True)


def display_signal(signal: ORBFVGSignal, index: int):
    """Display a single ORB+FVG signal"""
    
    # Signal header with color
    signal_emoji = "🟢" if signal.signal_type == "LONG" else "🔴"
    signal_color = "green" if signal.signal_type == "LONG" else "red"
    
    with st.container():
        st.markdown(f"## {signal_emoji} Signal #{index}: {signal.symbol} - {signal.signal_type}")
        
        # Confidence and timestamp
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Confidence", f"{signal.confidence:.1f}%")
        with col_info2:
            st.caption(f"🕐 {signal.timestamp.strftime('%I:%M %p')}")
        
        st.divider()
        
        # Trade parameters
        st.markdown("### 💰 Trade Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Entry Price", f"${signal.entry_price:.2f}")
            st.caption(f"Current: ${signal.current_price:.2f}")
        
        with col2:
            profit_pct = (signal.reward_amount / signal.entry_price) * 100
            st.metric("Target", f"${signal.target_price:.2f}", f"+{profit_pct:.1f}%")
            st.caption(f"💰 ${signal.reward_amount:.2f}")
        
        with col3:
            loss_pct = (signal.risk_amount / signal.entry_price) * 100
            st.metric("Stop Loss", f"${signal.stop_loss:.2f}", f"-{loss_pct:.1f}%")
            st.caption(f"🛑 ${signal.risk_amount:.2f}")
        
        with col4:
            st.metric("Risk/Reward", f"{signal.risk_reward_ratio:.1f}R")
            potential_profit_pct = (signal.target_price - signal.entry_price) / signal.entry_price * 100
            st.caption(f"🎯 {potential_profit_pct:+.1f}%")
        
        st.divider()
        
        # Technical details
        st.markdown("### 📊 Technical Analysis")
        
        col_tech1, col_tech2 = st.columns(2)
        
        with col_tech1:
            st.markdown("**Opening Range Breakout:**")
            st.write(f"• ORB High: ${signal.orb_high:.2f}")
            st.write(f"• ORB Low: ${signal.orb_low:.2f}")
            st.write(f"• Range Size: {signal.orb_range_pct:.2f}%")
            st.write(f"• Breakout: {'Above High' if signal.signal_type == 'LONG' else 'Below Low'}")
        
        with col_tech2:
            st.markdown("**Fair Value Gap:**")
            if signal.fvg:
                st.write(f"• FVG Type: {signal.fvg.gap_type.title()}")
                st.write(f"• FVG Top: ${signal.fvg.top:.2f}")
                st.write(f"• FVG Bottom: ${signal.fvg.bottom:.2f}")
                st.write(f"• FVG Strength: {signal.fvg.strength:.0f}/100")
                st.write(f"• Aligned: {'✅ Yes' if signal.fvg_alignment else '❌ No'}")
            else:
                st.write("• No FVG detected")
                st.write("• Breakout-only signal")
                st.write("• Use tighter stops")
        
        # Volume confirmation
        st.markdown("**Volume Analysis:**")
        volume_status = "🟢 Strong" if signal.volume_ratio >= 2.0 else "🟡 Moderate" if signal.volume_ratio >= 1.5 else "🔴 Weak"
        st.write(f"• Volume Ratio: {signal.volume_ratio:.1f}x average")
        st.write(f"• Status: {volume_status}")
        
        st.divider()
        
        # Trading recommendations
        st.markdown("### 💡 Trading Recommendations")
        
        if signal.confidence >= 75:
            st.success("✅ **HIGH CONFIDENCE SETUP** - Strong signal with good confluence")
            position_size = "Standard position size (5-10% of capital)"
        elif signal.confidence >= 60:
            st.info("🟡 **MODERATE CONFIDENCE** - Good setup but consider reducing size")
            position_size = "Reduced position size (3-5% of capital)"
        else:
            st.warning("⚠️ **LOW CONFIDENCE** - Wait for better setup or skip")
            position_size = "Minimal size (1-3% of capital) or skip"
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("**Position Sizing:**")
            st.write(f"• {position_size}")
            st.write(f"• Risk per trade: 1-2% of account")
            if signal.fvg_alignment:
                st.write("• FVG aligned = higher confidence")
            else:
                st.write("• No FVG alignment = reduce size")
        
        with col_rec2:
            st.markdown("**Exit Strategy:**")
            st.write(f"• Target: ${signal.target_price:.2f} ({signal.risk_reward_ratio:.1f}R)")
            st.write(f"• Stop: ${signal.stop_loss:.2f} (hard stop)")
            st.write("• Scale out at 1R if desired")
            st.write("• Move stop to breakeven at 1R")
        
        # Action buttons
        st.divider()
        col_action1, col_action2, col_action3 = st.columns(3)
        
        with col_action1:
            if st.button(f"✅ Copy to Order Form", key=f"copy_orb_{signal.symbol}_{index}", use_container_width=True):
                # Prefill order form in session state
                st.session_state['orb_prefill_symbol'] = signal.symbol
                st.session_state['orb_prefill_side'] = 'BUY' if signal.signal_type == 'LONG' else 'SELL'
                st.session_state['orb_prefill_entry'] = signal.entry_price
                st.session_state['orb_prefill_stop'] = signal.stop_loss
                st.session_state['orb_prefill_target'] = signal.target_price
                
                # Log to journal (will update when actual trade executed)
                try:
                    from services.orb_fvg_alerts import create_orb_fvg_alert_manager
                    alert_manager = create_orb_fvg_alert_manager()
                    
                    # Pre-log trade intent (quantity=0 until actual execution)
                    trade_id = alert_manager.log_trade_entry(
                        signal=signal,
                        quantity=0,  # Will be updated when trade executes
                        broker="TRADIER"  # Update based on active broker
                    )
                    
                    st.session_state[f'orb_trade_id_{signal.symbol}'] = trade_id
                    st.success(f"✅ Copied {signal.symbol} to order form! Trade logged to journal.")
                    
                except Exception as e:
                    logger.error(f"Error logging trade: {e}")
                    st.success(f"✅ Copied {signal.symbol} to order form!")
        
        with col_action2:
            if st.button(f"📋 Copy Trade Plan", key=f"plan_orb_{signal.symbol}_{index}", use_container_width=True):
                # Create trade plan text
                trade_plan = f"""
📊 ORB+FVG Trade Plan - {signal.symbol}

Signal: {signal.signal_type}
Confidence: {signal.confidence:.1f}%

Entry: ${signal.entry_price:.2f}
Target: ${signal.target_price:.2f} ({signal.risk_reward_ratio:.1f}R)
Stop: ${signal.stop_loss:.2f}

Risk: ${signal.risk_amount:.2f}
Reward: ${signal.reward_amount:.2f}

ORB Range: ${signal.orb_low:.2f} - ${signal.orb_high:.2f}
FVG: {'Yes' if signal.fvg else 'No'}
Volume: {signal.volume_ratio:.1f}x

Time: {signal.timestamp.strftime('%I:%M %p')}
                """
                st.code(trade_plan, language=None)
                st.info("Copy the trade plan above to your notes/journal")
        
        with col_action3:
            if st.button(f"🔄 Refresh {signal.symbol}", key=f"refresh_orb_{signal.symbol}_{index}", use_container_width=True):
                st.rerun()
        
        st.markdown("---")


def render_orb_fvg_education():
    """Render educational section about ORB+FVG strategy"""
    
    with st.expander("📚 Learn: ORB + FVG Strategy", expanded=False):
        st.markdown("""
        ## What is the 15-Min ORB + FVG Strategy?
        
        This strategy combines two powerful concepts:
        
        ### 1️⃣ Opening Range Breakout (ORB)
        - **Definition**: The high and low of the first 15 minutes of trading
        - **Why it works**: Establishes key levels that define early market sentiment
        - **Breakout**: When price moves decisively above/below the range
        - **Volume**: Must have strong volume (1.5x+ average) for confirmation
        
        ### 2️⃣ Fair Value Gap (FVG)
        - **Definition**: Price gaps where candles don't overlap (inefficient price zones)
        - **Bullish FVG**: Gap up (Candle 1 high < Candle 3 low)
        - **Bearish FVG**: Gap down (Candle 1 low > Candle 3 high)
        - **Purpose**: Acts as support/resistance and entry confirmation
        
        ## How to Trade It
        
        ### Step 1: Mark the Opening Range (9:30-9:45 AM)
        - Watch the first 15 minutes after market open
        - Note the high and low of this period
        - Wait for range to be established
        
        ### Step 2: Wait for Breakout (9:45-11:00 AM)
        - Look for price to break above high (bullish) or below low (bearish)
        - Confirm with volume (1.5x+ average)
        - Check for FVG in breakout direction
        
        ### Step 3: Enter the Trade
        - **Entry**: At current price on breakout confirmation
        - **Stop Loss**: Opposite side of ORB (low for long, high for short)
        - **Target**: 1.5-2x your risk (R:R ratio)
        
        ### Step 4: Manage the Trade
        - Move stop to breakeven at 1R
        - Scale out at 1R and 2R if desired
        - Exit at target or stop loss (no discretion)
        
        ## Why This Strategy Works
        
        ✅ **High Win Rate**: 60-70% with proper execution  
        ✅ **Clear Rules**: No guessing or emotions  
        ✅ **PDT-Safe**: One trade per day per ticker  
        ✅ **Morning Momentum**: Captures early market moves  
        ✅ **Risk Management**: Defined stop loss and target  
        
        ## Risk Management Tips
        
        1. **Position Size**: Risk 1-2% of account per trade
        2. **Stop Loss**: ALWAYS use hard stops (no mental stops)
        3. **One Per Day**: Maximum one trade per ticker per day
        4. **Best Time**: 9:45 AM - 11:00 AM ET (highest success rate)
        5. **Avoid**: Low volume stocks, choppy markets, news events
        
        ## Example Trade
        
        **Scenario**: SPY on a bullish day
        
        - **ORB**: 9:30-9:45 AM → High: $450.00, Low: $449.00
        - **Breakout**: 9:50 AM → Price breaks above $450.00 with 2x volume
        - **FVG**: Bullish FVG detected at $449.80-$450.20
        - **Entry**: $450.10
        - **Stop**: $449.00 (ORB low)
        - **Risk**: $1.10
        - **Target**: $452.30 (2R = $2.20)
        - **Result**: Hit target for +$2.20 profit per share
        
        ## Common Mistakes to Avoid
        
        ❌ **Trading before 9:45 AM** - ORB not established yet  
        ❌ **Entering without volume** - False breakouts are common  
        ❌ **No stop loss** - Never trade without protection  
        ❌ **Overtrading** - One trade per ticker per day MAX  
        ❌ **Ignoring FVG** - Higher success rate with FVG confirmation  
        
        ## Resources
        
        - Original Reddit post: r/tradingmillionaires ($2K payout)
        - Best timeframe: 5-minute charts for entries
        - Tools needed: Volume indicator, FVG indicator (built-in here!)
        """)

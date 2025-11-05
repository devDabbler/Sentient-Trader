"""
Crypto Signal Generation UI
Display trading signals from crypto strategies with detailed analysis
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
from services.crypto_strategies import (
    VWAPEMAScalper, BollingerMeanReversion, EMAMomentumNudge,
    EMASwingTrader, MACDRSISwing, BollingerSqueezeBreakout,
    TradingSignal
)
from services.crypto_watchlist_manager import CryptoWatchlistManager



def get_all_crypto_strategies():
    """Get all available crypto strategies"""
    return {
        'VWAP+EMA Pullback (Scalping)': VWAPEMAScalper(),
        'Bollinger Mean Reversion (Scalping)': BollingerMeanReversion(),
        'EMA Momentum Nudge (Scalping)': EMAMomentumNudge(),
        '10/21 EMA Swing': EMASwingTrader(),
        'MACD+RSI Confirmation': MACDRSISwing(),
        'Bollinger Squeeze Breakout': BollingerSqueezeBreakout()
    }


def display_signal_generation_header():
    """Display header for signal generation"""
    st.markdown("### 🎯 Crypto Signal Generation")
    st.write("Generate trading signals using advanced crypto strategies")


def display_strategy_selector():
    """Display strategy selection controls"""
    strategies = get_all_crypto_strategies()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_strategies = st.multiselect(
            "Select Strategies to Run",
            options=list(strategies.keys()),
            default=list(strategies.keys())[:3],
            help="Choose one or more strategies to generate signals"
        )
    
    with col2:
        analyze_all = st.checkbox(
            "Analyze All Watchlist",
            value=False,
            help="Run on entire watchlist vs. single symbol"
        )
    
    return selected_strategies, analyze_all


def display_single_symbol_input():
    """Display input for single symbol analysis"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input(
            "Crypto Pair Symbol",
            value="BTC/USD",
            placeholder="e.g., BTC/USD, ETH/USD",
            help="Enter crypto pair to analyze"
        ).upper()
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            ['1m', '5m', '15m', '1h', '4h', '1d'],
            index=2,
            help="Chart timeframe for analysis"
        )
    
    return symbol, timeframe


def display_signal_card(signal: TradingSignal, index: int, manager: CryptoWatchlistManager = None):
    """Display a trading signal card"""
    
    # Color coding for signal type
    if signal.signal_type == 'BUY':
        signal_color = "🟢"
        signal_bg = "green"
    elif signal.signal_type == 'SELL':
        signal_color = "🔴"
        signal_bg = "red"
    else:
        signal_color = "⚪"
        signal_bg = "gray"
    
    # Risk color
    risk_colors = {
        'LOW': '🟢',
        'MEDIUM': '🟡',
        'HIGH': '🟠',
        'EXTREME': '🔴'
    }
    risk_emoji = risk_colors.get(signal.risk_level, '⚪')
    
    # Card title
    card_title = (
        f"#{index} | {signal.symbol} | "
        f"{signal_color} {signal.signal_type} | "
        f"Confidence: {signal.confidence:.1f}% | "
        f"{risk_emoji} {signal.risk_level} Risk | "
        f"R:R {signal.risk_reward_ratio:.2f}"
    )
    
    with st.expander(card_title, expanded=(index <= 3)):
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"**{signal_color} Signal: {signal.signal_type}**")
            st.metric("Entry Price", f"${signal.entry_price:,.2f}")
            st.metric("Strategy", signal.strategy)
        
        with col2:
            st.markdown("**Risk Management**")
            st.metric("Stop Loss", f"${signal.stop_loss:,.2f}")
            risk_amount = abs(signal.entry_price - signal.stop_loss)
            risk_pct = (risk_amount / signal.entry_price) * 100
            st.text(f"Risk: {risk_pct:.2f}%")
        
        with col3:
            st.markdown("**Profit Target**")
            st.metric("Take Profit", f"${signal.take_profit:,.2f}")
            profit_amount = abs(signal.take_profit - signal.entry_price)
            profit_pct = (profit_amount / signal.entry_price) * 100
            st.text(f"Potential: +{profit_pct:.2f}%")
        
        with col4:
            st.markdown("**Signal Quality**")
            st.metric("Confidence", f"{signal.confidence:.1f}%")
            st.metric("Risk/Reward", f"{signal.risk_reward_ratio:.2f}:1")
            st.metric("Risk Level", signal.risk_level)
        
        st.divider()
        
        # Reasoning
        st.markdown("**📊 Strategy Reasoning:**")
        st.info(signal.reasoning)
        
        # Technical indicators
        if signal.indicators:
            with st.expander("📈 Technical Indicators", expanded=False):
                ind_cols = st.columns(3)
                
                ind_items = list(signal.indicators.items())
                for i, (key, value) in enumerate(ind_items):
                    with ind_cols[i % 3]:
                        if isinstance(value, float):
                            st.text(f"{key}: {value:.4f}")
                        else:
                            st.text(f"{key}: {value}")
        
        # Position sizing calculator
        with st.expander("💰 Position Size Calculator", expanded=False):
            pcol1, pcol2 = st.columns(2)
            
            with pcol1:
                account_size = st.number_input(
                    "Account Size ($)",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=10000.0,
                    step=100.0,
                    key=f"account_{signal.symbol}_{index}"
                )
                
                risk_per_trade = st.slider(
                    "Risk Per Trade (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    key=f"risk_{signal.symbol}_{index}"
                )
            
            with pcol2:
                # Calculate position size
                risk_amount = account_size * (risk_per_trade / 100)
                stop_distance = abs(signal.entry_price - signal.stop_loss)
                
                if stop_distance > 0:
                    position_size_usd = (risk_amount / stop_distance) * signal.entry_price
                    position_size_crypto = risk_amount / stop_distance
                    
                    st.markdown("**Recommended Position:**")
                    st.success(f"**USD Value:** ${position_size_usd:,.2f}")
                    st.success(f"**Crypto Amount:** {position_size_crypto:.6f} {signal.symbol.split('/')[0]}")
                    
                    # Potential profit/loss
                    potential_profit = position_size_crypto * abs(signal.take_profit - signal.entry_price)
                    potential_loss = risk_amount
                    
                    st.markdown("**Expected Outcomes:**")
                    st.text(f"✅ Potential Profit: ${potential_profit:,.2f}")
                    st.text(f"❌ Potential Loss: ${potential_loss:,.2f}")
                else:
                    st.warning("Invalid stop loss distance")
        
        # Timestamp
        st.divider()
        st.text(f"Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Action buttons
        st.divider()
        bcol1, bcol2, bcol3 = st.columns(3)
        
        with bcol1:
            if st.button("⚡ Quick Trade", key=f"quick_trade_{signal.symbol}_{index}"):
                # Set up quick trade with this signal
                st.session_state.crypto_quick_trade_signal = signal
                st.session_state.active_crypto_tab = "Quick Trade"
                st.success("📋 Signal copied to Quick Trade tab!")
        
        with bcol2:
            if manager and st.button("💾 Save to Watchlist", key=f"save_wl_{signal.symbol}_{index}"):
                with st.spinner(f"Saving {signal.symbol}..."):
                    # Create opportunity data from signal
                    opp_data = {
                        'symbol': signal.symbol,
                        'current_price': signal.entry_price,
                        'strategy': signal.strategy.replace(' ', '_').upper(),
                        'score': signal.confidence,
                        'confidence_level': 'HIGH' if signal.confidence >= 75 else 'MEDIUM' if signal.confidence >= 60 else 'LOW',
                        'risk_level': signal.risk_level,
                        'reasoning': signal.reasoning
                    }
                    
                    success = manager.add_crypto(signal.symbol, opp_data)
                    if success:
                        st.success(f"✅ Added {signal.symbol} to watchlist!")
                    else:
                        st.error(f"❌ Failed to add to watchlist")
        
        with bcol3:
            if st.button("📋 Copy Signal", key=f"copy_{signal.symbol}_{index}"):
                signal_text = f"""
{signal.symbol} - {signal.signal_type} Signal
Strategy: {signal.strategy}
Entry: ${signal.entry_price:,.2f}
Stop Loss: ${signal.stop_loss:,.2f}
Take Profit: ${signal.take_profit:,.2f}
Risk/Reward: {signal.risk_reward_ratio:.2f}:1
Confidence: {signal.confidence:.1f}%
Reasoning: {signal.reasoning}
                """.strip()
                st.code(signal_text, language=None)
                st.success("📋 Signal text displayed above - copy manually")


def display_no_signals_message():
    """Display message when no signals are generated"""
    st.info("""
    🔍 **No signals generated**
    
    This could mean:
    - Current market conditions don't meet strategy criteria
    - The crypto pair doesn't have enough data
    - No clear technical setups are present
    
    Try:
    - Different timeframes (5m, 15m, 1h)
    - Different strategies
    - Other crypto pairs
    - Running on your entire watchlist
    """)


def display_strategy_comparison(signals: List[TradingSignal]):
    """Display comparison table of signals from different strategies"""
    if len(signals) <= 1:
        return
    
    st.markdown("### 📊 Strategy Comparison")
    
    comparison_data = []
    for signal in signals:
        comparison_data.append({
            'Strategy': signal.strategy,
            'Signal': signal.signal_type,
            'Entry': f"${signal.entry_price:,.2f}",
            'Stop': f"${signal.stop_loss:,.2f}",
            'Target': f"${signal.take_profit:,.2f}",
            'R:R': f"{signal.risk_reward_ratio:.2f}",
            'Confidence': f"{signal.confidence:.1f}%",
            'Risk': signal.risk_level
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Consensus analysis
    buy_signals = sum(1 for s in signals if s.signal_type == 'BUY')
    sell_signals = sum(1 for s in signals if s.signal_type == 'SELL')
    
    if buy_signals > sell_signals:
        consensus = "🟢 BULLISH"
        consensus_pct = (buy_signals / len(signals)) * 100
    elif sell_signals > buy_signals:
        consensus = "🔴 BEARISH"
        consensus_pct = (sell_signals / len(signals)) * 100
    else:
        consensus = "⚪ NEUTRAL"
        consensus_pct = 50.0
    
    avg_confidence = sum(s.confidence for s in signals) / len(signals)
    
    st.markdown("**📈 Consensus Analysis:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Consensus", consensus)
    
    with col2:
        st.metric("Agreement", f"{consensus_pct:.1f}%")
    
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")


def render_signal_generation_tab(manager: CryptoWatchlistManager = None, kraken_client = None):
    """Main function to render signal generation tab"""
    display_signal_generation_header()
    
    # Strategy info
    with st.expander("ℹ️ Available Strategies", expanded=False):
        st.markdown("""
        ### 🎯 Scalping Strategies (5-30 minutes)
        
        **1. VWAP + EMA Pullback**
        - Trade with VWAP direction, enter on pullbacks to EMA
        - Best for: Intraday momentum with mean reversion entries
        - Timeframe: 5m, 15m
        
        **2. Bollinger Mean Reversion**
        - Fade extreme moves back to middle band
        - Best for: Range-bound or choppy markets
        - Timeframe: 5m, 15m
        
        **3. EMA Momentum Nudge**
        - 1-minute candle patterns with EMA/VWAP filter
        - Best for: Ultra-fast scalping with strong momentum
        - Timeframe: 1m, 5m
        
        ### 📊 Swing Strategies (Hours to Days)
        
        **4. 10/21 EMA Swing**
        - Pullbacks to fast EMA with trend filter
        - Best for: Trending markets with clear direction
        - Timeframe: 1h, 4h
        
        **5. MACD + RSI Confirmation**
        - Combined momentum indicators
        - Best for: High-probability setups with double confirmation
        - Timeframe: 1h, 4h, 1d
        
        **6. Bollinger Squeeze Breakout**
        - Low volatility compression followed by expansion
        - Best for: Catching the start of new trends
        - Timeframe: 1h, 4h, 1d
        """)
    
    st.divider()
    
    # Strategy selector
    selected_strategies, analyze_all = display_strategy_selector()
    
    if not selected_strategies:
        st.warning("⚠️ Please select at least one strategy")
        return
    
    st.divider()
    
    # Symbol input or watchlist analysis
    if not analyze_all:
        symbol, timeframe = display_single_symbol_input()
        
        if st.button("🎯 Generate Signals", type="primary", use_container_width=True):
            if not symbol:
                st.warning("Please enter a symbol")
                return
            
            with st.spinner(f"Generating signals for {symbol}..."):
                st.info("🚧 Signal generation requires historical data from Kraken API")
                st.markdown("""
                **To enable signal generation:**
                1. Fetch OHLCV data from Kraken API
                2. Pass DataFrame to strategy.generate_signal()
                3. Display results below
                
                **Example integration needed in services/crypto_signal_generator.py**
                """)
    
    else:
        # Analyze entire watchlist
        if st.button("🎯 Analyze Watchlist", type="primary", use_container_width=True):
            if not manager:
                st.error("Watchlist manager not available")
                return
            
            try:
                watchlist = manager.get_all_cryptos()
                
                if not watchlist:
                    st.warning("Your watchlist is empty. Add some cryptos first!")
                    return
                
                with st.spinner(f"Analyzing {len(watchlist)} cryptos..."):
                    st.info(f"🔄 Processing {len(watchlist)} symbols with {len(selected_strategies)} strategies...")
                    
                    # This would integrate with actual data fetching
                    st.markdown("""
                    **Watchlist analysis flow:**
                    1. Iterate through each symbol in watchlist
                    2. Fetch OHLCV data for each
                    3. Run selected strategies
                    4. Collect and rank signals
                    5. Display best opportunities
                    
                    **Integration point:** services/crypto_signal_generator.py
                    """)
            
            except Exception as e:
                st.error(f"Error analyzing watchlist: {e}")
                logger.error(f"Watchlist analysis error: {e}", exc_info=True)
    
    # Example signal display (for demonstration)
    with st.expander("📋 Example Signal Output", expanded=False):
        st.markdown("""
        Once integrated with live data, signals will appear here in cards like this:
        
        **#1 | BTC/USD | 🟢 BUY | Confidence: 78.5% | 🟢 LOW Risk | R:R 2.5**
        - Entry: $45,250
        - Stop Loss: $44,980 (0.6% risk)
        - Take Profit: $45,925 (1.5% profit)
        - Strategy: VWAP+EMA Pullback
        - Reasoning: Price above VWAP | Pullback to EMA9 complete | Momentum confirming (MACD)
        """)

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
from services.freqtrade_signal_adapter import get_freqtrade_strategy_wrappers
from services.crypto_watchlist_manager import CryptoWatchlistManager
from utils.crypto_pair_utils import normalize_crypto_pair



def get_all_crypto_strategies(kraken_client=None):
    """Get all available crypto strategies including Freqtrade strategies"""
    # Original strategies
    strategies = {
        'VWAP+EMA Pullback (Scalping)': VWAPEMAScalper(),
        'Bollinger Mean Reversion (Scalping)': BollingerMeanReversion(),
        'EMA Momentum Nudge (Scalping)': EMAMomentumNudge(),
        '10/21 EMA Swing': EMASwingTrader(),
        'MACD+RSI Confirmation': MACDRSISwing(),
        'Bollinger Squeeze Breakout': BollingerSqueezeBreakout()
    }
    
    # Add Freqtrade strategies if kraken_client is available
    if kraken_client:
        try:
            logger.debug(f"🔧 Loading Freqtrade strategies with kraken_client type: {type(kraken_client)}")
            freqtrade_strategies = get_freqtrade_strategy_wrappers(kraken_client)
            strategies.update(freqtrade_strategies)
            logger.info(f"✅ Added {len(freqtrade_strategies)} Freqtrade strategies to signal generator")
            logger.info(f"📊 Total strategies available: {len(strategies)}")
        except Exception as e:
            logger.error(f"❌ Could not load Freqtrade strategies: {e}", exc_info=True)
    else:
        logger.warning(f"⚠️ Kraken client is None - skipping Freqtrade strategies (only {len(strategies)} original strategies available)")
    
    return strategies


def display_signal_generation_header():
    """Display header for signal generation"""
    st.markdown("### 🎯 Crypto Signal Generation")
    st.write("Generate trading signals using advanced crypto strategies")


def display_strategy_selector(kraken_client=None):
    """Display strategy selection controls"""
    strategies = get_all_crypto_strategies(kraken_client)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show strategy count
        st.caption(f"💡 {len(strategies)} strategies available ({6 if kraken_client else 0} Freqtrade + 6 Original)")
        
        selected_strategies = st.multiselect(
            "Select Strategies to Run",
            options=list(strategies.keys()),
            default=list(strategies.keys())[:6],  # Default to first 6 (mix of both)
            help="Choose one or more strategies to generate signals. Freqtrade strategies are battle-tested from the crypto community."
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
    
    # Check if a symbol was passed from watchlist via session state
    default_symbol = "BTC/USD"
    if 'crypto_signal_symbol' in st.session_state and st.session_state.crypto_signal_symbol:
        default_symbol = st.session_state.crypto_signal_symbol
        # Clear the session state after using it
        st.session_state.crypto_signal_symbol = None
    
    with col1:
        symbol = st.text_input(
            "Crypto Pair Symbol",
            value=default_symbol,
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
                # Account size input with session state persistence
                account_key = f"account_{signal.symbol}_{index}"
                if account_key not in st.session_state:
                    st.session_state[account_key] = 10000.0
                
                account_size = st.number_input(
                    "Account Size ($)",
                    min_value=1.0,
                    max_value=1000000.0,
                    value=st.session_state[account_key],
                    step=1.0,
                    key=account_key
                )
                # Streamlit automatically updates session state for widgets with keys
                
                # Risk per trade slider with session state persistence
                risk_key = f"risk_{signal.symbol}_{index}"
                if risk_key not in st.session_state:
                    st.session_state[risk_key] = 2.0
                
                risk_per_trade = st.slider(
                    "Risk Per Trade (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=st.session_state[risk_key],
                    step=0.1,
                    key=risk_key
                )
                # Streamlit automatically updates session state for widgets with keys
            
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


def analyze_symbol_with_strategies(
    symbol: str,
    strategies: Dict,
    selected_strategy_names: List[str],
    kraken_client,
    timeframe: str = '15m'
) -> List[TradingSignal]:
    """
    Analyze a single symbol with selected strategies
    
    Args:
        symbol: Crypto pair symbol (e.g., 'BTC/USD')
        strategies: Dict of strategy name -> strategy instance
        selected_strategy_names: List of strategy names to run
        kraken_client: KrakenClient instance
        timeframe: Timeframe string (e.g., '15m', '1h')
    
    Returns:
        List of TradingSignal objects
    """
    signals = []
    
    if not kraken_client:
        logger.error("Kraken client not available")
        return signals
    
    try:
        # Normalize symbol
        normalized_symbol = normalize_crypto_pair(symbol)
        
        # Map timeframe to interval in minutes
        timeframe_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        interval = timeframe_map.get(timeframe, 15)
        
        # Fetch OHLCV data
        logger.info(f"Fetching OHLCV data for {normalized_symbol} (interval={interval})")
        ohlcv_list = kraken_client.get_ohlc_data(normalized_symbol, interval=interval)
        
        if not ohlcv_list or len(ohlcv_list) < 30:
            logger.warning(f"Insufficient OHLCV data for {normalized_symbol}: {len(ohlcv_list) if ohlcv_list else 0} candles")
            return signals
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_list)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in OHLCV data for {normalized_symbol}")
            return signals
        
        # Get current price (use last close or ticker)
        current_price = float(df['close'].iloc[-1])
        
        # Try to get more accurate current price from ticker
        try:
            ticker = kraken_client.get_ticker_data(normalized_symbol)
            if ticker and 'last_price' in ticker:
                current_price = float(ticker['last_price'])
        except Exception as e:
            logger.debug(f"Could not get ticker price for {normalized_symbol}: {e}")
        
        # Run each selected strategy
        for strategy_name in selected_strategy_names:
            if strategy_name not in strategies:
                logger.warning(f"Strategy {strategy_name} not found")
                continue
            
            strategy = strategies[strategy_name]
            
            try:
                # Generate signal
                signal = strategy.generate_signal(df, current_price)
                
                if signal:
                    # Update symbol to normalized format
                    signal.symbol = normalized_symbol
                    signals.append(signal)
                    logger.info(f"Generated {signal.signal_type} signal for {normalized_symbol} using {strategy_name} (confidence: {signal.confidence:.1f}%)")
                else:
                    logger.debug(f"No signal generated for {normalized_symbol} using {strategy_name}")
                    
            except Exception as e:
                logger.error(f"Error running {strategy_name} on {normalized_symbol}: {e}", exc_info=True)
                continue
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
    
    return signals


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
    st.dataframe(df, width='stretch')
    
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
    logger.info(f"🎯 render_signal_generation_tab called with kraken_client type: {type(kraken_client)}")
    logger.info(f"🎯 kraken_client is None: {kraken_client is None}")
    if kraken_client:
        logger.info(f"🎯 kraken_client object: {kraken_client}")
    
    display_signal_generation_header()
    
    # Strategy info
    with st.expander("ℹ️ Available Strategies", expanded=False):
        st.markdown("""
        ### 🔥 Freqtrade Strategies (Battle-Tested)
        
        **1. EMA Crossover + Heikin Ashi**
        - EMA 20/50/100 crossovers with Heikin Ashi confirmation
        - Best for: Trending markets with clear momentum
        - Timeframe: 5m, 15m
        
        **2. RSI + Stochastic + Hammer**
        - Oversold signals with Bollinger Bands and hammer candlestick
        - Best for: Mean reversion entries at support
        - Timeframe: 5m, 15m
        
        **3. Fisher RSI Multi-Indicator**
        - Fisher RSI with MFI, Stochastic, and EMA confirmation
        - Best for: High-probability oversold bounces
        - Timeframe: 5m, 15m
        
        **4. MACD + Volume + RSI**
        - MACD crossovers with volume spikes and Fisher RSI
        - Best for: Momentum trades with volume confirmation
        - Timeframe: 5m, 15m
        
        **5. Aggressive Scalping**
        - Fast EMA crosses with tight stops (1-3% targets)
        - Best for: Quick scalps in volatile markets
        - Timeframe: 1m, 5m
        
        **6. ORB + Fair Value Gap**
        - Opening Range Breakout with Fair Value Gap confirmation
        - Best for: Intraday breakout trades
        - Timeframe: 1m, 5m
        
        ---
        
        ### 🎯 Original Scalping Strategies (5-30 minutes)
        
        **7. VWAP + EMA Pullback**
        - Trade with VWAP direction, enter on pullbacks to EMA
        - Best for: Intraday momentum with mean reversion entries
        - Timeframe: 5m, 15m
        
        **8. Bollinger Mean Reversion**
        - Fade extreme moves back to middle band
        - Best for: Range-bound or choppy markets
        - Timeframe: 5m, 15m
        
        **9. EMA Momentum Nudge**
        - 1-minute candle patterns with EMA/VWAP filter
        - Best for: Ultra-fast scalping with strong momentum
        - Timeframe: 1m, 5m
        
        ### 📊 Original Swing Strategies (Hours to Days)
        
        **10. 10/21 EMA Swing**
        - Pullbacks to fast EMA with trend filter
        - Best for: Trending markets with clear direction
        - Timeframe: 1h, 4h
        
        **11. MACD + RSI Confirmation**
        - Combined momentum indicators
        - Best for: High-probability setups with double confirmation
        - Timeframe: 1h, 4h, 1d
        
        **12. Bollinger Squeeze Breakout**
        - Low volatility compression followed by expansion
        - Best for: Catching the start of new trends
        - Timeframe: 1h, 4h, 1d
        """)
    
    st.divider()
    
    # Strategy selector
    selected_strategies, analyze_all = display_strategy_selector(kraken_client)
    
    if not selected_strategies:
        st.warning("⚠️ Please select at least one strategy")
        return
    
    st.divider()
    
    # Symbol input or watchlist analysis
    if not analyze_all:
        symbol, timeframe = display_single_symbol_input()
        
        # Check if we should auto-trigger signal generation (from watchlist)
        auto_generate = st.session_state.get('crypto_auto_generate_signal', False)
        logger.info(f"🔍 Signal Generator: auto_generate = {auto_generate}")
        logger.info(f"🔍 Signal Generator: symbol = {symbol}")
        if auto_generate:
            logger.info(f"✅ Auto-trigger detected! Clearing flag and generating signals...")
            st.session_state.crypto_auto_generate_signal = False  # Clear flag
        
        # Generate signals on button click OR auto-trigger
        if st.button("🎯 Generate Signals", type="primary", width='stretch') or auto_generate:
            if not symbol:
                st.warning("Please enter a symbol")
                return
            
            if not kraken_client:
                st.error("Kraken client not available. Cannot fetch market data.")
                return
            
            with st.spinner(f"Generating signals for {symbol}..."):
                try:
                    # Get all strategies including Freqtrade
                    all_strategies = get_all_crypto_strategies(kraken_client)
                    
                    # Analyze symbol with selected strategies
                    signals = analyze_symbol_with_strategies(
                        symbol=symbol,
                        strategies=all_strategies,
                        selected_strategy_names=selected_strategies,
                        kraken_client=kraken_client,
                        timeframe=timeframe
                    )
                    
                    # Display results
                    if signals:
                        st.success(f"✅ Generated {len(signals)} signal(s) for {symbol}")
                        st.divider()
                        
                        # Sort by confidence (descending)
                        signals.sort(key=lambda x: x.confidence, reverse=True)
                        
                        # Display signals
                        for idx, signal in enumerate(signals, 1):
                            display_signal_card(signal, idx, manager)
                        
                        # Strategy comparison if multiple signals
                        if len(signals) > 1:
                            st.divider()
                            display_strategy_comparison(signals)
                    else:
                        display_no_signals_message()
                        
                except Exception as e:
                    st.error(f"Error generating signals: {e}")
                    logger.error(f"Signal generation error: {e}", exc_info=True)
    
    else:
        # Initialize session state for watchlist signals if not exists
        signals_key = "crypto_watchlist_signals"
        if signals_key not in st.session_state:
            st.session_state[signals_key] = []
        
        # Analyze entire watchlist
        if st.button("🎯 Analyze Watchlist", type="primary", width='stretch'):
            if not manager:
                st.error("Watchlist manager not available")
                return
            
            if not kraken_client:
                st.error("Kraken client not available. Cannot fetch market data.")
                return
            
            try:
                watchlist = manager.get_all_cryptos()
                
                if not watchlist:
                    st.warning("Your watchlist is empty. Add some cryptos first!")
                    return
                
                # Get all strategies including Freqtrade
                all_strategies = get_all_crypto_strategies(kraken_client)
                
                # Determine timeframe (default to 15m for watchlist analysis)
                timeframe = '15m'
                
                with st.spinner(f"Analyzing {len(watchlist)} cryptos with {len(selected_strategies)} strategies..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_signals = []
                    processed = 0
                    
                    for crypto in watchlist:
                        symbol = crypto.get('symbol', '')
                        if not symbol:
                            continue
                        
                        processed += 1
                        progress = processed / len(watchlist)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {symbol} ({processed}/{len(watchlist)})...")
                        
                        # Analyze symbol
                        signals = analyze_symbol_with_strategies(
                            symbol=symbol,
                            strategies=all_strategies,
                            selected_strategy_names=selected_strategies,
                            kraken_client=kraken_client,
                            timeframe=timeframe
                        )
                        
                        all_signals.extend(signals)
                    
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
                    # Store signals in session state (convert to dict for storage)
                    signals_data = []
                    for signal in all_signals:
                        # Convert TradingSignal to dict for storage
                        signal_dict = {
                            'symbol': signal.symbol,
                            'strategy': signal.strategy,
                            'signal_type': signal.signal_type,
                            'confidence': signal.confidence,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'risk_reward_ratio': signal.risk_reward_ratio,
                            'reasoning': signal.reasoning,
                            'indicators': signal.indicators,
                            'timestamp': signal.timestamp.isoformat(),
                            'risk_level': signal.risk_level
                        }
                        signals_data.append(signal_dict)
                    
                    st.session_state[signals_key] = signals_data
                    
                    # Display results
                    if all_signals:
                        st.success(f"✅ Generated {len(all_signals)} signal(s) from {len(watchlist)} symbols")
                        st.rerun()  # Rerun to display signals
                    else:
                        st.warning("No signals generated from watchlist analysis. Market conditions may not meet strategy criteria.")
                        display_no_signals_message()
            
            except Exception as e:
                st.error(f"Error analyzing watchlist: {e}")
                logger.error(f"Watchlist analysis error: {e}", exc_info=True)
        
        # Display stored signals if they exist
        if st.session_state.get(signals_key):
            stored_signals = st.session_state[signals_key]
            
            # Convert back to TradingSignal objects
            all_signals = []
            for signal_dict in stored_signals:
                signal = TradingSignal(
                    symbol=signal_dict['symbol'],
                    strategy=signal_dict['strategy'],
                    signal_type=signal_dict['signal_type'],
                    confidence=signal_dict['confidence'],
                    entry_price=signal_dict['entry_price'],
                    stop_loss=signal_dict['stop_loss'],
                    take_profit=signal_dict['take_profit'],
                    risk_reward_ratio=signal_dict['risk_reward_ratio'],
                    reasoning=signal_dict['reasoning'],
                    indicators=signal_dict['indicators'],
                    timestamp=datetime.fromisoformat(signal_dict['timestamp']),
                    risk_level=signal_dict['risk_level']
                )
                all_signals.append(signal)
            
            if all_signals:
                st.divider()
                st.success(f"✅ Showing {len(all_signals)} signal(s) from last analysis")
                
                # Sort by confidence (descending)
                all_signals.sort(key=lambda x: x.confidence, reverse=True)
                
                # Display top signals
                st.markdown(f"### 📊 Top Signals (showing top {min(20, len(all_signals))})")
                
                for idx, signal in enumerate(all_signals[:20], 1):
                    display_signal_card(signal, idx, manager)
                
                # Strategy comparison if multiple signals
                if len(all_signals) > 1:
                    st.divider()
                    display_strategy_comparison(all_signals[:20])

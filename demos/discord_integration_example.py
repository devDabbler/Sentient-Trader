"""
Example: How to integrate Discord alerts into your app.py

This shows two ways to integrate Discord:
1. Add Discord tab to Streamlit app
2. Use Discord data in AI analysis
"""

import streamlit as st
from discord_ui_tab import render_discord_tab
from discord_alert_listener import create_discord_manager
from ai_trading_signals import AITradingSignalGenerator

# ============================================
# OPTION 1: Add Discord Tab to Streamlit App
# ============================================

def add_discord_tab_to_app():
    """
    Add this to your app.py to include Discord tab
    
    Add to your existing tabs like this:
    """
    
    # In your app.py, find where you define tabs:
    # tabs = st.tabs(["Stock Intelligence", "Strategy Advisor", "Signal Builder", ...])
    
    # Add Discord tab:
    tabs = st.tabs([
        "Stock Intelligence",
        "Strategy Advisor", 
        "Signal Builder",
        "Signal History",
        "💬 Discord Alerts",  # <-- New tab!
        # ... other tabs
    ])
    
    # Then in the tab section:
    with tabs[4]:  # Discord tab
        render_discord_tab()


# ============================================
# OPTION 2: Use Discord Data in AI Analysis
# ============================================

def example_ai_analysis_with_discord():
    """
    Example of using Discord alerts in AI signal generation
    """
    
    # Initialize Discord manager (do this once at app startup)
    discord_mgr = create_discord_manager()
    if discord_mgr:
        discord_mgr.start()
    
    # When analyzing a symbol
    symbol = "TSLA"
    
    # Get technical data (your existing code)
    technical_data = {
        'price': 250.50,
        'rsi': 65,
        'macd_signal': 'BULLISH',
        # ... other technical indicators
    }
    
    # Get news data (your existing code)
    news_data = [
        {'title': 'Tesla announces...', 'sentiment': 0.8},
        # ... other news
    ]
    
    # Get sentiment data (your existing code)
    sentiment_data = {
        'score': 0.7,
        'signals': ['Positive earnings', 'Strong momentum']
    }
    
    # Get Discord alerts for this symbol
    discord_data = None
    if discord_mgr and discord_mgr.is_running():
        symbol_alerts = discord_mgr.get_symbol_alerts(symbol, limit=10)
        discord_data = {
            'alerts': symbol_alerts
        }
    
    # Generate AI signal WITH Discord data
    generator = AITradingSignalGenerator()
    signal = generator.generate_signal(
        symbol=symbol,
        technical_data=technical_data,
        news_data=news_data,
        sentiment_data=sentiment_data,
        social_data=None,
        discord_data=discord_data,  # <-- Discord alerts included!
        account_balance=10000.0,
        risk_tolerance='MEDIUM'
    )
    
    if signal:
        print(f"Signal: {signal.signal}")
        print(f"Confidence: {signal.confidence}%")
        print(f"Discord Score: {signal.discord_score}/100")
        print(f"Reasoning: {signal.reasoning}")


# ============================================
# OPTION 3: Standalone Discord Alerts
# ============================================

def example_standalone_discord_alerts():
    """
    Use Discord alerts standalone without AI integration
    """
    
    # Create and start Discord manager
    discord_mgr = create_discord_manager()
    if not discord_mgr:
        print("Discord not configured")
        return
    
    discord_mgr.start()
    
    # Wait for some alerts...
    import time
    time.sleep(10)
    
    # Get all recent alerts
    alerts = discord_mgr.get_alerts(limit=50)
    print(f"Received {len(alerts)} alerts")
    
    for alert in alerts:
        print(f"{alert['symbol']} - {alert['alert_type']} @ ${alert['price']}")
    
    # Get alerts for specific symbol
    tsla_alerts = discord_mgr.get_symbol_alerts('TSLA', limit=10)
    print(f"\nTSLA alerts: {len(tsla_alerts)}")


# ============================================
# OPTION 4: Display Discord Alerts in UI
# ============================================

def example_display_discord_in_existing_tab():
    """
    Add Discord alerts to your existing Stock Intelligence tab
    """
    
    st.header("Stock Intelligence for TSLA")
    
    # Your existing analysis...
    
    # Add Discord alerts section
    st.divider()
    st.subheader("💬 Discord Alerts")
    
    if 'discord_manager' in st.session_state and st.session_state.discord_manager:
        discord_mgr = st.session_state.discord_manager
        
        if discord_mgr.is_running():
            alerts = discord_mgr.get_symbol_alerts('TSLA', limit=5)
            
            if alerts:
                st.success(f"Found {len(alerts)} Discord alerts for TSLA")
                
                for alert in alerts:
                    with st.expander(f"{alert['alert_type']} @ ${alert['price']}"):
                        st.write(f"**Time:** {alert['timestamp']}")
                        st.write(f"**Channel:** {alert['channel_name']}")
                        st.write(f"**Message:** {alert['reasoning']}")
                        
                        if alert['target']:
                            st.write(f"**Target:** ${alert['target']}")
                        if alert['stop_loss']:
                            st.write(f"**Stop Loss:** ${alert['stop_loss']}")
            else:
                st.info("No Discord alerts for TSLA yet")
        else:
            st.warning("Discord bot not running. Start it in Discord Alerts tab.")
    else:
        st.info("Discord integration not initialized")


# ============================================
# RECOMMENDED: Simple Integration Steps
# ============================================

"""
STEP 1: Add to your app.py imports
------------------------------------
from discord_ui_tab import render_discord_tab
from discord_alert_listener import create_discord_manager

STEP 2: Initialize Discord in main()
------------------------------------
# At the top of your main() function:
if 'discord_manager' not in st.session_state:
    st.session_state.discord_manager = None

STEP 3: Add Discord tab
------------------------------------
# In your tabs section:
tabs = st.tabs([
    "...",
    "💬 Discord Alerts",
    "..."
])

with tabs[X]:  # Replace X with appropriate index
    render_discord_tab()

STEP 4: Use in AI analysis (optional)
------------------------------------
# When calling generate_signal():
discord_data = None
if st.session_state.discord_manager:
    alerts = st.session_state.discord_manager.get_symbol_alerts(symbol, 10)
    discord_data = {'alerts': alerts}

signal = generator.generate_signal(
    ...,
    discord_data=discord_data,
    ...
)

That's it! 🎉
"""


if __name__ == "__main__":
    print("Discord Integration Examples")
    print("See code above for integration options")

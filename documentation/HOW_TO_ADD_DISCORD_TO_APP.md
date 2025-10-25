# How to Add Discord Tab to Your App

## Simple 3-Step Integration

### Step 1: Add Import at Top of app.py

Find the imports section at the top of your `app.py` and add:

```python
# Import Discord integration
from discord_ui_tab import render_discord_tab
```

### Step 2: Find Your Tabs Section

Look for where you create tabs in your app.py. It should look something like:

```python
tabs = st.tabs([
    "📊 Stock Intelligence",
    "🎯 Strategy Advisor",
    "📝 Generate Trading Signal",
    "📜 Signal History",
    # ... other tabs
])
```

### Step 3: Add Discord Tab

Add the Discord tab to your tabs list:

```python
tabs = st.tabs([
    "📊 Stock Intelligence",
    "🎯 Strategy Advisor",
    "📝 Generate Trading Signal",
    "📜 Signal History",
    "💬 Discord Alerts",  # <-- ADD THIS LINE
    # ... other tabs
])
```

Then add the corresponding tab content:

```python
# Find where you have: with tabs[0], with tabs[1], etc.
# Add this for your Discord tab (adjust index number as needed):

with tabs[4]:  # Change index to match position in your tabs list
    render_discord_tab()
```

## Complete Example

Here's what the tabs section might look like after adding Discord:

```python
import streamlit as st
# ... other imports ...
from discord_ui_tab import render_discord_tab  # ADD THIS

def main():
    st.title("Sentient Trader Platform")
    
    # Create tabs
    tabs = st.tabs([
        "📊 Stock Intelligence",
        "🎯 Strategy Advisor",
        "📝 Generate Trading Signal",
        "📜 Signal History",
        "💬 Discord Alerts",        # NEW TAB
        "📈 Advanced Analytics",
        "🔗 Tradier Integration",
        "🤖 LLM Strategy Analyzer"
    ])
    
    # Tab 0: Stock Intelligence
    with tabs[0]:
        # Your existing code
        pass
    
    # Tab 1: Strategy Advisor
    with tabs[1]:
        # Your existing code
        pass
    
    # Tab 2: Generate Trading Signal
    with tabs[2]:
        # Your existing code
        pass
    
    # Tab 3: Signal History
    with tabs[3]:
        # Your existing code
        pass
    
    # Tab 4: Discord Alerts (NEW!)
    with tabs[4]:
        render_discord_tab()
    
    # Continue with other tabs...
```

## Optional: Integrate Discord Data into AI Analysis

If you want Discord alerts to be included in your AI signal generation, find where you call `generate_signal()` and add:

```python
# Before generating signal, get Discord data
discord_data = None
if 'discord_manager' in st.session_state and st.session_state.discord_manager:
    if st.session_state.discord_manager.is_running():
        discord_alerts = st.session_state.discord_manager.get_symbol_alerts(symbol, limit=10)
        discord_data = {'alerts': discord_alerts}

# Then pass it to generate_signal
signal = generator.generate_signal(
    symbol=symbol,
    technical_data=technical_data,
    news_data=news_data,
    sentiment_data=sentiment_data,
    social_data=social_data,
    discord_data=discord_data,  # <-- ADD THIS
    account_balance=account_balance,
    risk_tolerance=risk_tolerance
)
```

## That's It! 🎉

After making these changes:

1. Save `app.py`
2. Restart your Streamlit app: `streamlit run app.py`
3. You'll see the new "💬 Discord Alerts" tab
4. Follow the in-tab instructions to configure your bot

## Need Help?

- **Quick Setup:** See `DISCORD_QUICK_START.md`
- **Full Guide:** See `DISCORD_INTEGRATION_GUIDE.md`
- **Examples:** See `discord_integration_example.py`

## Visual Reference

```
Your App Before:
┌─────────────────────────────────────────┐
│ Tab1 | Tab2 | Tab3 | Tab4 | Tab5       │
└─────────────────────────────────────────┘

Your App After:
┌─────────────────────────────────────────────────┐
│ Tab1 | Tab2 | Tab3 | Tab4 | 💬Discord | Tab5  │
└─────────────────────────────────────────────────┘
                              ↑
                           NEW TAB!
```

## Minimal Code Changes Required

✅ **1 line** to import: `from discord_ui_tab import render_discord_tab`
✅ **1 line** to add tab: `"💬 Discord Alerts"` in tabs list
✅ **2 lines** for tab content:
```python
with tabs[X]:
    render_discord_tab()
```

**Total: ~4 lines of code!**

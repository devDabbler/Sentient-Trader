# Discord Integration - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Discord Library
```powershell
pip install discord.py>=2.3.0
```

Or install all requirements:
```powershell
pip install -r requirements.txt
```

### Step 2: Get Your Bot Token

1. Go to https://discord.com/developers/applications
2. Click "New Application" → Name it "Trading Alert Bot"
3. Go to "Bot" tab → Click "Add Bot"
4. **IMPORTANT:** Enable "MESSAGE CONTENT INTENT" under Privileged Gateway Intents
5. Copy the bot token

### Step 3: Add Bot to Discord Server

1. Go to "OAuth2" → "URL Generator"
2. Select: `bot` scope
3. Select permissions: "Read Messages/View Channels", "Read Message History"
4. Copy URL and open in browser
5. Select your server and authorize

### Step 4: Get Channel ID

1. Enable Developer Mode: Discord Settings → Advanced → Developer Mode ✅
2. Right-click the channel you want to monitor
3. Click "Copy Channel ID"
4. Save this ID

**Your free channel from Discord link:**
- Channel ID: `1427896857274617939`
- Server ID: `658433759497945120`

### Step 5: Configure Environment

Edit your `.env` file:

```env
# Add these lines
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_CHANNEL_IDS=1427896857274617939
```

### Step 6: Run Your App

```powershell
streamlit run app.py
```

### Step 7: Start Discord Bot

1. Go to "💬 Discord Alerts" tab
2. Enter your bot token (or it will load from .env)
3. Add channel IDs
4. Click "▶️ Start Bot"
5. Wait for alerts!

## ✅ Verify It's Working

You should see:
- Bot status changes to "🟢 Running"
- Alerts appear in the dashboard as they come in
- Symbol analysis shows recent alerts

## 🎯 Integration Modes

### Mode 1: Standalone (View Only)
Just monitor alerts in the Discord tab - no AI integration needed.

### Mode 2: AI-Integrated (Recommended)
Discord alerts automatically included in your AI trading analysis!

To enable AI integration, Discord alerts are automatically passed when you run AI analysis on a symbol. The AI will see:
- Recent Discord alerts for that symbol
- Entry/exit signals from traders
- Discord sentiment (bullish/bearish)
- Price targets and stop losses

## 📊 Example Alert Format

Discord messages like these will be automatically parsed:

```
✅ ENTRY: $TSLA @ $250.50
Target: $265
SL: $245
High conviction play!

🚀 RUNNER: NVDA breaking out above $500

🔔 ALERT: Watch AAPL potential setup forming

❌ EXIT: Closed SPY @ $455, +2.5% profit
```

## 🔍 What Gets Parsed

- ✅ Ticker symbols ($TSLA, TSLA, ticker: TSLA)
- ✅ Entry prices (@ $250, price: 250, entry: 250)
- ✅ Targets (target: 265, TP: 265)
- ✅ Stop losses (stop loss: 245, SL: 245)
- ✅ Alert types (ENTRY, EXIT, RUNNER, ALERT, STOP)

## 🛠️ Troubleshooting

### Bot Won't Connect
- ✅ Check bot token is correct
- ✅ Verify MESSAGE CONTENT INTENT is enabled
- ✅ Confirm bot was added to server

### No Alerts Showing
- ✅ Verify channel ID is correct
- ✅ Check bot has permissions in that channel
- ✅ Try posting a test message: `$TSLA @ $250`

### Can't See Channel
- ✅ Bot needs "View Channel" permission
- ✅ Channel must be text channel (not voice)
- ✅ Bot must be member of the server

## 🎓 Next Steps

1. ✅ Monitor free channel to see alerts
2. ✅ Test AI analysis with Discord data
3. ✅ Add premium channels if you have access
4. ✅ Export alerts for your records
5. ✅ Customize alert parsing if needed

## 💡 Pro Tips

- **Don't blindly follow alerts** - Use as ONE data point among many
- **Combine with your analysis** - Technical, news, sentiment + Discord
- **Track alert performance** - Export regularly and review
- **Start conservative** - Test with small positions first
- **Monitor multiple channels** - Diversify your signal sources

## 📚 Full Documentation

For detailed setup and advanced features, see:
- `DISCORD_INTEGRATION_GUIDE.md` - Complete setup guide
- `discord_integration_example.py` - Code examples

## ⚠️ Important Notes

1. **Bot runs in background** - Continues monitoring even when you're not looking at the tab
2. **Alerts stored in memory** - Export regularly, they're cleared when you restart
3. **Rate limits apply** - Discord has limits, but normal usage is fine
4. **Your responsibility** - You make the trading decisions, not the bot
5. **Privacy** - Bot only sees channels it's added to

## 🎉 You're Ready!

Once you see the bot status turn green and alerts start appearing, you're all set! The Discord integration will enhance your trading analysis with real-time alerts from professional traders.

---

**Questions?** Check the logs in `trading_signals.log` or review the full guide in `DISCORD_INTEGRATION_GUIDE.md`

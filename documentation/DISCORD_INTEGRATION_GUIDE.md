# Discord Trading Alerts Integration Guide

This guide explains how to integrate Discord trading alerts into your AI Options Trader platform.

## Overview

The Discord integration allows you to:
- ✅ Monitor free and premium Discord channels for trading alerts
- ✅ Automatically parse ticker symbols, prices, targets, and stop losses
- ✅ Feed Discord alerts into your AI analysis for comprehensive decision-making
- ✅ View Discord alerts standalone in the Streamlit UI
- ✅ Track alert history and export to JSON/CSV

## Architecture

### Two Integration Modes:

1. **Standalone Mode**: View Discord alerts separately without AI analysis
2. **AI-Integrated Mode**: Discord alerts are included as a signal source in your AI trading analysis (alongside technical, news, and social sentiment)

## Setup Steps

### 1. Create a Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name (e.g., "Trading Alert Bot")
3. Go to the "Bot" tab and click "Add Bot"
4. Enable the following **Privileged Gateway Intents**:
   - ✅ MESSAGE CONTENT INTENT
   - ✅ SERVER MEMBERS INTENT (optional)
   - ✅ PRESENCE INTENT (optional)
5. Copy the **Bot Token** (you'll need this later)

### 2. Add Bot to Your Discord Server

1. Go to the "OAuth2" > "URL Generator" tab
2. Select scopes:
   - ✅ `bot`
3. Select bot permissions:
   - ✅ Read Messages/View Channels
   - ✅ Read Message History
4. Copy the generated URL and open it in your browser
5. Select the Discord server where you want to add the bot
6. Authorize the bot

### 3. Get Channel IDs

To monitor specific channels, you need their Channel IDs:

1. Enable Developer Mode in Discord:
   - User Settings > Advanced > Developer Mode (toggle ON)
2. Right-click on the channel you want to monitor
3. Click "Copy Channel ID"
4. Save these IDs for configuration

**Example Channel ID from your link:**
```
Channel ID: 1427896857274617939
Server ID: 658433759497945120
```

### 4. Configure Environment Variables

Add these to your `.env` file:

```env
# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_bot_token_here

# Free channels (comma-separated)
DISCORD_CHANNEL_IDS=1427896857274617939

# Premium channels (comma-separated, optional)
DISCORD_PREMIUM_CHANNEL_IDS=premium_channel_id_1,premium_channel_id_2
```

### 5. Alternative: JSON Configuration

You can also use a JSON config file for more detailed settings:

```json
{
  "bot_token": "YOUR_BOT_TOKEN_HERE",
  "channels": [
    {
      "channel_id": "1427896857274617939",
      "channel_name": "Free Alerts Channel",
      "is_premium": false,
      "enabled": true,
      "alert_types": ["ENTRY", "EXIT", "RUNNER", "ALERT", "STOP"]
    },
    {
      "channel_id": "PREMIUM_CHANNEL_ID",
      "channel_name": "Premium Alerts",
      "is_premium": true,
      "enabled": true,
      "alert_types": null
    }
  ]
}
```

Save as `discord_config.json` in your project root.

## Alert Parsing

The system automatically parses Discord messages for:

### Ticker Symbols
- `$TSLA` - with dollar sign
- `TSLA` - standalone (2-5 characters)
- `ticker: TSLA` - with label

### Prices
- `@ $150.50` - entry price
- `price: 150.50` - explicit price
- `entry: 150.50` - entry label

### Targets
- `target: 160` - price target
- `TP: 160` - take profit

### Stop Loss
- `stop loss: 145`
- `SL: 145`
- `stop: 145`

### Alert Types
- **ENTRY** - Buy signals (keywords: entry, buy, long, entering)
- **EXIT** - Sell signals (keywords: exit, sell, close, took profit)
- **RUNNER** - Momentum alerts (keywords: runner, running, breakout)
- **ALERT** - General watch alerts (keywords: alert, watch, monitoring)
- **STOP** - Stop hit notifications (keywords: stopped, stop hit)

### Example Alert Messages

```
✅ ENTRY: $TSLA @ $250.50, Target: $265, SL: $245
🚀 RUNNER: NVDA breaking out above $500
🔔 ALERT: Watch AAPL, potential setup forming
❌ EXIT: Closed SPY @ $455, +2.5% profit
```

## Usage

### Starting the Discord Bot

```python
from discord_alert_listener import create_discord_manager

# Create and start manager
discord_mgr = create_discord_manager()
if discord_mgr:
    discord_mgr.start()
    print("Discord bot started!")
```

### Getting Alerts

```python
# Get recent alerts
alerts = discord_mgr.get_alerts(limit=50)

# Get alerts for specific symbol
tsla_alerts = discord_mgr.get_symbol_alerts('TSLA', limit=10)

# Check if bot is running
if discord_mgr.is_running():
    print("Bot is connected and monitoring channels")
```

### Integration with AI Analysis

When generating AI trading signals, Discord alerts are automatically included:

```python
from ai_trading_signals import AITradingSignalGenerator

generator = AITradingSignalGenerator()

# Discord data structure
discord_data = {
    'alerts': discord_mgr.get_symbol_alerts('TSLA', limit=10)
}

# Generate signal with Discord data
signal = generator.generate_signal(
    symbol='TSLA',
    technical_data=tech_data,
    news_data=news_data,
    sentiment_data=sentiment_data,
    discord_data=discord_data,  # <-- Discord alerts included!
    account_balance=10000.0,
    risk_tolerance='MEDIUM'
)
```

The AI will analyze:
1. ✅ Technical indicators (RSI, MACD, etc.)
2. ✅ News and sentiment
3. ✅ Social media (Reddit/Twitter)
4. ✅ **Discord alerts from professional traders**
5. ✅ Risk/reward ratio

## Streamlit UI Integration

The Discord integration includes a dedicated tab in your Streamlit app:

### Features:
- 📊 View all recent alerts
- 🔍 Filter by symbol, type, or channel
- 📈 Alert statistics and trends
- 💾 Export alerts to CSV/JSON
- ⚙️ Bot status and configuration
- 🎯 Symbol-specific alert history

## Testing

Test your Discord bot setup:

```python
import asyncio
from discord_alert_listener import DiscordAlertListener

async def test_bot():
    token = "YOUR_BOT_TOKEN"
    channels = {"1427896857274617939": False}  # channel_id: is_premium
    
    bot = DiscordAlertListener(token, channels)
    await bot.start_listening()

# Run test
asyncio.run(test_bot())
```

## Troubleshooting

### Bot Not Connecting
- ✅ Verify bot token is correct
- ✅ Check bot has proper permissions in Discord server
- ✅ Ensure MESSAGE CONTENT INTENT is enabled
- ✅ Confirm channel IDs are correct

### No Alerts Being Captured
- ✅ Verify bot can see messages in the channel
- ✅ Check channel permissions (bot needs "Read Messages" and "Read Message History")
- ✅ Test with manual message in the channel
- ✅ Review alert parsing patterns (may need customization)

### Alert Parsing Issues
- ✅ Review raw message content
- ✅ Adjust regex patterns in `DiscordAlertParser` if needed
- ✅ Enable debug logging to see what's being parsed

## Security & Best Practices

### Security
- ⚠️ **Never commit your bot token to version control**
- ⚠️ Keep `.env` file in `.gitignore`
- ⚠️ Use environment variables for production
- ⚠️ Rotate bot token if exposed

### Best Practices
- ✅ Start with free channels to test
- ✅ Monitor bot logs for errors
- ✅ Set reasonable alert limits (100-500)
- ✅ Export alerts regularly for backup
- ✅ Combine Discord alerts with your own analysis (don't blindly follow!)

## Advanced Configuration

### Custom Alert Parsing

Customize parsing patterns in `discord_alert_listener.py`:

```python
class DiscordAlertParser:
    # Add custom ticker patterns
    TICKER_PATTERNS = [
        r'\$([A-Z]{1,5})',  # $TSLA
        r'TICKER:\s*([A-Z]{2,5})',  # Custom pattern
    ]
    
    # Add custom alert types
    ALERT_TYPE_KEYWORDS = {
        'ENTRY': ['entry', 'buy', 'long'],
        'CUSTOM': ['custom_keyword'],  # Your custom type
    }
```

### Multiple Discord Servers

Monitor multiple servers by adding multiple channel IDs:

```env
DISCORD_CHANNEL_IDS=channel1,channel2,channel3
DISCORD_PREMIUM_CHANNEL_IDS=premium1,premium2
```

## Performance Considerations

- Bot uses asyncio for non-blocking operation
- Alert storage uses deque with max size (default: 100)
- Alerts are kept in memory (export regularly for persistence)
- Minimal CPU usage when idle

## Rate Limits

Discord has rate limits:
- 50 requests per second per bot
- Message content is received via gateway (no REST API calls needed)
- Bot automatically handles rate limiting

## Future Enhancements

Potential additions:
- 📊 Alert sentiment scoring
- 🤖 Machine learning for alert quality
- 📈 Performance tracking (alert success rate)
- 🔔 Real-time notifications
- 📱 Mobile alerts via webhook

## Support

For issues or questions:
1. Check logs in `trading_signals.log`
2. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
3. Review Discord bot permissions
4. Test alert parsing with example messages

## Example Workflow

1. **Morning**: Start Discord bot, review overnight alerts
2. **Pre-Market**: Check Discord alerts for potential plays
3. **Market Open**: Use AI analysis (with Discord data) to validate entries
4. **Intraday**: Monitor for runner/exit alerts
5. **End of Day**: Export alerts, review performance

## Legal Disclaimer

⚠️ **Important**: 
- Discord alerts are for **informational purposes only**
- Always perform your own analysis
- Past performance doesn't guarantee future results
- You are responsible for your trading decisions
- Consider Discord alerts as ONE input among many data sources

---

**Ready to start?** Follow the setup steps above and begin monitoring Discord channels for trading opportunities! 🚀

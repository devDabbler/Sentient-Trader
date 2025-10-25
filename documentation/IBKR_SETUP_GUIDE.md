# Interactive Brokers (IBKR) Day Trading Setup Guide

This guide will help you set up Interactive Brokers integration for day trading and scalping with your AI Options Trader application.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installing IB Gateway](#installing-ib-gateway)
3. [Configuring API Access](#configuring-api-access)
4. [Installing Python Dependencies](#installing-python-dependencies)
5. [Connecting to IBKR](#connecting-to-ibkr)
6. [Using the Trading Interface](#using-the-trading-interface)
7. [Important Notes for Cash Accounts](#important-notes-for-cash-accounts)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Active IBKR Account**: You need an Interactive Brokers account (paper trading or live)
- **Account Type**: Cash account (as mentioned) or margin account
- **Python 3.8+**: Required for the application
- **Windows/Mac/Linux**: IB Gateway or TWS is available for all platforms

---

## Installing IB Gateway

IB Gateway is a lighter-weight alternative to Trader Workstation (TWS) that's perfect for API connections.

### Step 1: Download IB Gateway

1. Go to [Interactive Brokers Download Center](https://www.interactivebrokers.com/en/index.php?f=16457)
2. Download **IB Gateway** (Standalone Edition) for your operating system
3. Choose between:
   - **Stable** version (recommended for production)
   - **Latest** version (for newest features)

### Step 2: Install IB Gateway

1. Run the installer
2. Follow the installation wizard
3. Launch IB Gateway after installation

### Step 3: Log In

1. Enter your IBKR username and password
2. For paper trading: Use your paper trading credentials
3. For live trading: Use your live account credentials

---

## Configuring API Access

### Enable API Connections in IB Gateway

1. **In IB Gateway:**
   - Click on **Configure** → **Settings** → **API** → **Settings**
   
2. **Enable ActiveX and Socket Clients:**
   - Check the box "Enable ActiveX and Socket Clients"
   
3. **Set Socket Port:**
   - Paper Trading: Default is `4002`
   - Live Trading: Default is `4001`
   - **Note:** TWS uses `7497` for paper and `7496` for live
   
4. **Trusted IP Addresses:**
   - By default, only `127.0.0.1` (localhost) is trusted
   - For local use, this is sufficient
   
5. **Read-Only API:**
   - **Uncheck** "Read-Only API" to enable trading
   - Leave checked if you only want market data
   
6. **Allow Connections from localhost only:**
   - Keep this checked for security

### Alternative: Using TWS (Trader Workstation)

If you prefer TWS over IB Gateway:

1. Launch TWS
2. Go to **File** → **Global Configuration** → **API** → **Settings**
3. Same configuration as above
4. TWS uses ports `7497` (paper) and `7496` (live) by default

---

## Installing Python Dependencies

### Step 1: Install ib_insync

```bash
pip install ib_insync
```

Or, since it's in your `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```python
python -c "import ib_insync; print(ib_insync.__version__)"
```

You should see the version number (e.g., `0.9.86`).

---

## Connecting to IBKR

### Option 1: Using Environment Variables

Add these to your `.env` file:

```env
# IBKR Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=4002           # 4002 for paper, 4001 for live (IB Gateway)
                         # 7497 for paper, 7496 for live (TWS)
IBKR_CLIENT_ID=1         # Unique ID (1-32)
```

### Option 2: Using the UI

1. Open the app: `streamlit run app.py`
2. Navigate to the **📈 IBKR Trading** tab
3. Enter connection details:
   - **Host:** `127.0.0.1`
   - **Port:** 
     - `4002` (IB Gateway paper)
     - `4001` (IB Gateway live)
     - `7497` (TWS paper)
     - `7496` (TWS live)
   - **Client ID:** `1` (or any number 1-32)
4. Click **🔗 Connect to IBKR**

### Connection Checklist

- [ ] IB Gateway or TWS is running and logged in
- [ ] API is enabled in settings
- [ ] Port number is correct
- [ ] Socket port is configured (not Master API port)
- [ ] "Read-Only API" is unchecked (if you want to trade)
- [ ] You see "Connected to IBKR account XXXXXXX" message

---

## Using the Trading Interface

### Account Information

Once connected, you'll see:
- **Net Liquidation**: Total account value
- **Buying Power**: Available for trading
- **Cash**: Settled cash balance
- **Day Trades Left**: Important for PDT rules

### Viewing Positions

- Click **🔄 Refresh Positions** to update
- Shows all open positions with P&L
- Quick close buttons for each position

### Placing Orders

**Market Order** (immediate execution at current price):
1. Enter symbol (e.g., `SPY`)
2. Select action: `BUY` or `SELL`
3. Enter quantity
4. Select `MARKET`
5. Click **🚀 Place Order**

**Limit Order** (execute at specific price or better):
1. Same as above but select `LIMIT`
2. Enter limit price
3. Order fills only at limit price or better

**Stop Order** (trigger at specific price):
1. Select `STOP`
2. Enter stop price
3. Order becomes market order when stop price is reached

### Managing Orders

- View all open orders in the **📝 Open Orders** section
- Cancel individual orders by Order ID
- Cancel all orders with **❌❌ Cancel ALL Orders**

### Real-Time Market Data

- Enter a symbol
- Click **📈 Get Quote**
- See bid/ask spread, last price, and volume

### Emergency: Flatten Position

Use the **Close [SYMBOL]** buttons to immediately exit positions with market orders.

---

## Important Notes for Cash Accounts

### Trading Restrictions

1. **T+2 Settlement**: Cash from stock sales settles in 2 business days
2. **Good Faith Violations**: Buying and selling before settlement uses good faith
3. **Free-Riding**: Buying with unsettled funds and selling before settlement
4. **Pattern Day Trader (PDT)**: 
   - **< $25,000**: Max 3 day trades per 5 trading days
   - **≥ $25,000**: Unlimited day trades

### Best Practices for Cash Accounts

1. **Track Settlement Dates**: Know when your cash settles
2. **Use Settled Cash**: Only trade with settled funds to avoid violations
3. **Monitor Day Trades**: Stay within your 3 day trade limit if under $25k
4. **Plan Positions**: With T+2 settlement, plan your trades carefully

### Scalping with Cash Account

For active scalping with a cash account:

1. **Split Capital**: Use 1/3 of capital per day to rotate through settlements
2. **Trade High Volume Stocks**: Easier to enter/exit (penny stocks may be risky)
3. **Use Limit Orders**: Control execution prices
4. **Set Stop Losses**: Protect against large losses
5. **Consider Upgrading**: Margin accounts offer more flexibility for day trading

---

## Day Trading Strategies

### Momentum Scalping

1. Scan for high volume movers (use **💰 Top Penny Stocks** tab)
2. Enter on breakouts with tight stops
3. Take quick profits (0.5% - 2%)
4. Exit before end of day

### Range Trading

1. Identify support/resistance levels
2. Buy near support, sell near resistance
3. Use limit orders at key levels
4. Tight stops below support

### News Trading

1. Monitor **Stock Intelligence** tab for catalysts
2. Trade on earnings, upgrades, news
3. Be cautious of volatility
4. Use smaller position sizes

---

## Troubleshooting

### Connection Issues

**"Failed to connect to IBKR"**
- Ensure IB Gateway/TWS is running and logged in
- Check that API is enabled in settings
- Verify correct port number
- Check firewall settings

**"Connection timeout"**
- IB Gateway may be frozen - restart it
- Check Task Manager for zombie processes
- Increase timeout in connection settings

**"Already connected" error**
- Disconnect first, then reconnect
- Check for other applications using same client ID
- Restart IB Gateway

### API Configuration Issues

**"Socket port is not configured"**
- Go to API Settings in IB Gateway
- Enable "Enable ActiveX and Socket Clients"
- Set socket port (4002 for paper, 4001 for live)

**"Not authenticated"**
- Wrong username/password
- Paper trading credentials needed for paper account
- Check if account is locked

### Trading Issues

**"Order rejected"**
- Insufficient buying power
- Stock not available for trading
- Market is closed
- Check account restrictions

**"No market data"**
- Subscribe to market data in IBKR account
- Check data subscriptions in Account Management
- Some exchanges require additional subscriptions

**"Position not updating"**
- Click **🔄 Refresh Positions**
- Check if using correct account
- Verify API permissions

### Performance Issues

**Slow order execution**
- Check internet connection
- IB Gateway may need restart
- High market volatility can cause delays

**Data delays**
- Market data subscriptions needed
- Check data feed status in IB Gateway
- Some free data is delayed 15 minutes

---

## Advanced Features

### Multiple Client IDs

Run multiple instances by using different client IDs (1-32):
- App 1: Client ID = 1
- App 2: Client ID = 2
- TWS/Gateway: Client ID = 0 (default)

### Automated Trading

For automated strategies:

1. Use the **🎯 Strategy Advisor** tab to analyze
2. Use **📊 Generate Signal** to plan trades
3. Use IBKR Trading tab to execute
4. Monitor in **📊 Current Positions**

### Risk Management

Set up stops for every position:
1. Use STOP orders for automatic protection
2. Calculate position size based on account size
3. Never risk more than 1-2% per trade
4. Use the app's risk calculators

---

## Resources

### Interactive Brokers

- [IB API Documentation](https://interactivebrokers.github.io/)
- [IB Knowledge Base](https://www.interactivebrokers.com/en/support/index.php)
- [IB Client Portal](https://www.interactivebrokers.com/portal)

### ib_insync Library

- [Documentation](https://ib-insync.readthedocs.io/)
- [GitHub Repository](https://github.com/erdewit/ib_insync)
- [Examples](https://github.com/erdewit/ib_insync/tree/master/notebooks)

### Support

- IBKR Customer Service: Available 24/7
- API Technical Support: Via Client Portal tickets
- Community Forums: IBKR Community, Reddit r/interactivebrokers

---

## Safety Reminders

⚠️ **Trading involves risk of loss. Always:**
- Start with paper trading to learn the platform
- Use stop losses on every trade
- Never trade with money you can't afford to lose
- Understand PDT rules for your account type
- Monitor positions actively during market hours
- Be aware of settlement rules for cash accounts

🔒 **Security:**
- Keep API read/write permissions secure
- Only allow localhost (127.0.0.1) connections
- Use strong passwords
- Enable two-factor authentication on IBKR account
- Never share account credentials

---

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review IB Gateway logs: `File > Settings > Logging`
3. Check app logs: `trading_signals.log`
4. Contact IBKR support for account/API issues
5. Check ib_insync documentation for library issues

---

**Happy Trading! 📈**

Remember: The best trades are the ones you're prepared for. Use the app's intelligence features to research before trading.

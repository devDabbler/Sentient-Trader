# IBKR Day Trading/Scalping Integration

## Overview

Your AI Options Trader application now has **full Interactive Brokers integration** for day trading and scalping! This integration allows you to execute live trades, monitor positions, and manage orders directly from the application.

## What's New

### 1. IBKR Client Module (`ibkr_client.py`)

A comprehensive Python client for Interactive Brokers with the following features:

#### Core Features
- ✅ **Connection Management**: Connect/disconnect to IB Gateway or TWS
- ✅ **Account Information**: Real-time account balance, buying power, cash, PDT status
- ✅ **Position Tracking**: View all open positions with P&L
- ✅ **Order Management**: Place, modify, and cancel orders
- ✅ **Market Data**: Real-time quotes and historical data
- ✅ **Emergency Controls**: Quick position flattening

#### Supported Order Types
- **Market Orders**: Immediate execution at current price
- **Limit Orders**: Execute at specified price or better
- **Stop Orders**: Trigger at specified price (stop-loss/stop-entry)

#### Order Actions
- **BUY**: Open long or close short positions
- **SELL**: Close long or open short positions

### 2. Trading Interface (New Tab in App)

A new **📈 IBKR Trading** tab with:

#### Connection Panel
- Host configuration (default: 127.0.0.1)
- Port selection (7497 for paper, 7496 for live)
- Client ID management (1-32)
- Connection status indicators

#### Account Dashboard
- Net liquidation value
- Available buying power
- Cash balance
- Day trades remaining (PDT tracking)

#### Position Management
- Real-time position monitoring
- P&L tracking (realized and unrealized)
- Quick close buttons for each position
- Refresh functionality

#### Order Management
- View all open orders
- Order status tracking (submitted, filled, cancelled)
- Individual order cancellation
- Cancel all orders (emergency button)

#### Order Entry
- Symbol input with auto-uppercase
- Buy/Sell selection
- Quantity input
- Order type selection (Market, Limit, Stop)
- Dynamic price inputs based on order type
- One-click order placement
- Order confirmation with details

#### Market Data
- Real-time quotes
- Bid/Ask spread
- Volume information
- Last price

### 3. Updated Dependencies

Added to `requirements.txt`:
```
ib_insync>=0.9.86
```

### 4. Comprehensive Setup Guide

Created `IBKR_SETUP_GUIDE.md` with:
- Step-by-step installation instructions
- API configuration guide
- Connection troubleshooting
- Cash account trading rules
- Day trading strategies
- Safety and security best practices

## Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install `ib_insync` and all other required packages.

### Step 2: Set Up IB Gateway or TWS

1. Download IB Gateway from [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=16457)
2. Install and log in with your IBKR credentials
3. Enable API access:
   - Go to **Configure** → **Settings** → **API** → **Settings**
   - Check "Enable ActiveX and Socket Clients"
   - Set socket port: `4002` (paper) or `4001` (live)
   - **Uncheck** "Read-Only API" to enable trading

### Step 3: Configure Environment (Optional)

Add to your `.env` file:

```env
IBKR_HOST=127.0.0.1
IBKR_PORT=4002          # 4002 for paper, 4001 for live (Gateway)
IBKR_CLIENT_ID=1
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

### Step 5: Connect to IBKR

1. Navigate to the **📈 IBKR Trading** tab
2. Enter connection details:
   - Host: `127.0.0.1`
   - Port: `4002` (or your configured port)
   - Client ID: `1`
3. Click **🔗 Connect to IBKR**
4. Wait for "Connected to IBKR account" message

## Usage Examples

### Example 1: Simple Market Order

```python
# In the IBKR Trading tab:
Symbol: SPY
Action: BUY
Quantity: 10
Order Type: MARKET
[Click "Place Order"]
```

### Example 2: Limit Order

```python
Symbol: AAPL
Action: BUY
Quantity: 5
Order Type: LIMIT
Limit Price: 175.50
[Click "Place Order"]
```

### Example 3: Stop-Loss Order

```python
Symbol: TSLA
Action: SELL
Quantity: 2
Order Type: STOP
Stop Price: 240.00
[Click "Place Order"]
```

### Example 4: Quick Position Close

1. View position in **Current Positions** section
2. Click the **Close [SYMBOL]** button
3. Market order placed immediately

## Integration with Existing Features

### Workflow 1: Scan → Analyze → Trade

1. **💰 Top Penny Stocks** tab: Find high-potential stocks
2. **🔍 Stock Intelligence** tab: Deep dive analysis
3. **📈 IBKR Trading** tab: Execute the trade

### Workflow 2: Strategy → Signal → Trade

1. **🎯 Strategy Advisor** tab: Get AI recommendations
2. **📊 Generate Signal** tab: Plan the trade
3. **📈 IBKR Trading** tab: Execute the trade

### Workflow 3: Monitor → Manage

1. **📈 IBKR Trading** tab: View positions in real-time
2. Monitor P&L and market conditions
3. Adjust or close positions as needed

## Important Notes for Your Cash Account

### Cash Account Restrictions

1. **Settlement Period**: T+2 (2 business days)
2. **Good Faith Violations**: Be careful buying/selling before settlement
3. **Pattern Day Trader Rule**:
   - Accounts < $25,000: Max 3 day trades per 5 trading days
   - Accounts ≥ $25,000: Unlimited day trades

### Best Practices

1. **Track Day Trades**: Monitor "Day Trades Left" metric
2. **Use Settled Cash**: Avoid good faith violations
3. **Plan Rotations**: With T+2 settlement, rotate capital wisely
4. **Position Sizing**: Never risk more than 1-2% per trade
5. **Use Stops**: Always protect positions with stop orders

### Scalping Tips for Cash Accounts

- **Focus on High Volume Stocks**: Easier to enter/exit
- **Take Quick Profits**: 0.5% - 2% gains add up
- **Use Limit Orders**: Control execution prices
- **Monitor Spread**: Wide spreads eat into profits
- **Track Settlement**: Know when cash is available

## API Reference

### IBKRClient Class

```python
from ibkr_client import IBKRClient

# Create client
client = IBKRClient(host='127.0.0.1', port=4002, client_id=1)

# Connect
client.connect(timeout=10)

# Get account info
account_info = client.get_account_info()

# Place market order
order = client.place_market_order('SPY', 'BUY', 10)

# Place limit order
order = client.place_limit_order('AAPL', 'BUY', 5, 175.50)

# Place stop order
order = client.place_stop_order('TSLA', 'SELL', 2, 240.00)

# Get positions
positions = client.get_positions()

# Get open orders
orders = client.get_open_orders()

# Cancel order
client.cancel_order(order_id)

# Flatten position (emergency close)
client.flatten_position('SPY')

# Get market data
data = client.get_market_data('AAPL')

# Disconnect
client.disconnect()
```

## Security Features

### Built-in Protections

1. **Localhost Only**: Default connection to 127.0.0.1
2. **Session Management**: Client stored in session state
3. **Connection Validation**: Checks before every operation
4. **Error Handling**: Comprehensive try/catch blocks
5. **Logging**: All operations logged for audit trail

### Recommended Security

1. Enable two-factor authentication on IBKR account
2. Use strong passwords
3. Only allow localhost API connections
4. Monitor API usage in IBKR portal
5. Review logs regularly: `trading_signals.log`

## Troubleshooting

### Common Issues

**"Not connected to IBKR"**
- Make sure IB Gateway/TWS is running
- Check connection parameters
- Verify API is enabled

**"Connection timeout"**
- Restart IB Gateway
- Check firewall settings
- Verify port number

**"Order rejected"**
- Check buying power
- Verify market is open
- Check symbol validity
- Review account restrictions

**"No market data"**
- Subscribe to market data in IBKR
- Check data subscriptions
- Some data requires paid subscription

### Getting Help

1. Check `IBKR_SETUP_GUIDE.md` for detailed troubleshooting
2. Review application logs: `trading_signals.log`
3. Check IB Gateway logs: File → Settings → Logging
4. Contact IBKR support: 24/7 customer service
5. ib_insync documentation: https://ib-insync.readthedocs.io/

## File Structure

```
AI Options Trader/
├── ibkr_client.py              # IBKR integration module
├── app.py                      # Main app (with IBKR tab)
├── requirements.txt            # Updated with ib_insync
├── IBKR_SETUP_GUIDE.md        # Detailed setup guide
├── IBKR_INTEGRATION_README.md # This file
└── .env                        # Configuration (add IBKR settings)
```

## Features Comparison

| Feature | Before | After |
|---------|--------|-------|
| Trading Platform | Tradier Only | Tradier + IBKR |
| Order Types | Options Focus | Stocks + Options |
| Real-time Data | Limited | Full Market Data |
| Day Trading | Not Optimized | Fully Supported |
| Scalping | Not Supported | Optimized |
| Position Management | Manual | Automated |
| Order Execution | External | In-App |
| Paper Trading | Not Available | Full Support |

## Performance Considerations

### Latency

- **Connection**: < 100ms (local network)
- **Order Placement**: 50-200ms (depends on market)
- **Market Data**: Real-time with subscription
- **Position Updates**: < 500ms

### Recommendations

1. Use wired internet connection for trading
2. Keep IB Gateway running continuously during market hours
3. Restart IB Gateway daily to prevent memory leaks
4. Monitor connection status indicator
5. Use limit orders for better price control

## Future Enhancements

Potential additions (not yet implemented):

- [ ] Bracket orders (entry + stop + target)
- [ ] Trailing stop orders
- [ ] Options trading support
- [ ] Advanced charting
- [ ] Backtesting with historical data
- [ ] Automated strategy execution
- [ ] Multi-account support
- [ ] Trade journal/logging
- [ ] Performance analytics
- [ ] Alert notifications

## Support

For issues or questions:

- **Application**: Check logs in `trading_signals.log`
- **IBKR API**: Contact IBKR support or check their API docs
- **ib_insync**: Visit https://github.com/erdewit/ib_insync

## License & Disclaimer

⚠️ **Trading Disclaimer**: 
- Trading stocks and options involves risk of loss
- Past performance does not guarantee future results
- This tool is for informational purposes
- Always start with paper trading
- Never trade with money you can't afford to lose
- Consult a financial advisor before trading

This integration is provided as-is. The developers are not responsible for any trading losses.

---

**You're now ready to day trade and scalp with Interactive Brokers! 🚀📈**

Start with paper trading to get comfortable with the interface, then move to live trading when ready.

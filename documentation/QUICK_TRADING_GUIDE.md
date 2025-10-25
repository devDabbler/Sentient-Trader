# 🚀 Quick Trading Guide - Tradier Integration

## How to Execute Trades in the App

### ✅ Prerequisites
1. **Tradier Account Setup** - Make sure your `.env` file has:
   ```
   TRADIER_ACCOUNT_ID=your_account_id
   TRADIER_ACCESS_TOKEN=your_access_token
   TRADIER_API_URL=https://sandbox.tradier.com
   ```
2. **App Running** - Start with `streamlit run app.py`

---

## 🎯 Method 1: Quick Buy from Dashboard (NEW!)

This is the **fastest way** to trade after analyzing a stock:

### Steps:
1. **Go to 🏠 Dashboard tab**
2. **Enter a ticker** (e.g., TSLA, AAPL, SOFI)
3. **Select your trading style** (Day Trade, Swing Trade, Options, etc.)
4. **Click "🔍 Analyze Stock"**
5. **Review the analysis:**
   - ML-Enhanced Confidence Score
   - Comprehensive Trading Verdict
   - Risk/Reward metrics
6. **Click "💰 Quick Buy (Tradier)"** button at the bottom
7. **Configure your order:**
   - Symbol (pre-filled)
   - Action (buy, sell, sell_short, buy_to_cover)
   - Quantity (shares)
   - Order Type (market or limit)
   - Price (if limit order)
8. **Review the Order Summary:**
   - Estimated Cost
   - Verdict from analysis
   - Stop Loss & Target suggestions
9. **Click "✅ Place Order"**
10. **Confirmation** - You'll see the Order ID and status

### Features:
- ✅ **Pre-filled with analysis data** - Symbol and price auto-populated
- ✅ **Shows your verdict** - Reminds you if it's STRONG BUY, CAUTIOUS, etc.
- ✅ **Stop Loss & Target suggestions** - Based on support/resistance
- ✅ **Risk warnings** - Alerts you if analysis suggests caution
- ✅ **Market or Limit orders** - Your choice
- ✅ **Instant execution** - Direct to Tradier API

---

## 🏦 Method 2: Tradier Account Tab (Full Control)

For more advanced order management:

### Steps:
1. **Go to 🏦 Tradier Account tab**
2. **Click "🔄 Refresh Account Data"** to see:
   - Total Cash
   - Buying Power
   - Current Positions
   - Recent Orders
3. **Scroll to "🎯 Manual Order Placement"**
4. **Click "Place Custom Order" expander**
5. **Configure your order:**
   - Order Class (equity, option, multileg, etc.)
   - Symbol
   - Side (buy, sell, buy_to_open, sell_to_close, etc.)
   - Quantity
   - Order Type (market, limit, stop, etc.)
   - Duration (day, gtc, pre, post)
   - Price (if applicable)
6. **Click "📤 Place Order"**

### Features:
- ✅ **Full order types** - Market, Limit, Stop, Stop-Limit, Credit, Debit
- ✅ **Options support** - Buy/sell calls and puts
- ✅ **Multi-leg strategies** - Spreads, iron condors, etc.
- ✅ **Order management** - Check status, cancel orders
- ✅ **Account overview** - See all positions and P&L

---

## 📊 Order Types Explained

### Market Order
- **Executes immediately** at current market price
- **Best for:** Liquid stocks, when speed matters
- **Risk:** May get filled at slightly different price than expected

### Limit Order
- **Executes only at your specified price or better**
- **Best for:** Getting exact entry price, less liquid stocks
- **Risk:** May not fill if price doesn't reach your limit

### Example Scenarios:

#### Day Trading TSLA
```
Symbol: TSLA
Action: buy
Quantity: 10
Order Type: market
→ Buys 10 shares immediately at market price
```

#### Swing Trading SOFI with Limit
```
Symbol: SOFI
Action: buy
Quantity: 100
Order Type: limit
Price: $10.50
→ Buys 100 shares only if price drops to $10.50 or below
```

---

## ⚠️ Important Safety Tips

### Before Trading:
1. **Check your verdict score** - Don't trade if it says "AVOID / WAIT"
2. **Review ML confidence** - Higher confidence = better setup
3. **Check system agreement** - ML and Technical should align
4. **Set stop losses** - Use the suggested support level
5. **Position sizing** - Follow the recommended % of portfolio

### Risk Management:
- 🔴 **STRONG BUY (75+):** 2-5% of portfolio
- 🟢 **BUY (60-74):** 1-3% of portfolio
- 🟡 **CAUTIOUS BUY (45-59):** 0.5-1.5% of portfolio
- 🔴 **AVOID (<45):** Skip this trade

### Common Mistakes to Avoid:
- ❌ Trading against the verdict
- ❌ Ignoring ML/Technical divergence warnings
- ❌ Not setting stop losses
- ❌ Over-sizing positions
- ❌ Trading penny stocks with large positions
- ❌ Holding through earnings without planning

---

## 🔍 Monitoring Your Trades

### Check Order Status:
1. Go to **🏦 Tradier Account** tab
2. Click **"🔄 Refresh Account Data"**
3. View **"📋 Recent Orders"** section
4. Or enter Order ID in **"Get Order Status"** section

### Cancel an Order:
1. Go to **🏦 Tradier Account** tab
2. Find **"Cancel Order"** section
3. Enter the Order ID
4. Click **"❌ Cancel Order"**

---

## 💡 Pro Tips

### For Day Trading:
- Use **market orders** for quick entries
- Set **tight stop losses** (3-5%)
- Take profits at resistance levels
- Close all positions before market close

### For Swing Trading:
- Use **limit orders** for better entries
- Set **wider stops** (5-8%)
- Hold through minor pullbacks
- Watch for catalyst dates

### For Options:
- Check **IV Rank** before trading
- High IV (>60%) = Sell premium
- Low IV (<40%) = Buy options
- Avoid trading right before earnings (unless planned)

---

## 🆘 Troubleshooting

### "Tradier not connected" error:
1. Check your `.env` file has correct credentials
2. Go to **🏦 Tradier Account** tab
3. Click **"🔍 Test Connection"**
4. If fails, verify your API token is valid

### Order rejected:
- **Insufficient funds:** Check buying power
- **Invalid symbol:** Verify ticker is correct
- **Market closed:** Use GTC orders or wait for market hours
- **Invalid quantity:** Options require multiples of 100

### Can't see positions:
1. Click **"🔄 Refresh Account Data"**
2. Check if you're using sandbox vs. live account
3. Verify account ID matches your Tradier account

---

## 📈 Example Workflow

### Complete Trading Flow:
1. **Analyze:** Dashboard → Enter TSLA → Select "Day Trade" → Analyze
2. **Review:** 
   - ML Score: 78/100 (HIGH)
   - Verdict: BUY (Score 68/100)
   - System Agreement: ✅ Strong
3. **Trade:** Click "💰 Quick Buy"
   - Quantity: 10 shares
   - Type: Market
   - Estimated: $2,650
4. **Execute:** Click "✅ Place Order"
5. **Confirm:** Order ID: 12345 - Filled
6. **Monitor:** Set alert at support ($260) for stop loss
7. **Exit:** Target at resistance ($275) or EOD

---

## 🎓 Learning Resources

- **Strategy Guide tab** - Learn different options strategies
- **Strategy Analyzer tab** - Backtest your ideas
- **Signal History tab** - Review past signals and performance

---

## ⚡ Quick Reference

| Action | Location | Button |
|--------|----------|--------|
| Quick trade after analysis | Dashboard | 💰 Quick Buy |
| Manual order placement | Tradier Account | 📤 Place Order |
| Check account balance | Tradier Account | 🔄 Refresh |
| View positions | Tradier Account | Account Overview |
| Cancel order | Tradier Account | ❌ Cancel Order |
| Check order status | Tradier Account | 🔍 Get Order Status |

---

**Remember:** This is a paper trading sandbox. Test thoroughly before using with real money!

**Support:** Check logs in terminal for detailed error messages.

# 🎯 DEX Launch Hunter - Quick Start

Get started catching early token launches in 15 minutes.

---

## ⚡ 5-Minute Setup

### Step 1: Add Discord Webhook (2 minutes)

**Get instant alerts on promising launches:**

1. Open Discord
2. Go to Server Settings → Integrations → Webhooks
3. Click "New Webhook"
4. Name it "DEX Hunter"
5. Copy webhook URL
6. Add to `.env`:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_url_here
   ```

### Step 2: Open DEX Hunter Tab (1 minute)

1. Run Sentient Trader: `streamlit run app.py`
2. Navigate to: **₿ Crypto Trading** → **🎯 DEX Launch Hunter**
3. You should see 6 tabs:
   - 🔍 Launch Scanner
   - 🐋 Smart Money
   - ⚙️ Configuration
   - 🚨 Alerts
   - 📊 Portfolio
   - 📚 Resources

### Step 3: Configure Settings (2 minutes)

Go to **⚙️ Configuration** tab:

**Copy these beginner-safe settings:**
```
Enabled Chains: ✅ Ethereum, ✅ BSC
Min Liquidity: $10,000
Max Liquidity: $500,000
Max Buy Tax: 10%
Max Sell Tax: 10%
Require LP Locked: ✅ YES
Min LP Lock Days: 30
Maximum Risk Level: MEDIUM
Min Composite Score: 60
Scan Interval: 60 seconds
```

Click **💾 Save Configuration**

---

## 🚀 First Use

### Test with Known Token (5 minutes)

**Verify system works before live hunting:**

1. Go to **🔍 Launch Scanner**
2. Scroll to "🔍 Analyze Specific Token"
3. Paste a known token address:
   - **ETH Example**: `0xdac17f958d2ee523a2206206994597c13d831ec7` (USDT)
   - **BSC Example**: `0x55d398326f99059fF775485246999027B3197955` (USDT)
4. Select chain (ethereum or bsc)
5. Click **🔬 Analyze Token**
6. You should see full analysis with safety scores

**What you'll see:**
- Composite score (0-100)
- Safety breakdown
- Contract checks (honeypot, taxes, LP lock)
- Risk level

### Start Live Scanning (3 minutes)

1. Go to **🔍 Launch Scanner**
2. Click **🚀 Start Scanner**
3. Scanner runs in background
4. Check **🚨 Alerts** tab for promising launches
5. You'll get Discord notification when found

---

## 📊 Your First Trade

### When You Get an Alert

**⚠️ STOP! Don't rush. Follow this checklist:**

✅ **Safety Checks (5 minutes):**
1. ✅ Safety Score > 70?
2. ✅ Not a honeypot?
3. ✅ Buy/Sell tax < 10%?
4. ✅ LP locked > 30 days?
5. ✅ Renounced ownership?

✅ **Manual Verification (10 minutes):**
1. Click "📊 View Chart" → Check DexScreener
2. Visit Etherscan/BSCScan → Verify contract
3. Search Twitter for token name → Check community
4. Search Telegram → Look for official group
5. Google search → Check for warnings

✅ **Red Flag Check:**
- ❌ Anonymous team?
- ❌ Copied whitepaper?
- ❌ "Guaranteed 1000x" promises?
- ❌ Telegram filled with bots?
- ❌ No liquidity lock?

**If ANY red flags → SKIP. Next opportunity.**

### Executing Your First Trade

**If all checks pass:**

1. **Open Trust Wallet app**
2. **Go to DApps browser**
3. **Visit:**
   - BSC tokens → PancakeSwap.finance
   - ETH tokens → app.uniswap.org
4. **Connect wallet**
5. **Paste contract address**
6. **Set slippage:** 12-15% for new tokens
7. **Enter amount:** Start with $25-50 (learning trade)
8. **Confirm transaction**

**Immediately after buying:**
- Note entry price
- Set mental stop loss (-50% = exit)
- Set profit targets (2x, 5x, 10x)
- Add price alert on DexScreener

### Exit Strategy

**Take profits incrementally:**
- **At 2x**: Sell 50% (recover investment)
- **At 5x**: Sell 25% (lock gains)
- **At 10x**: Sell 15% (secure profits)
- **Let 10% ride** with trailing stop

**Stop loss:**
- If down 50%+ → Exit
- Don't hold losers hoping for recovery
- Take the L and move on

---

## 🐋 Add Your First Smart Money Wallet

**Track successful traders:**

### Finding a Good Wallet (10 minutes)

1. **Go to DexScreener**
2. **Find a token that did 100x+**
3. **Click on contract → View on Etherscan**
4. **Go to "Holders" tab**
5. **Find wallets that bought in first hour**
6. **Click on wallet → Check transaction history:**
   - Multiple early successful buys?
   - Quick entries on new launches?
   - High win rate?

### Adding to Tracker

1. Go to **🐋 Smart Money** tab
2. Click **➕ Add Wallet to Track**
3. Fill in:
   ```
   Wallet Address: 0x...
   Name: "Whale #1 - ETH Early Bird"
   Chain: ethereum
   Tags: whale, early_adopter
   Min Transaction: $1000
   ```
4. Click **Add Wallet**

**Now when this wallet buys a new token → You get alerted!**

---

## 💰 Position Sizing Guide

### Your First 10 Trades

**Learning Phase Strategy:**

```
Total DEX Fund: $250 (money you can lose)
Per Trade: $25
Number of Trades: 10

Expected Results:
- 7 losses: -$175 (70% loss rate)
- 2 small wins (2x): +$50
- 1 medium win (5x): +$100
Total: -$25 (break even or small loss)

Goal: LEARN, not profit
```

### After Gaining Experience (20+ trades)

```
Total DEX Fund: $1,000
Per Trade: $50-100
Conservative allocation: 5% per token

Expected Results (with experience):
- 6 losses: -$300
- 3 small wins (2-3x): +$200
- 1 big win (10x+): +$500
Total: +$400 profit (40% ROI)
```

### Never Exceed

- ❌ More than 10% of crypto portfolio in DEX speculation
- ❌ More than 2% per token
- ❌ Trading on leverage (spot only)
- ❌ Borrowing money to trade
- ❌ Money you need for bills

---

## 🚨 Emergency Procedures

### If Token Starts Dumping

**Price dropping fast? Act quickly:**

1. **Check if honeypot** - Try to sell small amount
2. **If can sell** - Exit immediately
3. **If honeypot** - You're rugged, learn from it
4. **Don't panic hold** - Hope doesn't work

### If You Suspect Rug Pull

**Indicators:**
- Devs leave Telegram
- Liquidity suddenly drops
- Wallet shows can't sell
- Contract ownership changed

**Action:**
1. Try to sell ASAP
2. If honeypot, you're stuck
3. Report to community
4. Mark wallet/dev as scammer
5. Move on, don't revenge trade

### Managing Emotions

**Common mistakes:**
- ✋ Holding losers hoping for recovery → CUT LOSSES
- 🏃 Chasing pumps → MISS IT, NEXT ONE
- 💎 "Diamond hands" on scam → SMART EXIT
- 😤 Revenge trading after loss → TAKE A BREAK
- 🤑 Going all-in on "sure thing" → POSITION SIZE!

---

## 📈 Tracking Your Performance

### Keep a Simple Spreadsheet

**Columns to track:**
```
Date | Token | Entry $ | Exit $ | % Gain | Holding Time | Reason | Lessons
```

**Example:**
```
1/15 | DOGE2 | $25 | $50  | +100% | 2 days | Good chart, locked LP | Sold too early, went to 5x
1/16 | SHIB3 | $25 | $12  | -52%  | 1 day  | Honeypot missed | Check better next time
1/17 | PEPE5 | $25 | $125 | +400% | 3 days | Smart money signal | Nailed it! Patient hold
```

**After 10 trades, review:**
- What's your win rate?
- Average gain on winners?
- Average loss on losers?
- What patterns do winners share?
- What red flags did you miss?

---

## ⚙️ Recommended Settings by Experience

### Beginner (First 20 trades)

```
Chains: ETH, BSC only
Min Liquidity: $50,000 (more established)
Max Liquidity: $300,000
Max Buy/Sell Tax: 5% (lower = safer)
Risk Level: LOW to MEDIUM
Min Composite Score: 70 (high quality only)
Position Size: $10-25
```

### Intermediate (20-100 trades)

```
Chains: ETH, BSC, Base
Min Liquidity: $20,000
Max Liquidity: $500,000
Max Buy/Sell Tax: 10%
Risk Level: MEDIUM
Min Composite Score: 60
Position Size: $25-50
```

### Advanced (100+ trades, profitable)

```
Chains: All (ETH, BSC, Solana, Base, Arbitrum)
Min Liquidity: $10,000
Max Liquidity: $1,000,000
Max Buy/Sell Tax: 15% (evaluate case by case)
Risk Level: MEDIUM to HIGH
Min Composite Score: 50
Position Size: $50-200
```

---

## 🎯 Success Metrics

### Week 1 Goals

- ✅ Complete 3-5 test analyses
- ✅ Execute 1-2 small test trades ($10-25)
- ✅ Add 3-5 smart money wallets
- ✅ Receive first Discord alert
- ✅ Learn to spot scams

### Month 1 Goals

- ✅ Complete 10-20 trades
- ✅ Break even or small loss (learning cost)
- ✅ Track all trades in spreadsheet
- ✅ Identify 2-3 winning patterns
- ✅ Build portfolio of 10+ watched wallets

### Month 3 Goals (If Continuing)

- ✅ Win rate > 30%
- ✅ Net positive ROI
- ✅ Find 1 big winner (10x+)
- ✅ Develop personal entry/exit system
- ✅ Confident in safety checks

---

## 📚 Next Steps

### Essential Reading

1. **Full Guide**: `docs/DEX_LAUNCH_HUNTER_GUIDE.md`
2. **Risk Management**: Section on position sizing
3. **Smart Money Tracking**: How to find wallets
4. **Troubleshooting**: Common issues

### Join Communities

**Learn from others:**
- r/CryptoMoonShots (filter carefully, 90% scams)
- Crypto Twitter (follow @0xQuit, @CryptoKaleo)
- Telegram alpha groups (start with 2-3 reputable ones)
- Discord DEX trading servers

**⚠️ Warning:** Most "alpha" is from people who bought before posting. Think critically.

### Practice First

**Before risking real money:**
1. Analyze 20-30 tokens without trading
2. Track which ones pump
3. See if you can spot winners
4. Identify your mistake patterns
5. Then start with tiny positions

---

## 🎓 Learning Resources

### Must-Watch Videos

- "How to Find New Crypto Tokens Early" (YouTube)
- "Honeypot Detection Tutorial" (YouTube)
- "Reading Etherscan Transactions" (YouTube)

### Must-Read Guides

- [How to Use DexScreener](https://docs.dexscreener.com/)
- [Contract Safety Basics](https://honeypot.is/docs)
- [Understanding Liquidity Pools](https://academy.binance.com/en/articles/what-are-liquidity-pools-in-defi)

### Tools to Master

- **DexScreener** - Price charts, new pairs
- **Etherscan/BSCScan** - Contract verification
- **Trust Wallet** - Mobile trading
- **Telegram** - Community monitoring

---

## ⚠️ Final Warnings

### Before You Start

**Understand this:**
- 70-90% of new tokens are scams
- Most will go to zero
- You will lose trades
- This is not passive income
- Requires active monitoring
- Very time consuming
- Extremely stressful
- Most traders lose money

**Are you ready to:**
- ✅ Lose money while learning?
- ✅ Spend 1-2 hours daily?
- ✅ Stay calm during losses?
- ✅ Be disciplined with stops?
- ✅ Take small profits consistently?

**If NO to any → This isn't for you. Stick to established cryptos.**

---

## 🚀 Ready to Start?

### Your Checklist

- [ ] Discord webhook configured
- [ ] Settings configured (beginner safe)
- [ ] Test token analysis works
- [ ] Scanner started
- [ ] First smart money wallet added
- [ ] Trust Wallet set up
- [ ] Tracking spreadsheet ready
- [ ] Read full guide
- [ ] Understand risks

### First Steps

1. **Today**: Set up system, test analyze 5 tokens
2. **This week**: Get first alert, do safety checks, don't trade yet
3. **Next week**: Execute first tiny trade ($10-25)
4. **This month**: Complete 10-20 learning trades
5. **Month 2**: Refine strategy, increase position size if profitable

---

**Good luck, stay safe, and may you catch a 100x! 🚀**

**Remember: The goal is consistent small wins, not one lucky lottery ticket.**

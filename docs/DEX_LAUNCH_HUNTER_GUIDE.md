# 🎯 DEX Launch Hunter - Complete Guide

**Catch early token launches on DEXs before they pump.**

⚠️ **EXTREMELY HIGH RISK** - Only for experienced traders who understand the risks of DEX speculation.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup & Configuration](#setup--configuration)
4. [API Keys Required](#api-keys-required)
5. [How to Use](#how-to-use)
6. [Risk Management](#risk-management)
7. [Finding Smart Money Wallets](#finding-smart-money-wallets)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

### What is DEX Launch Hunter?

A comprehensive system for discovering and analyzing new token launches on decentralized exchanges (DEXs) like:
- Uniswap (Ethereum)
- PancakeSwap (BSC)
- Raydium (Solana)
- Base DEXs
- Arbitrum DEXs

### Key Features

✅ **Real-time Launch Detection** - Monitor DexScreener for new pairs
✅ **Contract Safety Analysis** - Honeypot detection, tax analysis, LP lock verification
✅ **Smart Money Tracking** - Follow successful whale wallets
✅ **Social Sentiment** - Aggregate buzz from Reddit, Twitter, Telegram
✅ **Risk Scoring** - 0-100 safety score with detailed breakdown
✅ **Discord Alerts** - Get notified of promising launches instantly

### What Makes It Different?

Unlike your main Kraken crypto trading:
- **DEX Focus**: Catches tokens BEFORE major exchange listings
- **Ultra-Early**: Monitors launches < 1 hour old
- **High Risk/Reward**: Target 10-500x gains (with high failure rate)
- **Manual Execution**: Uses Trust Wallet + PancakeSwap/Uniswap
- **Separate Portfolio**: Keep DEX speculation isolated from main trading

---

## 🏗️ Architecture

### Component Overview

```
DEX Launch Hunter
├── Data Models (models/dex_models.py)
│   ├── TokenLaunch
│   ├── ContractSafety
│   ├── SmartMoneyActivity
│   └── LaunchAlert
│
├── API Clients
│   ├── DexScreenerClient - New pairs discovery
│   └── (Future: On-chain monitoring via Web3)
│
├── Services
│   ├── TokenSafetyAnalyzer - Honeypot/rug detection
│   ├── SmartMoneyTracker - Whale wallet monitoring
│   └── DexLaunchHunter - Main orchestrator
│
└── UI (ui/tabs/dex_hunter_tab.py)
    ├── Launch Scanner
    ├── Smart Money Tracker
    ├── Configuration
    ├── Alerts
    └── Resources
```

### Data Flow

```
1. DexScreener API → New pairs feed
2. Token Safety Analyzer → Honeypot check + contract audit
3. Smart Money Tracker → Check if whales buying
4. Scoring Engine → Calculate pump potential (0-100)
5. Alert System → Discord notification if score > threshold
6. Manual Execution → User trades via Trust Wallet
```

---

## ⚙️ Setup & Configuration

### Step 1: Install Dependencies

All required packages are already in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `httpx` - Async HTTP client for APIs
- `crawl4ai` - Social sentiment scraping

### Step 2: Configure API Keys

Add these to your `.env` file:

```bash
# DexScreener (Optional - free tier works)
DEXSCREENER_API_KEY=your_key_here  # Optional, increases rate limits

# DeBank (for wallet tracking)
DEBANK_API_KEY=your_key_here  # Optional but recommended

# Discord Webhook (for alerts)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url

# Optional: Nansen (paid, for advanced wallet analytics)
NANSEN_API_KEY=your_key_here
```

### Step 3: Configure Settings

In the app, go to: **🎯 DEX Launch Hunter → ⚙️ Configuration**

**Recommended Beginner Settings:**
- Enabled Chains: Ethereum, BSC (avoid Solana until experienced)
- Min Liquidity: $10,000
- Max Liquidity: $500,000 (focus on early launches)
- Max Buy Tax: 10%
- Max Sell Tax: 10%
- Require LP Locked: ✅ YES
- Min LP Lock Days: 30
- Max Risk Level: MEDIUM
- Min Composite Score: 60

### Step 4: Add Smart Money Wallets

Go to: **🎯 DEX Launch Hunter → 🐋 Smart Money**

**How to Find Good Wallets:**
1. Find a token that did a 100x+ run
2. Look at early buyers on Etherscan/BSCScan
3. Check their history - do they have multiple successful early buys?
4. Add top performers to your tracker

**Preset Whales Included:**
- Several known successful wallets are pre-loaded
- Check the "Tracked Wallets" section

---

## 🔑 API Keys Required

### Essential (Free Tier Available)

#### 1. **DexScreener API**
- **Purpose**: Find new token launches
- **Cost**: Free tier available (limited rate)
- **Paid Tier**: $49/mo for real-time alerts
- **Sign up**: https://dexscreener.com/
- **Note**: Free tier has delay, paid is recommended for serious hunting

#### 2. **Discord Webhook**
- **Purpose**: Receive instant alerts
- **Cost**: FREE
- **Setup**:
  1. Create Discord server (or use existing)
  2. Go to Server Settings → Integrations → Webhooks
  3. Create webhook, copy URL
  4. Add to `.env` as `DISCORD_WEBHOOK_URL`

### Recommended (Enhance Accuracy)

#### 3. **GoPlus Security API**
- **Purpose**: Contract safety checks
- **Cost**: FREE (10k requests/day)
- **Built-in**: Already integrated, no key needed
- **Website**: https://gopluslabs.io/

#### 4. **Honeypot.is API**
- **Purpose**: Honeypot detection
- **Cost**: FREE
- **Built-in**: Already integrated, no key needed
- **Website**: https://honeypot.is/

#### 5. **DeBank API**
- **Purpose**: Track wallet activity
- **Cost**: Free tier available
- **Sign up**: https://debank.com/
- **Add to .env**: `DEBANK_API_KEY`

### Advanced (Optional)

#### 6. **Nansen**
- **Purpose**: Advanced wallet analytics
- **Cost**: $150/mo+
- **Sign up**: https://www.nansen.ai/
- **For**: Serious degen hunters only

---

## 🚀 How to Use

### Basic Workflow

#### 1. **Start the Scanner**
- Go to: **🎯 DEX Launch Hunter → 🔍 Launch Scanner**
- Click **"🚀 Start Scanner"**
- Scanner runs in background, checking every 60 seconds

#### 2. **Review Alerts**
- Go to: **🎯 DEX Launch Hunter → 🚨 Alerts**
- See high-priority launches
- Each alert shows:
  - Token symbol & chain
  - Safety score
  - Pump potential
  - Risk level
  - Reasons for alert

#### 3. **Analyze Token**
- Click on alert to expand full details
- Review safety analysis:
  - ✅ Not a honeypot
  - ✅ Low taxes (< 10%)
  - ✅ LP locked
  - ✅ Renounced ownership
  - ❌ Red flags

#### 4. **Manual Verification**
**ALWAYS do these checks manually:**
- Visit DexScreener chart (click "📊 View Chart")
- Check contract on Etherscan/BSCScan
- Verify liquidity lock on Unicrypt/Team Finance
- Search Twitter/Telegram for team/community
- Look for red flags:
  - Anonymous team
  - Copied code
  - Pump & dump language
  - Too good to be true promises

#### 5. **Execute Trade (if confident)**
**In Trust Wallet:**
1. Open Trust Wallet app
2. Go to DApp browser
3. Navigate to PancakeSwap (BSC) or Uniswap (ETH)
4. Connect wallet
5. Paste contract address
6. Set slippage (usually 12-15% for new tokens)
7. Buy small amount first ($50-200)
8. Set stop loss mentally (e.g., -50%)
9. Set take profit targets (2x, 5x, 10x)

#### 6. **Monitor Position**
- Track manually in spreadsheet or notes
- Use DEX price tracker like DexScreener
- Set price alerts
- Take profits incrementally
- Move stop loss up as price increases

### Advanced: Manual Token Analysis

Paste any contract address to analyze:
1. Go to: **🎯 DEX Launch Hunter → 🔍 Launch Scanner**
2. Scroll to "🔍 Analyze Specific Token"
3. Enter contract address
4. Select chain (ethereum, bsc, solana, etc.)
5. Click "🔬 Analyze Token"
6. Review complete safety breakdown

---

## ⚠️ Risk Management

### The Reality of DEX Launches

**Success Rate:** ~5-10% of new tokens survive
**Scam Rate:** ~50-70% are rugs or honeypots
**Big Winners:** ~1-2% do 100x+
**Expected Loss:** Assume 80% of trades lose money

### Position Sizing

**NEVER invest more than you can afford to lose 100%.**

**Recommended Allocation:**
- Total DEX Portfolio: Max 5-10% of total crypto holdings
- Per Token: Max 1-2% of DEX portfolio
- Example: $10k crypto portfolio → $500 DEX fund → $10-20 per token

**Example Portfolio:**
```
Total Crypto: $10,000
DEX Speculation: $500 (5%)
Per Token: $10-25

Results after 20 trades:
- 16 losses: -$320 (16 x $20)
- 3 small wins (2x): +$60 (3 x $20 profit)
- 1 big win (20x): +$400 ($20 → $400)
Total: +$140 (28% ROI)
```

### Exit Strategy

**ALWAYS have an exit plan BEFORE buying:**

**Take Profits Incrementally:**
- At 2x: Sell 50% (recover initial investment)
- At 5x: Sell 25% (lock in gains)
- At 10x: Sell 15% (secure profits)
- Let remaining 10% ride with trailing stop

**Stop Loss:**
- Mental stop at -50% to -75%
- DEXs don't have stop loss orders
- Use price alerts and manual selling
- Don't hold losers hoping for recovery

**Time Limit:**
- Most pumps happen in first 24-48 hours
- If nothing happens in 1 week, consider exiting
- Don't marry your bags

### Red Flags - AVOID These

🚩 **Contract Red Flags:**
- Is honeypot (can't sell)
- Buy/sell tax > 15%
- No LP lock or < 30 days
- Mintable tokens
- Hidden owner
- Blacklist function
- Can pause trading

🚩 **Team Red Flags:**
- Anonymous dev
- No social media presence
- Copied whitepaper
- Fake team photos (reverse image search)
- Promises of guaranteed returns
- "Next 1000x" marketing

🚩 **Community Red Flags:**
- Telegram filled with bots
- Fake engagement (bought likes/followers)
- Overly aggressive shilling
- No organic discussion
- Admins deleting questions
- Price talk only, no tech discussion

### Green Flags - Look For These

✅ **Contract Green Flags:**
- LP locked 6+ months
- Renounced ownership
- Low taxes (< 5%)
- Audited by reputable firm
- Open source verified code
- No blacklist/pause functions
- Fair launch (no presale)

✅ **Team Green Flags:**
- Doxxed team (verified identities)
- Active Twitter/Telegram
- Regular updates
- Transparent communication
- Prior successful projects
- Clear roadmap
- Real utility/use case

✅ **Community Green Flags:**
- Organic growth
- Technical discussions
- Patient community
- Long-term holders
- Real partnerships
- Active development
- Growing adoption

---

## 🐋 Finding Smart Money Wallets

### Where to Find Successful Wallets

#### 1. **From Past Winners**
1. Find a token that did 100x+ (use DexScreener historical)
2. Go to Etherscan/BSCScan
3. Look at "Holders" tab
4. Click on wallets that bought within first hour
5. Check their transaction history:
   - Do they have multiple early successful buys?
   - What's their win rate?
   - How quickly do they sell?
6. Add top performers to tracker

#### 2. **From Nansen (Paid)**
- "Smart Money" labeled wallets
- Filter by profitability
- Track 50-100 top performers

#### 3. **From On-Chain Analytics**
- Use Arkham Intelligence
- Use DeBank
- Use Zerion
- Look for wallets with:
  - High ROI (5x+ average)
  - Multiple early positions
  - Quick entries on new tokens

#### 4. **From Social Media**
- Crypto Twitter whales who share wallets
- On-chain alpha groups
- Telegram alpha calls that show wallet addresses
- **Always verify** - don't trust blindly

### Adding Wallets to Tracker

In the app:
1. Go to: **🎯 DEX Launch Hunter → 🐋 Smart Money**
2. Click "➕ Add Wallet to Track"
3. Fill in:
   - Wallet Address (0x... or Solana address)
   - Name/Label (e.g., "Whale #1 - ETH Gems")
   - Description (optional notes)
   - Chain (ethereum, bsc, solana, etc.)
   - Tags (whale, dev, influencer, etc.)
   - Min Transaction Size ($1,000+ to avoid spam)
4. Click "Add Wallet"

### Wallet Alert Logic

When a tracked wallet:
- Buys a new token
- Transaction > your minimum threshold
- On a chain you're monitoring

**You'll get an alert with:**
- Wallet name
- Token bought
- Amount spent
- Link to transaction
- Token safety analysis

---

## 🐞 Troubleshooting

### Scanner Not Finding Tokens

**Possible Causes:**
1. **DexScreener API Issue**
   - Free tier has delays
   - Check API status: https://status.dexscreener.com/
   - Consider upgrading to paid tier

2. **Filters Too Restrictive**
   - Lower min liquidity
   - Increase max liquidity
   - Raise risk tolerance
   - Lower min composite score

3. **Wrong Chains Selected**
   - Most launches on BSC and ETH
   - Enable more chains in config

**Solution:**
- Try manual token analysis first
- Paste a known new token to verify system works
- Check logs for errors

### Safety Checks Failing

**Possible Causes:**
1. **API Rate Limits**
   - Honeypot.is: 1 request per 2 seconds
   - GoPlus: 10k per day
   - Solution: Add delays between checks

2. **Token Not Yet Indexed**
   - Very new tokens (< 5 minutes) might not be in APIs
   - Wait a few minutes and retry

3. **Unsupported Chain**
   - Some chains not supported by all APIs
   - Check which APIs support your target chain

### Discord Alerts Not Working

**Checklist:**
1. ✅ Webhook URL in `.env`?
2. ✅ `DISCORD_WEBHOOK_URL` format correct?
3. ✅ Alerts enabled in config?
4. ✅ Composite score meeting threshold?
5. ✅ Token passing filters?

**Test Webhook:**
```python
import requests
requests.post(
    "your_webhook_url",
    json={"content": "Test alert from DEX Hunter"}
)
```

### Smart Money Tracker Not Showing Activity

**Possible Causes:**
1. **DeBank API Key Missing**
   - Add `DEBANK_API_KEY` to `.env`

2. **Wallets Not Active**
   - Whales might not be trading during your scan
   - Check wallet on Etherscan to verify activity

3. **Transaction Threshold Too High**
   - Lower min transaction size
   - Current setting shown in wallet details

### High Memory Usage

**If app is slow:**
1. Reduce scan interval (increase from 60s to 120s)
2. Limit tracked wallets (< 50 recommended)
3. Clear old discovered tokens (automatic after 7 days)
4. Reduce max alerts stored (default 500)

---

## 📊 Expected Results

### Realistic Expectations

**Month 1: Learning Phase**
- Find 20-50 potential launches
- Analyze manually
- Make 5-10 small test trades ($10-25 each)
- Expect to lose money
- Goal: Learn to spot scams

**Month 2-3: Pattern Recognition**
- Start seeing patterns in safe vs scam
- Win rate improves to 20-30%
- Find 1-2 promising tokens per week
- Break even or small profit

**Month 4+: Profitable (if you're good)**
- Win rate 30-40%
- Average 1 big winner (10x+) per month
- Several small wins (2-5x)
- Many small losses
- Net positive ROI

### Success Stories (Reality Check)

**What's Possible:**
- $SHIB: 10,000,000x (insane outlier)
- $PEPE: 10,000x in first month
- Many legit tokens: 100-1000x
- Typical winner: 5-20x before major dump

**What's Likely:**
- 50-70% of trades lose money
- 20-30% small wins (1.5-3x)
- 5-10% medium wins (3-10x)
- 1-2% big wins (10x+)
- Overall: Profitable if you catch 1-2 big winners

### Time Commitment

**To use effectively:**
- 1-2 hours daily monitoring
- Quick reactions (minutes matter)
- Active during high-volume hours (US market open)
- Weekend monitoring (crypto never sleeps)

**This is NOT passive income.**

---

## 📚 Additional Resources

### Essential Reading
- [DexScreener Guide](https://docs.dexscreener.com/)
- [Honeypot Detection Guide](https://honeypot.is/docs)
- [Smart Money Tracking Strategies](https://www.nansen.ai/guides/smart-money)

### Tools You'll Need
- **Trust Wallet** - Mobile wallet for DEX trading
- **MetaMask** - Browser wallet alternative
- **Etherscan/BSCScan** - On-chain verification
- **DexScreener** - Price charts and new pairs
- **Telegram** - Alpha groups (be careful of scams)
- **Twitter** - Crypto Twitter for early calls

### Communities (Use Cautiously)
- r/CryptoMoonShots - Reddit (lots of scams, filter carefully)
- Crypto Twitter - Follow @0xQuit, @CryptoKaleo, etc.
- Telegram Alpha Groups - Join 2-3 reputable ones
- Discord Servers - DEX trading communities

---

## ⚖️ Legal Disclaimer

**This tool is for educational purposes only.**

- Not financial advice
- No guarantees of profit
- High risk of total loss
- Do your own research
- Comply with local laws
- Report taxes on gains

**You are responsible for:**
- Your own trading decisions
- Managing your risk
- Securing your wallets
- Verifying token safety
- Following regulations

---

## 🛠️ Support & Updates

### Getting Help

1. **Check Documentation First**
2. **Search Logs** - Look for error messages
3. **GitHub Issues** - Report bugs
4. **Discord** - Community support (if available)

### Feature Roadmap

**Coming Soon:**
- ✅ On-chain event monitoring (real-time new pairs)
- ✅ Multi-chain aggregation (cross-DEX monitoring)
- ✅ Automated sniper bot integration (buy instantly)
- ✅ Portfolio tracking (P&L, win rate, etc.)
- ✅ Social sentiment scoring (Twitter, Reddit, Telegram)
- ✅ Telegram bot alerts
- ✅ Machine learning models (predict pump probability)

**Community Requests:**
- Submit feature ideas
- Vote on priorities
- Contribute code

---

## 🎯 Final Tips

### Do's ✅
- ✅ Start with $10-25 per token
- ✅ Always check contract safety
- ✅ Take profits incrementally
- ✅ Learn from every trade
- ✅ Track all trades in spreadsheet
- ✅ Join alpha communities
- ✅ Follow successful wallets
- ✅ Be patient and disciplined

### Don'ts ❌
- ❌ FOMO into pumps
- ❌ Hold losers hoping for recovery
- ❌ Skip safety checks
- ❌ Trust anonymous teams
- ❌ Invest money you need
- ❌ Trade emotional
- ❌ Chase every new token
- ❌ Ignore red flags

---

**Good luck hunting! May you find the next 1000x. 🚀**

**Remember: Most tokens go to zero. Only risk what you can lose.**

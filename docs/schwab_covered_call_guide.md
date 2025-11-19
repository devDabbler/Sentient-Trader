# Charles Schwab Roth IRA - Covered Call Trading Guide

## Prerequisites

### 1. Enable Options Trading
Before you can trade covered calls, you need options approval:

1. Log in to Charles Schwab
2. Go to **Trade** → **Options** → **Apply for Options Trading**
3. Request **Level 2** approval (required for covered calls)
4. Fill out the options application:
   - Investment experience
   - Financial information
   - Risk tolerance
5. Wait 1-3 business days for approval

**Note:** Roth IRAs typically qualify for Levels 0-2 only (no naked options, no margin).

---

## Step-by-Step: Executing a Covered Call

### Step 1: Buy the Underlying Stock
You must own 100 shares per contract before selling calls.

**Example: SOFI at $27.04**
1. Navigate to **Trade** → **Stocks & ETFs**
2. Enter ticker: `SOFI`
3. Action: **Buy**
4. Quantity: **100** shares (or 200 for 2 contracts, etc.)
5. Order type: **Market** or **Limit**
6. Review & Submit
7. **Wait for fill confirmation**

**Capital Required:** $27.04 × 100 = **$2,704**

---

### Step 2: Sell Call Option (After Owning Stock)
Once you own the shares, you can sell covered calls.

1. Navigate to **Trade** → **Options**
2. Enter ticker: `SOFI`
3. In the options chain:
   - Select **Expiration Date** (e.g., 29 days from today)
   - Find your desired **Strike Price** (e.g., $9.00)
   - Click on the **Call** side (right column)

4. **IMPORTANT:** Select **Sell to Open** (not Buy to Open)
   - Action: **Sell to Open**
   - Quantity: **1** contract (for 100 shares)
   - Order Type: **Limit**
   - Limit Price: Enter the **Bid price** or slightly above for better fill

5. Click **Review Order**

---

### Step 3: Review Order Details
Schwab will show:
- ✅ **Strategy:** Covered Call
- ✅ **Requirements:** 100 shares of SOFI (already owned)
- ✅ **Premium Received:** $X.XX per share × 100 = $XXX
- ✅ **Max Profit:** Premium + (Strike - Current Price) × 100
- ✅ **Max Loss:** If stock goes to $0, you lose stock value minus premium
- ✅ **Breakeven:** Current stock price - premium per share

**Verify:**
- "Sell to Open" (not Buy)
- Correct strike price
- Correct expiration date
- Account shows "Covered" (not "Naked")

---

### Step 4: Submit Order
1. Click **Place Order**
2. Wait for fill confirmation
3. Check **Positions** to verify:
   - **Stock:** 100 shares SOFI
   - **Options:** -1 SOFI CALL $9 (negative = short/sold)

---

## After Order Fills

### Monitoring Your Position

**In Charles Schwab:**
1. Go to **Positions** tab
2. You'll see:
   - **+100 SOFI** @ $27.04
   - **-1 SOFI CALL $9** (Expiry: XX/XX/XXXX)

**Track:**
- Current stock price
- Option value (mark price)
- Days to expiration
- P&L on both stock and option

---

## Three Possible Outcomes

### Outcome 1: Stock Stays Below Strike ($9)
- Option expires **worthless**
- You **keep the premium** + **keep your shares**
- You can sell another call (wheel strategy)

**Example:**
- SOFI drops to $8.50 by expiration
- Call expires worthless
- You keep $1,807.57 premium
- Still own 100 shares at $8.50
- Can sell another call at lower strike

---

### Outcome 2: Stock Stays Above Strike ($9+)
- Option gets **exercised** (shares called away)
- You **sell 100 shares at $9.00**
- You **keep the premium**

**Example:**
- SOFI stays at $27 by expiration
- Buyer exercises call
- Your shares sold automatically at $9.00 = **$900**
- You keep premium: **$1,807.57**
- **Total received:** $2,707.57
- **Cost basis:** $2,704
- **Net profit:** $3.57 (0.13% ROI)

**⚠️ Warning:** This is a **loss** compared to just holding the stock!

---

### Outcome 3: Close Position Early
Before expiration, you can **buy back** the call to close:

1. Navigate to **Trade** → **Options**
2. Find your sold call
3. Action: **Buy to Close**
4. Quantity: **1** contract
5. Pay current **Ask price** to close

**When to close early:**
- Call value dropped 50%+ (take profit)
- Stock rallying above strike (avoid assignment)
- Rolling to new strike/expiration

---

## Important Schwab-Specific Notes

### Roth IRA Restrictions
✅ **Allowed:**
- Covered calls (Level 2)
- Cash-secured puts (Level 2)
- Long calls/puts (Level 1)

❌ **Not Allowed:**
- Naked options (no margin in Roth IRA)
- Spreads requiring margin
- Short selling stocks

### Assignment Process
If your call is **in-the-money (ITM)** at expiration:
1. **Automatic assignment** occurs after market close on expiration Friday
2. Your 100 shares are **sold at strike price**
3. Cash **settles T+1** (next business day)
4. Premium **already in your account** from when you sold the call

**Early Assignment Risk:**
- Deep ITM calls can be exercised **anytime** before expiration
- Schwab will notify you via email if assigned
- Check positions daily if call is deep ITM

### Tax Benefits (Roth IRA)
✅ **All gains are tax-free** (if qualified distribution)
- Premium income: tax-free
- Capital gains: tax-free
- No wash sale rules within Roth IRA

---

## Your Specific Trade: SOFI $9 Strike

### Trade Parameters
- **Stock:** SOFI @ $27.04
- **Strike:** $9.00 (deep ITM)
- **Premium:** $18.08 per share (estimated)
- **Contracts:** 1 (100 shares)
- **Days to Expiry:** 29

### Corrected Calculations

**Capital Required:**
- Buy 100 shares: $27.04 × 100 = **$2,704**

**Premium Received:**
- Sell call: $18.08 × 100 = **$1,808** (approx)

**Max Profit (if called away):**
- Premium: $1,808
- Stock sale at $9: $900
- Total received: $2,708
- Cost: $2,704
- **Net profit: $4** (0.15% ROI)

**Max Loss (if stock goes to $0):**
- Stock loss: $2,704
- Keep premium: +$1,808
- **Net loss: $896**

**Breakeven:**
- $27.04 - $18.08 = **$8.96**
- If stock drops below $8.96, you start losing money

### Analysis & Recommendation

**This is a deep ITM covered call strategy, which is essentially:**
1. Selling your shares at $9.00 with a delayed settlement
2. Collecting massive premium as compensation for below-market sale
3. Extremely low risk (85% downside protection from premium)

**Pros:**
- ✅ 66.8% downside protection ($27.04 → $8.96 breakeven)
- ✅ Guaranteed $4 profit if assigned (0.15% ROI in 29 days)
- ✅ Very low risk

**Cons:**
- ❌ Capped upside at $9.00 (giving up $18+ in potential gains)
- ❌ Nearly zero profit if called away
- ❌ Early assignment risk (likely immediate assignment)

**Better Alternatives:**
If you're **bullish on SOFI**, consider:
1. **$30 strike call** → Collect ~$2.50/share premium, keep upside above $30
2. **$35 strike call** → Collect ~$1.50/share premium, keep upside above $35
3. **No call at all** → Keep 100% upside if SOFI continues rallying

If you're **bearish or neutral**, this $9 strike makes sense as a "synthetic sale" with downside protection.

---

## Troubleshooting

### "Insufficient shares" Error
- **Fix:** Buy 100 shares first, wait for settlement, then sell call

### "Account not approved for options"
- **Fix:** Apply for Level 2 options approval (see Prerequisites)

### "Cannot trade in Roth IRA"
- **Fix:** Ensure Level 2 approval for your Roth IRA specifically

### Option Not Available
- **Fix:** Check if SOFI has weekly or monthly options for your desired date

### Early Assignment Surprise
- **Check:** Positions daily if call is deep ITM
- **Action:** Buy to close if you want to keep shares

---

## Contact Support

If you need help:
1. **Schwab Chat:** Available 24/7 at schwab.com
2. **Phone:** 1-800-435-4000
3. **In-Branch:** Find local branch at schwab.com/branches

---

## Summary Checklist

Before executing your SOFI $9 covered call:

- [ ] Level 2 options approval for Roth IRA
- [ ] Own 100 shares of SOFI in Roth IRA
- [ ] Understand max profit = ~$4 (0.15% ROI)
- [ ] Accept that shares will be called away at $9
- [ ] Understand breakeven = $8.96
- [ ] Know early assignment is likely (deep ITM)
- [ ] Have plan for what to do with cash after assignment

**Questions?** Run the analysis again in Sentient Trader with corrected calculations!

# Option Alpha + Tradier Paper Trading Setup Guide

This guide will walk you through setting up the integration between Option Alpha webhooks and your Tradier paper trading account.

## Prerequisites

- Option Alpha account with webhook access
- Tradier account (free sandbox account available)
- Python environment with the required packages

## Step 1: Create Your Environment Variables

1. Copy the `config_template.env` file to `.env`:
   ```bash
   copy config_template.env .env
   ```

2. Edit the `.env` file with your actual credentials (see steps below for obtaining them).

## Step 2: Option Alpha Webhook Setup

### 2.1 Access Option Alpha Webhooks
1. Log into your Option Alpha account
2. Navigate to **Settings** → **Webhooks**
3. Click **"New Webhook"** button

### 2.2 Create Your Webhook
1. **Name**: Give your webhook a descriptive name (e.g., "AI Trading Bot")
2. **Description**: Optional description of what this webhook does
3. **Webhook URL**: This will be generated for you - copy it!
4. **Security**: Keep this URL secret - it contains your unique access code

### 2.3 Configure Bot Automation
1. Go to **Bots** section in Option Alpha
2. Create or edit a bot that will receive the webhook signals
3. Add automation: **"When webhook is triggered"**
4. Configure the bot's trading strategy (e.g., sell puts, covered calls, etc.)
5. Link the webhook to your bot

### 2.4 Update Your .env File
```env
OPTION_ALPHA_WEBHOOK_URL=https://app.optionalpha.com/api/webhooks/YOUR_ACTUAL_WEBHOOK_ID
```

## Step 3: Tradier Paper Trading Account Setup

### 3.1 Create Tradier Sandbox Account
1. Go to [Tradier Developer Portal](https://developer.tradier.com/)
2. Sign up for a free developer account
3. Navigate to **Sandbox Account Access**

### 3.2 Generate Sandbox Account
1. Click **"+ Generate Account!"** button
2. Copy the generated **Account Number**
3. Click **"Create"** to generate your **Access Token**
4. **Important**: Use `sandbox.tradier.com` API endpoint for all requests

### 3.3 Update Your .env File
```env
TRADIER_ACCOUNT_ID=your_generated_account_number
TRADIER_ACCESS_TOKEN=your_generated_access_token
TRADIER_API_URL=https://sandbox.tradier.com
```

## Step 4: Test Your Setup

### 4.1 Test Option Alpha Webhook
Run this Python script to test your webhook:

```python
import requests
import json

webhook_url = "YOUR_OPTION_ALPHA_WEBHOOK_URL"

test_signal = {
    "ticker": "SPY",
    "action": "SELL_PUT",
    "strike": 450,
    "expiry": "2024-02-16",
    "qty": 1,
    "note": "Test signal from AI bot"
}

response = requests.post(webhook_url, json=test_signal)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
```

### 4.2 Test Tradier API Connection
```python
import requests

account_id = "YOUR_TRADIER_ACCOUNT_ID"
access_token = "YOUR_TRADIER_ACCESS_TOKEN"
api_url = "https://sandbox.tradier.com"

headers = {
    'Authorization': f'Bearer {access_token}',
    'Accept': 'application/json'
}

# Test account balance
response = requests.get(f'{api_url}/v1/accounts/{account_id}/balances', headers=headers)
print(f"Account Balance: {response.json()}")
```

## Step 5: Integration Configuration

### 5.1 Webhook Rate Limits
- **Option Alpha**: Maximum 1 trigger per minute per webhook
- **Tradier**: 120 requests per minute for sandbox

### 5.2 Security Best Practices
1. **Never share your webhook URLs publicly**
2. **Keep access tokens secure**
3. **Use environment variables for all credentials**
4. **Test thoroughly in paper mode before live trading**

## Step 6: Running Your Application

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

3. Configure your settings in the sidebar:
   - Set **Paper Trading Mode** to `True` initially
   - Enter your **Option Alpha Webhook URL**
   - Configure **Risk Limits**

## Step 7: Monitoring and Maintenance

### 7.1 Logs
- Check `trading_signals.log` for signal activity
- Monitor Option Alpha bot performance
- Review Tradier account activity

### 7.2 Troubleshooting
- **Webhook not triggering**: Check URL and bot configuration
- **API errors**: Verify credentials and rate limits
- **Paper trading issues**: Ensure using sandbox.tradier.com endpoint

## Next Steps

1. **Test thoroughly** in paper mode
2. **Monitor performance** for at least a week
3. **Adjust parameters** based on results
4. **Consider live trading** only after successful paper trading

## Support Resources

- [Option Alpha Webhook Documentation](https://help.optionalpha.com/)
- [Tradier API Documentation](https://developer.tradier.com/documentation)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Important Notes

⚠️ **Always test in paper mode first!**
⚠️ **Keep your credentials secure!**
⚠️ **Monitor your trades regularly!**
⚠️ **Understand the risks before live trading!**

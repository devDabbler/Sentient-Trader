# 🚀 Server Deployment Checklist - 24/7 Operations

**Created:** Nov 24, 2024  
**Status:** Ready for deployment with recommendations

---

## ✅ WHAT'S READY (Already Configured)

### 1. **Monitoring Services** ✓
- **Stock Monitor** - Scans watchlist for opportunities (alerts only)
- **Crypto Breakout Monitor** - Detects crypto breakout patterns
- **DEX Launch Monitor** - Monitors new DEX token launches

### 2. **Infrastructure** ✓
- Systemd service definitions with auto-restart
- Cross-platform runners (Windows + Linux)
- Comprehensive logging (50MB rotation, 30-day retention)
- Lazy imports fix for environment compatibility
- Error recovery with 3-retry logic

### 3. **AI Capabilities** ✓
- Multi-agent trading system (orchestrator, risk, sentiment, technical)
- AI-powered position management framework
- Multiple LLM providers (OpenRouter, Anthropic, Ollama)
- Cost-effective free models configured (Gemini 2.0 Flash)

---

## ⚠️ CRITICAL - MUST DO BEFORE DEPLOYMENT

### 1. **ADD AI TRADING SERVICE** (NEW)

**Created:** `windows_services/runners/run_crypto_ai_position_manager_simple.py`

This service monitors open crypto positions 24/7 and uses AI to make intelligent exit decisions.

**Features:**
- Real-time position monitoring (every 60 seconds)
- AI-powered exit decisions with multi-agent analysis
- Trailing stops and break-even protection
- Partial profit taking strategies
- Configurable confidence thresholds

**Safety Controls:**
- Set `require_manual_approval=True` for monitoring only
- Set `require_manual_approval=False` for auto-execution ⚠️
- Respects `CRYPTO_PAPER_TRADING` mode from `.env`

**Status:** Service created but DISABLED by default in deployment script (safe)

---

### 2. **VERIFY ENVIRONMENT VARIABLES**

Your `.env` is comprehensive locally, but ensure server has:

```bash
# CRITICAL FOR 24/7 OPERATION
KRAKEN_API_KEY=...
KRAKEN_API_SECRET=...
OPENROUTER_API_KEY=...
DISCORD_WEBHOOK_URL=...

# IMPORTANT FOR MONITORING
COINGECKO_API_KEY=...
ETH_RPC_URL=...
BSC_RPC_URL=...
SOLANA_RPC_URL=...

# SAFETY CONFIGURATION
CRYPTO_PAPER_TRADING=True  # ⚠️ Set to False only when ready for live trading
IS_PAPER_TRADING=True
PAPER_TRADING_MODE=True

# AI MODEL CONFIGURATION
AI_CONFIDENCE_MODEL=google/gemini-2.0-flash-exp:free
AI_TRADING_MODEL=google/gemini-2.0-flash-exp:free
AI_ANALYZER_MODEL=google/gemini-2.0-flash-exp:free
```

---

### 3. **ENABLE AI TRADING SERVICE** (OPTIONAL)

Only enable if you want AI to actively manage positions:

**Edit:** `deploy/setup_systemd_services.sh`

**Uncomment these lines:**
```bash
# Line 111
sudo systemctl enable sentient-crypto-ai-trader

# Line 121
sudo systemctl start sentient-crypto-ai-trader
```

**Test first with:**
```bash
# Manual test on server
cd /root/sentient-trader
source venv/bin/activate
python3 windows_services/runners/run_crypto_ai_position_manager_simple.py
# Press Ctrl+C after verifying it starts successfully
```

---

### 4. **ADD HEALTH MONITORING** (HIGHLY RECOMMENDED)

**Missing:** External health check system

**Options:**

**A. UptimeRobot (Free)**
- Sign up: https://uptimerobot.com/
- Monitor: Your server IP every 5 minutes
- Alert: Email/SMS/Discord on downtime

**B. Healthchecks.io (Free)**
- Sign up: https://healthchecks.io/
- Create check with 15-minute interval
- Add cron job on server:

```bash
# Add to crontab (crontab -e):
*/10 * * * * curl -fsS --retry 3 https://hc-ping.com/YOUR-PING-KEY > /dev/null
```

**C. Discord Heartbeat (Custom)**
Create a simple heartbeat service that pings Discord every hour with status.

---

### 5. **CONFIGURE LOG MONITORING**

**Recommendation:** Set up log aggregation to catch errors early

**Option 1: Simple Discord Alerts**
```bash
# Add to crontab (runs every hour)
0 * * * * grep -i "error\|fatal\|crash" /root/sentient-trader/logs/*.log | tail -20 | curl -X POST -H 'Content-Type: application/json' -d "{\"content\": \"\`\`\`$(cat -)\`\`\`\"}" YOUR_DISCORD_WEBHOOK
```

**Option 2: Logrotate with alerts**
Already configured for rotation, but add alert on critical errors.

---

## 📋 PRE-DEPLOYMENT TESTING

### **Local Testing (Do First)**

```powershell
# On Windows - Test all services locally
cd "C:\Users\seaso\Sentient Trader"

# Test Stock Monitor
python windows_services\runners\run_stock_monitor_simple.py
# Verify: Sees "SERVICE READY" and starts scanning

# Test Crypto Breakout
python windows_services\runners\run_crypto_breakout_simple.py
# Verify: Initializes without errors

# Test DEX Launch
python windows_services\runners\run_dex_launch_simple.py
# Verify: Starts monitoring announcements

# Test AI Position Manager (NEW)
python windows_services\runners\run_crypto_ai_position_manager_simple.py
# Verify: Kraken client connects, AI manager initializes
```

---

## 🚀 DEPLOYMENT STEPS

### **Step 1: Upload Code**

```powershell
# On Windows
cd "C:\Users\seaso\Sentient Trader"

# Create clean deployment package
$items = @("*.py", "*.md", "*.txt", "*.json", "analyzers", "clients", 
           "deploy", "models", "services", "ui", "utils", 
           "windows_services", ".streamlit")

Compress-Archive -Path $items -DestinationPath sentient-trader.zip -Force

# Upload to server
scp sentient-trader.zip root@YOUR_SERVER_IP:/root/sentient-trader/
```

### **Step 2: Server Setup**

```bash
# On server
cd /root/sentient-trader
unzip -o sentient-trader.zip
rm sentient-trader.zip

# Create venv if not exists
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p logs data backups
```

### **Step 3: Configure Environment**

```bash
# Copy your local .env or create new one
nano .env

# CRITICAL: Add all API keys
# CRITICAL: Set CRYPTO_PAPER_TRADING=True initially
# Save: Ctrl+X, Y, Enter

# Secure permissions
chmod 600 .env
```

### **Step 4: Deploy Services**

```bash
# Run setup script
chmod +x deploy/setup_systemd_services.sh
./deploy/setup_systemd_services.sh

# This will:
# - Create systemd service files
# - Enable services to start on boot
# - Start all monitoring services
# - Show service status
```

### **Step 5: Verify Deployment**

```bash
# Check all services are running
sudo systemctl status sentient-stock-monitor --no-pager
sudo systemctl status sentient-crypto-breakout --no-pager
sudo systemctl status sentient-dex-launch --no-pager

# Watch logs (live)
tail -f logs/stock_monitor_service.log

# Look for: "SERVICE READY" and scan activity
# Should see scans starting within minutes
```

---

## 🔍 POST-DEPLOYMENT MONITORING (First 24 Hours)

### **Hour 1: Immediate Checks**

```bash
# Are all services running?
systemctl list-units --type=service --state=running | grep sentient

# Any errors in logs?
grep -i "error\|fatal\|crash" logs/*.log | tail -50

# Are scans happening?
grep "scan\|alert" logs/*.log | tail -20
```

### **Hour 6: Stability Check**

```bash
# Check service uptime
systemctl status sentient-* --no-pager | grep "Active:"

# Check memory usage
free -h
ps aux | grep python | awk '{print $6}' | awk '{sum+=$1} END {print sum/1024 " MB"}'

# Review alerts sent
grep "DISCORD\|ALERT\|HIGH SCORE" logs/*.log | tail -30
```

### **Hour 24: Full Review**

```bash
# Service restart count (should be 0)
sudo journalctl -u sentient-stock-monitor | grep -i "restart\|stopped"

# Successful scan count
grep "Scan complete\|Scan #" logs/stock_monitor_service.log | wc -l

# Alert count
grep "HIGH SCORE\|ALERT" logs/*.log | wc -l

# Error patterns
grep -i "error" logs/*.log | cut -d':' -f4- | sort | uniq -c | sort -rn
```

---

## ⚙️ OPTIMIZATION RECOMMENDATIONS

### **1. Memory Management**

**Current:** 1GB droplet may be tight with AI features

**Monitor:**
```bash
# Check memory every hour
watch -n 3600 free -h
```

**If memory issues:**
- Upgrade to 2GB ($12/month) or 4GB ($24/month)
- Or add swap:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### **2. Scan Intervals**

**Current settings:**
- Stock Monitor: 30 minutes
- Crypto Breakout: 15 minutes (300s)
- DEX Launch: 5 minutes (300s)
- AI Position Manager: 1 minute (60s)

**Optimization:**
For cost/performance balance on free LLM tier:
- Reduce AI scan frequency if hitting rate limits
- Increase intervals if no positions to monitor
- Use caching aggressively

### **3. Rate Limit Protection**

**Already configured in `.env`:**
```bash
KRAKEN_RATE_LIMIT_DELAY=0.5
COINGECKO_RATE_LIMIT=10
```

**Monitor for 429 errors:**
```bash
grep "429\|rate limit\|too many requests" logs/*.log
```

### **4. AI Cost Management**

**Using free models (good!):**
- `google/gemini-2.0-flash-exp:free`

**If upgrading to paid:**
- Set daily budget in OpenRouter
- Monitor costs: https://openrouter.ai/account/usage
- Implement request throttling

### **5. Backup Strategy**

**Recommended:**
```bash
# Add to crontab (daily at 3 AM)
0 3 * * * cd /root/sentient-trader && tar -czf backups/backup-$(date +\%Y\%m\%d).tar.gz data/ logs/*.log .env
```

**Auto-cleanup old backups:**
```bash
# Keep last 7 days
0 4 * * * find /root/sentient-trader/backups -name "backup-*.tar.gz" -mtime +7 -delete
```

---

## 🔒 SECURITY HARDENING

### **1. Firewall Configuration**

```bash
# Enable firewall
sudo ufw enable

# Allow only SSH
sudo ufw allow OpenSSH

# If running Streamlit UI (optional)
sudo ufw allow 8501/tcp
```

### **2. SSH Key Authentication**

```bash
# On server
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add your public key
nano ~/.ssh/authorized_keys
# Paste your public key
chmod 600 ~/.ssh/authorized_keys

# Disable password auth (optional, be careful!)
# sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
# sudo systemctl restart sshd
```

### **3. API Key Security**

```bash
# Verify .env permissions
ls -l .env
# Should show: -rw------- (600)

# Never commit .env
grep -q ".env" .gitignore || echo ".env" >> .gitignore
```

---

## 🆘 TROUBLESHOOTING GUIDE

### **Services Won't Start**

```bash
# Check detailed errors
sudo journalctl -u sentient-stock-monitor -n 100 --no-pager

# Test manually
cd /root/sentient-trader
source venv/bin/activate
python3 windows_services/runners/run_stock_monitor_simple.py
# Watch for error messages
```

### **Import Errors**

```bash
# Reinstall dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Check Python version
python3 --version  # Should be 3.11+

# Verify key packages
pip list | grep -E "loguru|requests|pandas|kraken"
```

### **Kraken Connection Failures**

```bash
# Test Kraken API
python3 -c "
from clients.kraken_client import get_kraken_client
client = get_kraken_client()
print(client.check_connection())
"
```

### **AI Service Not Working**

```bash
# Check LLM connectivity
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('OpenRouter Key:', os.getenv('OPENROUTER_API_KEY')[:20] + '...')
"

# Test AI model
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

### **High Memory Usage**

```bash
# Identify memory hogs
ps aux --sort=-%mem | head -10

# Check for memory leaks
watch -n 60 'ps aux | grep python'

# Restart service if needed
sudo systemctl restart sentient-stock-monitor
```

---

## 📊 SUCCESS METRICS

After 24 hours, you should see:

- ✅ All services show "active (running)" status
- ✅ No crash/restart entries in journalctl
- ✅ Scans completing successfully every interval
- ✅ Discord alerts being sent for opportunities
- ✅ Memory usage stable under 700MB (1GB droplet)
- ✅ No repeated error patterns in logs
- ✅ Services survive server reboot test

---

## 🎯 NEXT STEPS AFTER STABLE DEPLOYMENT

### **Week 1:**
1. Monitor daily for stability
2. Tune scan intervals based on results
3. Review alert quality (false positives?)
4. Optimize watchlists based on alerts

### **Week 2:**
5. Enable AI position manager (if desired)
6. Test with paper trading first
7. Monitor AI decision quality
8. Adjust confidence thresholds

### **Month 1:**
9. Review cost analysis (LLM usage, server costs)
10. Consider upgrading to paid LLM if needed
11. Implement advanced features (backtesting results)
12. Plan transition to live trading (if applicable)

---

## 📞 SUPPORT RESOURCES

**Documentation:**
- Main README: `README.md`
- Windows Services: `windows_services/README.md`
- Deployment: `deploy/README_DEPLOYMENT.md`
- Quick Deploy: `deploy/QUICK_DEPLOY.md`

**Logs Location:**
- `/root/sentient-trader/logs/`
- Stock: `stock_monitor_service.log`
- Crypto: `crypto_breakout_service.log`
- DEX: `dex_launch_service.log`
- AI: `crypto_ai_position_manager_service.log`

**Common Commands:**
```bash
# Status
sudo systemctl status sentient-*

# Restart all
sudo systemctl restart sentient-stock-monitor sentient-crypto-breakout sentient-dex-launch

# View logs
tail -f logs/*.log

# Check errors
grep -i error logs/*.log | tail -50
```

---

**🎉 Your application is production-ready! Deploy with confidence.**

**Last Updated:** Nov 24, 2024

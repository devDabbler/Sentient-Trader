# 🚀 Deploy Sentient Trader to Digital Ocean

**Current Status:** You're connected to your server and ready to deploy!

This guide will help you deploy your entire Sentient Trader application with all background services running 24/7.

---

## 📋 What Gets Deployed

✅ **Background Services:**
- Stock Monitor (scans stocks from watchlist)
- Crypto Breakout Monitor (detects crypto breakouts)
- DEX Launch Monitor (monitors new DEX launches)

✅ **Optional:**
- Streamlit Web UI (access your dashboard remotely)

✅ **Features:**
- Auto-restart on crash
- Auto-start on server reboot
- Persistent logging
- All your trading logic and AI analysis

---

## 🎯 Quick Start (3 Steps)

### Step 1: Prepare Server (You Are Here ✅)

**On your Digital Ocean server**, run this single command:

```bash
wget -O - https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/deploy/first_time_server_setup.sh | bash
```

**OR** create the setup script manually:

```bash
# Create project directory
mkdir -p /root/sentient-trader
cd /root/sentient-trader

# Download and run setup
cat > setup.sh << 'EOF'
#!/bin/bash
set -e
apt-get update && apt-get upgrade -y
apt-get install -y python3.11 python3.11-venv python3-pip git unzip wget curl
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
mkdir -p logs data backups
echo "✅ Server ready!"
EOF

chmod +x setup.sh
./setup.sh
```

---

### Step 2: Upload Your Code

**On your local Windows machine**, open PowerShell:

```powershell
cd "C:\Users\seaso\Sentient Trader"

# Run upload script (replace YOUR_SERVER_IP)
.\deploy\upload_to_server.ps1 -ServerIP YOUR_SERVER_IP
```

This will:
- Create a clean deployment package
- Upload only necessary files (no venv, logs, cache)
- Show you the next steps

**Alternative: Manual Upload**

```powershell
# Create zip
Compress-Archive -Path * -DestinationPath deploy.zip -Force -Exclude venv,__pycache__,.git,logs,data

# Upload
scp deploy.zip root@YOUR_SERVER_IP:/root/sentient-trader/

# Clean up
Remove-Item deploy.zip
```

---

### Step 3: Deploy on Server

**Back on your Digital Ocean server:**

```bash
cd /root/sentient-trader

# Extract files
unzip -o sentient-trader-deploy.zip
rm sentient-trader-deploy.zip

# Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Configure environment (IMPORTANT!)
nano .env
# Add your API keys, then save: Ctrl+X, Y, Enter

# Deploy services
chmod +x deploy/setup_systemd_services.sh
./deploy/setup_systemd_services.sh
```

---

## 🔐 Required Environment Variables

Edit `/root/sentient-trader/.env` and add:

```env
# LLM API Keys (at least one required)
OPENROUTER_API_KEY=sk-or-v1-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Discord Alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Broker API Keys (if using trading features)
TRADIER_API_KEY=...
IBKR_USERNAME=...
IBKR_PASSWORD=...
KRAKEN_API_KEY=...
KRAKEN_API_SECRET=...

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

---

## ✅ Verify Deployment

### Check Services

```bash
# Status of all services
sudo systemctl status sentient-stock-monitor --no-pager
sudo systemctl status sentient-crypto-breakout --no-pager
sudo systemctl status sentient-dex-launch --no-pager

# View live logs
tail -f logs/stock_monitor_service.log
```

**Look for:** `🚀 SERVICE READY` in the logs

### Test a Service

```bash
# View last 50 log lines
tail -n 50 logs/stock_monitor_service.log

# Search for activity
grep "Scanning" logs/stock_monitor_service.log
grep "HIGH SCORE" logs/stock_monitor_service.log
grep "ERROR" logs/stock_monitor_service.log
```

---

## 🎛️ Management Commands

### Start/Stop/Restart

```bash
# Restart a service
sudo systemctl restart sentient-stock-monitor

# Stop a service
sudo systemctl stop sentient-crypto-breakout

# Start a service
sudo systemctl start sentient-dex-launch

# Restart all services
sudo systemctl restart sentient-stock-monitor sentient-crypto-breakout sentient-dex-launch
```

### View Logs

```bash
# Live tail (Ctrl+C to exit)
tail -f logs/stock_monitor_service.log

# Last 100 lines
tail -n 100 logs/crypto_breakout_service.log

# Search for errors
grep "ERROR" logs/*.log

# View systemd journal
sudo journalctl -u sentient-stock-monitor -n 50 --no-pager
```

### Update Code

When you make changes locally:

```powershell
# On Windows
cd "C:\Users\seaso\Sentient Trader"
.\deploy\upload_to_server.ps1 -ServerIP YOUR_SERVER_IP
```

```bash
# On Server
cd /root/sentient-trader
unzip -o /root/sentient-trader-deploy.zip
rm /root/sentient-trader-deploy.zip
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart sentient-stock-monitor sentient-crypto-breakout sentient-dex-launch
```

---

## 🌐 Optional: Deploy Streamlit UI

To access your trading dashboard remotely:

```bash
cd /root/sentient-trader
source venv/bin/activate

# Create Streamlit service
sudo tee /etc/systemd/system/sentient-streamlit.service > /dev/null << EOF
[Unit]
Description=Sentient Trader - Streamlit UI
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/sentient-trader
Environment="PATH=/root/sentient-trader/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/root/sentient-trader/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start Streamlit
sudo systemctl daemon-reload
sudo systemctl enable sentient-streamlit
sudo systemctl start sentient-streamlit

# Open firewall
sudo ufw allow 8501/tcp
sudo ufw reload
```

**Access at:** `http://YOUR_SERVER_IP:8501`

⚠️ **Security Note:** This exposes your dashboard publicly. For production:
1. Use SSH tunnel: `ssh -L 8501:localhost:8501 root@YOUR_SERVER_IP`
2. Access at: `http://localhost:8501`
3. Or setup Nginx with password protection

---

## 🔍 Troubleshooting

### Services Won't Start

```bash
# Check detailed error logs
sudo journalctl -u sentient-stock-monitor -n 100 --no-pager
cat logs/stock_monitor_error.log

# Check Python errors
source venv/bin/activate
python3 windows_services/runners/run_stock_monitor_simple.py
# (Press Ctrl+C after you see the issue)
```

### Import Errors

```bash
# Reinstall dependencies
cd /root/sentient-trader
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Out of Memory

Your $6/month droplet has 1GB RAM. If you see memory errors:

1. **Check memory usage:**
   ```bash
   free -h
   htop  # (install with: apt install htop)
   ```

2. **Upgrade droplet:**
   - DigitalOcean Dashboard → Your Droplet → Resize
   - Choose 2GB ($12/month) or 4GB ($24/month)

3. **Add swap space (temporary fix):**
   ```bash
   fallocate -l 2G /swapfile
   chmod 600 /swapfile
   mkswap /swapfile
   swapon /swapfile
   ```

### Service Logs Not Updating

```bash
# Check if service is actually running
ps aux | grep python

# Manually test the runner
cd /root/sentient-trader
source venv/bin/activate
python3 windows_services/runners/run_stock_monitor_simple.py
```

---

## 📊 What's Running?

After successful deployment:

| Service | Purpose | Log File | Port |
|---------|---------|----------|------|
| `sentient-stock-monitor` | Stock watchlist monitoring | `logs/stock_monitor_service.log` | - |
| `sentient-crypto-breakout` | Crypto breakout detection | `logs/crypto_breakout_service.log` | - |
| `sentient-dex-launch` | DEX launch monitoring | `logs/dex_launch_service.log` | - |
| `sentient-streamlit` (optional) | Web dashboard | - | 8501 |

---

## 💰 Costs

| Component | Monthly Cost |
|-----------|-------------|
| 1GB Droplet (sufficient for monitoring) | $6 |
| 2GB Droplet (recommended for Streamlit) | $12 |
| 4GB Droplet (if running heavy analysis) | $24 |
| Bandwidth (1TB included) | $0 |
| **Total (recommended)** | **$12/month** |

Compare to running your Windows PC 24/7:
- Electricity: ~$20-40/month
- Wear on hardware: $$
- Internet/power outages: downtime

---

## 🔒 Security Best Practices

1. **Use SSH Keys (not passwords)**
   ```bash
   # On server
   mkdir -p ~/.ssh
   nano ~/.ssh/authorized_keys
   # Paste your public key
   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/authorized_keys
   ```

2. **Setup Firewall**
   ```bash
   sudo ufw allow OpenSSH
   sudo ufw enable
   ```

3. **Keep System Updated**
   ```bash
   apt-get update && apt-get upgrade -y
   ```

4. **Protect API Keys**
   - Never commit `.env` to git
   - Use restrictive file permissions: `chmod 600 .env`

---

## 📚 Additional Resources

- **Digital Ocean Docs:** https://docs.digitalocean.com/
- **Systemd Services:** `man systemd.service`
- **Python Venv:** https://docs.python.org/3/library/venv.html
- **Streamlit Deployment:** https://docs.streamlit.io/deploy

---

## 🆘 Need Help?

If you encounter issues:

1. **Check logs first:**
   ```bash
   tail -f logs/stock_monitor_service.log
   sudo journalctl -u sentient-stock-monitor -n 100
   ```

2. **Test manually:**
   ```bash
   cd /root/sentient-trader
   source venv/bin/activate
   python3 windows_services/runners/run_stock_monitor_simple.py
   ```

3. **Verify environment:**
   ```bash
   cat .env  # Check API keys are set
   which python3  # Should be /root/sentient-trader/venv/bin/python3
   pip list  # Check installed packages
   ```

---

## ✨ Success Checklist

- [ ] Server setup complete (Python, venv, directories)
- [ ] Code uploaded to `/root/sentient-trader`
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with API keys
- [ ] Systemd services created and started
- [ ] Services show "SERVICE READY" in logs
- [ ] Logs show scanning activity
- [ ] Services auto-restart on reboot (test with `sudo reboot`)

---

**🎉 Once everything is green, your app is running 24/7 in the cloud!**

Monitor for 24 hours to ensure stability, then set up GitHub for easier updates.

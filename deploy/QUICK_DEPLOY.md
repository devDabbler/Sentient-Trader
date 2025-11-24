# Quick Deploy to Digital Ocean (No GitHub Required)

## You Are Here: Connected to Server ✅

You've successfully SSH'd into your server. Now follow these steps:

---

## Step 1: Setup Server Environment

Run this command on your **Digital Ocean server**:

```bash
# Create and run setup script
mkdir -p /root/sentient-trader && cd /root/sentient-trader && \
cat > setup.sh << 'EOF'
#!/bin/bash
set -e
echo "Installing dependencies..."
apt-get update
apt-get install -y python3.11 python3.11-venv python3-pip git unzip wget curl
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
mkdir -p logs data backups
echo "✅ Server ready for file upload!"
EOF
chmod +x setup.sh && ./setup.sh
```

---

## Step 2: Upload Files from Windows

On your **local Windows machine**, open PowerShell:

```powershell
# Navigate to project
cd "C:\Users\seaso\Sentient Trader"

# Option A: Create clean zip (recommended)
# Exclude unnecessary files
$items = @(
    "*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml",
    "analyzers", "clients", "deploy", "docs", "models", 
    "services", "src", "ui", "utils", "windows_services",
    ".env", ".streamlit"
)

# Create temporary directory
$tempDir = "C:\Temp\sentient-deploy"
New-Item -ItemType Directory -Force -Path $tempDir
foreach ($item in $items) {
    Copy-Item -Path $item -Destination $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Zip it
Compress-Archive -Path "$tempDir\*" -DestinationPath sentient-trader.zip -Force
Remove-Item -Recurse -Force $tempDir

# Option B: Quick zip (includes everything except venv)
# Compress-Archive -Path * -DestinationPath sentient-trader.zip -Force -Exclude venv,__pycache__,.git

# Upload to server (REPLACE YOUR_SERVER_IP)
scp sentient-trader.zip root@YOUR_SERVER_IP:/root/sentient-trader/

# Clean up
Remove-Item sentient-trader.zip
```

**Alternative**: Use WinSCP or FileZilla for GUI upload
- Download: https://winscp.net/
- Connect to your server IP
- Upload all files to `/root/sentient-trader/`

---

## Step 3: Extract and Install

Back on your **Digital Ocean server**:

```bash
cd /root/sentient-trader
unzip -o sentient-trader.zip
rm sentient-trader.zip

# Install Python dependencies
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 4: Configure Environment

```bash
# Edit .env file
nano .env
```

**Required variables:**
```env
# API Keys
OPENROUTER_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
DISCORD_WEBHOOK_URL=your_webhook

# Environment
ENVIRONMENT=production
```

Save: `Ctrl+X`, then `Y`, then `Enter`

---

## Step 5: Setup Background Services

```bash
cd /root/sentient-trader
chmod +x deploy/setup_systemd_services.sh
./deploy/setup_systemd_services.sh
```

This creates and starts:
- `sentient-stock-monitor` - Stock monitoring service
- `sentient-crypto-breakout` - Crypto breakout detection
- `sentient-dex-launch` - DEX launch monitoring

---

## Step 6: Verify Services

```bash
# Check service status
sudo systemctl status sentient-stock-monitor --no-pager
sudo systemctl status sentient-crypto-breakout --no-pager
sudo systemctl status sentient-dex-launch --no-pager

# View live logs
tail -f logs/stock_monitor_service.log
```

Look for: `🚀 SERVICE READY` in the logs

---

## Optional: Setup Streamlit Web UI

If you want to access the Streamlit UI remotely:

```bash
# Install streamlit in venv
source venv/bin/activate
pip install streamlit

# Create systemd service for Streamlit
sudo tee /etc/systemd/system/sentient-streamlit.service > /dev/null << EOF
[Unit]
Description=Sentient Trader - Streamlit UI
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/sentient-trader
Environment="PATH=/root/sentient-trader/venv/bin"
ExecStart=/root/sentient-trader/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
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
```

Access at: `http://YOUR_SERVER_IP:8501`

---

## Management Commands

### View Logs
```bash
tail -f logs/stock_monitor_service.log
tail -n 100 logs/crypto_breakout_service.log
grep "HIGH SCORE" logs/stock_monitor_service.log
```

### Restart Services
```bash
sudo systemctl restart sentient-stock-monitor
sudo systemctl restart sentient-crypto-breakout
sudo systemctl restart sentient-dex-launch
```

### Stop Services
```bash
sudo systemctl stop sentient-stock-monitor
```

### Update Code
```bash
cd /root/sentient-trader
# Upload new zip from Windows, then:
unzip -o sentient-trader.zip
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart sentient-stock-monitor sentient-crypto-breakout sentient-dex-launch
```

---

## Troubleshooting

### Services won't start
```bash
# Check detailed logs
sudo journalctl -u sentient-stock-monitor -n 100 --no-pager
cat logs/stock_monitor_error.log
```

### Import errors
```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Out of memory
Upgrade droplet to 2GB+ in Digital Ocean dashboard

---

## Summary

✅ **What's Running:**
- 3 background services monitoring stocks/crypto
- All logs saved to `logs/` directory
- Services auto-restart on crash
- Services auto-start on server reboot

✅ **Costs:**
- $6/month for 1GB droplet (sufficient for monitoring services)
- Upgrade to $12/month for 2GB if running Streamlit UI

✅ **Next Steps:**
1. Monitor logs for 24 hours
2. Set up GitHub repo for easier updates (optional)
3. Add monitoring (UptimeRobot, Healthchecks.io)

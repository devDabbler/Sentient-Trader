# DigitalOcean Setup Guide

## Step 1: Create Droplet

1. Go to: https://cloud.digitalocean.com/droplets/new
2. Choose:
   - **Image**: Ubuntu 22.04 LTS (x64)
   - **Plan**: Basic - $6/month (1GB RAM / 1 CPU)
   - **Region**: Closest to you (e.g., San Francisco, New York)
   - **Authentication**: SSH Key (recommended) or Password
   - **Hostname**: sentient-trader
3. Click **Create Droplet**
4. Note the **IP address** (e.g., `164.92.xxx.xxx`)

---

## Step 2: Connect to Server

### From Windows (PowerShell):

```powershell
# If using SSH key:
ssh root@YOUR_DROPLET_IP

# If using password:
ssh root@YOUR_DROPLET_IP
# Enter password when prompted
```

---

## Step 3: Run Setup Script

Once connected to the server:

```bash
# Download setup script
wget https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/deploy/deploy_to_digitalocean.sh

# Make executable
chmod +x deploy_to_digitalocean.sh

# Run setup
./deploy_to_digitalocean.sh
```

---

## Step 4: Upload Your Code

### Option A: Git Clone (Recommended)

```bash
cd /home/$USER/sentient-trader
git init
git remote add origin YOUR_GIT_REPO_URL
git pull origin main
```

### Option B: SCP from Windows

From your local Windows PowerShell:

```powershell
# Compress project
Compress-Archive -Path "C:\Users\seaso\Sentient Trader\*" -DestinationPath sentient-trader.zip

# Upload to server
scp sentient-trader.zip root@YOUR_DROPLET_IP:/home/root/sentient-trader/

# On server, unzip:
ssh root@YOUR_DROPLET_IP
cd /home/root/sentient-trader
unzip sentient-trader.zip
```

### Option C: Use FileZilla/WinSCP

1. Download FileZilla or WinSCP
2. Connect to your droplet IP
3. Drag and drop your project files

---

## Step 5: Configure Environment

On the server:

```bash
cd /home/$USER/sentient-trader

# Edit environment file
nano .env

# Add your API keys:
# OPENROUTER_API_KEY=sk-or-...
# DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
# etc.

# Save: Ctrl+X, then Y, then Enter
```

---

## Step 6: Install Python Dependencies

```bash
cd /home/$USER/sentient-trader
source venv/bin/activate
pip install -r requirements.txt
```

If you don't have `requirements.txt`, create it:

```bash
pip install loguru requests python-dotenv yfinance pandas numpy aiohttp asyncio
pip freeze > requirements.txt
```

---

## Step 7: Setup Services

```bash
# Make setup script executable
chmod +x deploy/setup_systemd_services.sh

# Run setup
./deploy/setup_systemd_services.sh
```

---

## Step 8: Verify Services Running

```bash
# Check service status
sudo systemctl status sentient-stock-monitor
sudo systemctl status sentient-crypto-breakout
sudo systemctl status sentient-dex-launch

# View logs
tail -f logs/stock_monitor_service.log

# Should see "SERVICE READY" and scan activity
```

---

## Management Commands

### View Logs (Live)

```bash
tail -f logs/stock_monitor_service.log
tail -f logs/crypto_breakout_service.log
tail -f logs/dex_launch_service.log
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
sudo systemctl stop sentient-crypto-breakout
sudo systemctl stop sentient-dex-launch
```

### Check Service Status

```bash
sudo systemctl status sentient-stock-monitor --no-pager
```

### View Last 50 Log Lines

```bash
tail -n 50 logs/stock_monitor_service.log
```

### Search Logs

```bash
grep "HIGH SCORE" logs/stock_monitor_service.log
grep "ERROR" logs/stock_monitor_service.log
```

---

## Firewall Setup (Optional)

If you want to access logs via web interface:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 8080/tcp
sudo ufw enable
```

---

## Auto-Update Script (Optional)

Create a script to pull latest code and restart:

```bash
cat > /home/$USER/sentient-trader/update.sh << 'EOF'
#!/bin/bash
cd /home/$USER/sentient-trader
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart sentient-stock-monitor
sudo systemctl restart sentient-crypto-breakout
sudo systemctl restart sentient-dex-launch
echo "✅ Updated and restarted all services"
EOF

chmod +x update.sh
```

Then to update:

```bash
./update.sh
```

---

## Monitoring with Cron (Optional)

Add a health check that alerts you if services die:

```bash
crontab -e

# Add this line:
*/15 * * * * systemctl is-active sentient-stock-monitor || systemctl start sentient-stock-monitor
```

---

## Cost

- **Droplet**: $6/month
- **Bandwidth**: 1TB included (more than enough)
- **Total**: ~$6/month

---

## Troubleshooting

### Service won't start

```bash
# Check logs
sudo journalctl -u sentient-stock-monitor -n 100 --no-pager

# Check Python errors
cat logs/stock_monitor_error.log
```

### Import errors

```bash
# Reinstall dependencies
cd /home/$USER/sentient-trader
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Out of memory

Upgrade to 2GB droplet:
1. Power off droplet
2. Resize in DigitalOcean dashboard
3. Power on

---

## Benefits of This Setup

✅ **Runs 24/7** - No need to keep your PC on
✅ **No Windows issues** - Linux handles background processes perfectly
✅ **Auto-restart** - Services restart automatically if they crash
✅ **Easy monitoring** - Check logs from anywhere via SSH
✅ **Professional** - Same setup used by real production services
✅ **Cheap** - $6/month vs running PC 24/7 (more expensive electricity)

---

## Next Steps After Setup

1. Monitor logs for first 24 hours to ensure stability
2. Set up GitHub repo for easier code updates
3. Optional: Add monitoring/alerting (e.g., UptimeRobot, Healthchecks.io)
4. Optional: Set up Nginx to view logs via web browser

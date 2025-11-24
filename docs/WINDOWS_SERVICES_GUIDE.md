# Windows Services Guide - Sentient Trader

## Overview

**Phase 3 - Windows Services** enables your monitoring services to run as native Windows services with:

- ✅ **Auto-start on system boot** - Services start automatically when Windows starts
- ✅ **Windows Event Log integration** - All logs visible in Event Viewer
- ✅ **Service management via Services panel** - Control via Windows Services UI
- ✅ **Automatic restart on failure** - Built-in resilience
- ✅ **Background operation** - No terminal window required
- ✅ **User logout persistence** - Services continue running after logout

## Available Services

### 1. Stock Informational Monitor
**Service Name:** `SentientStockMonitor`

Monitors stocks for trading opportunities without executing trades. Sends Discord alerts for high-probability setups.

**Features:**
- LOW priority LLM requests (cost-efficient)
- 15-minute cache TTL
- Multi-factor validation (Technical + ML + LLM)
- Customizable watchlist
- Priority-based Discord alerts

**Configuration:** `config_stock_informational.py`

---

### 2. DEX Launch Monitor
**Service Name:** `SentientDEXLaunch`

Monitors crypto DEX launches and announcements in real-time.

**Features:**
- Tracks Pump.fun (Solana)
- Monitors DexScreener boosted tokens
- Twitter mentions (if configured)
- AI token analysis
- High-score opportunity alerts

**Scan Interval:** 5 minutes

---

### 3. Crypto Breakout Monitor
**Service Name:** `SentientCryptoBreakout`

Monitors cryptocurrency markets on Kraken for breakout patterns.

**Features:**
- High-volume breakout detection
- EMA crossovers and trend changes
- RSI/MACD/Bollinger Band signals
- AI-confirmed opportunities
- Discord instant alerts

**Scan Interval:** 5 minutes

---

## Installation

### Prerequisites

1. **Administrator Access** - Required for installing Windows services
2. **Python 3.8+** - Installed and in PATH
3. **pywin32 Package** - Installed automatically with requirements

### Step 1: Install pywin32

```bash
pip install pywin32
```

After installation, run the post-install script:

```bash
python -m win32serviceutil
```

### Step 2: Install Services

**Option A: Install All Services (Recommended)**

```bash
# Run PowerShell as Administrator
cd "c:\Users\seaso\Sentient Trader"
python windows_services\manage_services.py install-all
```

**Option B: Install Specific Service**

```bash
# Stock Monitor
python windows_services\manage_services.py install stock

# DEX Launch Monitor
python windows_services\manage_services.py install dex

# Crypto Breakout Monitor
python windows_services\manage_services.py install crypto
```

**Option C: Install Directly**

```bash
# Stock Monitor
python windows_services\stock_monitor_service.py install

# DEX Launch Monitor
python windows_services\dex_launch_service.py install

# Crypto Breakout Monitor
python windows_services\crypto_breakout_service.py install
```

---

## Service Management

### Using the Management Script (Recommended)

```bash
# Check status of all services
python windows_services\manage_services.py status

# Start all services
python windows_services\manage_services.py start-all

# Stop all services
python windows_services\manage_services.py stop-all

# Start specific service
python windows_services\manage_services.py start stock
python windows_services\manage_services.py start dex
python windows_services\manage_services.py start crypto

# Stop specific service
python windows_services\manage_services.py stop stock

# Restart specific service
python windows_services\manage_services.py restart stock
```

### Using Windows Services Panel

1. Press `Win + R`, type `services.msc`, press Enter
2. Find services starting with "Sentient"
3. Right-click → Properties
4. Set Startup Type to "Automatic" for auto-start
5. Start/Stop/Restart as needed

### Using Command Line

```bash
# Start service
sc start SentientStockMonitor

# Stop service
sc stop SentientStockMonitor

# Query status
sc query SentientStockMonitor
```

---

## Configuration

### Auto-Start on Boot

Set services to start automatically:

```bash
# Via management script (sets to automatic)
python windows_services\manage_services.py install stock

# Via sc command
sc config SentientStockMonitor start=auto

# Via Services panel
services.msc → Right-click service → Properties → Startup Type: Automatic
```

### Delayed Start

For services that depend on network connectivity:

```bash
sc config SentientStockMonitor start=delayed-auto
```

### Recovery Options

Configure automatic restart on failure:

1. Open `services.msc`
2. Right-click service → Properties → Recovery tab
3. Set:
   - First failure: Restart the Service
   - Second failure: Restart the Service
   - Subsequent failures: Restart the Service
   - Restart service after: 1 minute

---

## Monitoring & Logs

### View Logs

**Service-Specific Logs:**
```bash
# Stock Monitor
logs\SentientStockMonitor.log

# DEX Launch Monitor
logs\SentientDEXLaunch.log

# Crypto Breakout Monitor
logs\SentientCryptoBreakout.log
```

**Windows Event Log:**
1. Press `Win + R`, type `eventvwr`, press Enter
2. Navigate to: Windows Logs → Application
3. Filter by Source: "Sentient*"

### Real-Time Monitoring

```bash
# Watch log file (using PowerShell)
Get-Content logs\SentientStockMonitor.log -Wait -Tail 50
```

---

## Troubleshooting

### Service Won't Install

**Error: Access Denied**
- **Solution:** Run PowerShell/Command Prompt as Administrator
- Right-click → "Run as Administrator"

**Error: pywin32 not found**
```bash
pip install pywin32
python -m win32serviceutil
```

**Error: Python not found**
- Ensure Python is in PATH
- Use full path: `C:\Python39\python.exe windows_services\stock_monitor_service.py install`

---

### Service Won't Start

**Check Event Viewer for errors:**
1. `eventvwr` → Windows Logs → Application
2. Look for errors from "Sentient*" sources

**Common Issues:**

1. **Import errors**
   - Verify all packages installed: `pip install -r requirements.txt`
   - Check virtual environment if using one

2. **Configuration errors**
   - Verify `.env` file exists with required keys
   - Check config files (e.g., `config_stock_informational.py`)

3. **Permission errors**
   - Ensure service has access to project directory
   - Check log file permissions in `logs/` folder

**Manually start to see errors:**
```bash
python windows_services\stock_monitor_service.py debug
```

---

### Service Crashes/Stops

**Check logs:**
```bash
type logs\SentientStockMonitor.log
```

**Check Windows Event Log:**
- Look for crash reports in Application log

**Set up automatic restart:**
- Services panel → Recovery tab → Restart the Service

---

### Service Running but No Alerts

1. **Check Discord webhook:**
   - Verify `DISCORD_WEBHOOK_URL` in `.env`
   - Test webhook: `python -c "import requests; requests.post('YOUR_WEBHOOK_URL', json={'content': 'Test'})"`

2. **Check monitoring is active:**
   - Watch log file for scan activity
   - Verify watchlist is populated

3. **Check LLM quota:**
   - View LLM Usage tab in Streamlit app
   - Check cost limits not exceeded

---

## Updating Services

After code changes, update running services:

```bash
# Stop service
python windows_services\manage_services.py stop stock

# Update service (re-install)
python windows_services\manage_services.py install stock

# Start service
python windows_services\manage_services.py start stock
```

Or use restart:
```bash
python windows_services\manage_services.py restart stock
```

---

## Uninstallation

### Remove All Services

```bash
# Stop and remove all services
python windows_services\manage_services.py uninstall-all
```

### Remove Specific Service

```bash
# Via management script
python windows_services\manage_services.py uninstall stock

# Via service script
python windows_services\stock_monitor_service.py remove
```

---

## Advanced Configuration

### Custom Service Configuration

Edit service files to customize:

**Stock Monitor:** `windows_services\stock_monitor_service.py`
```python
# Modify run_service() method
monitor = get_stock_informational_monitor(
    watchlist=['AAPL', 'MSFT', 'TSLA'],  # Custom watchlist
    scan_interval_minutes=15  # Custom interval
)
```

**DEX Launch:** `windows_services\dex_launch_service.py`
```python
# Modify scan interval
monitor = get_announcement_monitor(scan_interval=600)  # 10 min
```

**Crypto Breakout:** `windows_services\crypto_breakout_service.py`
```python
# Modify thresholds
monitor = CryptoBreakoutMonitor(
    scan_interval_seconds=300,
    min_score=80.0,  # Higher threshold
    min_confidence='HIGH'
)
```

After changes, reinstall service.

---

### Running Multiple Instances

To run multiple instances of a service with different configs:

1. Copy service file (e.g., `stock_monitor_service.py` → `stock_monitor_service2.py`)
2. Change `_svc_name_` to unique name (e.g., `"SentientStockMonitor2"`)
3. Modify configuration in `run_service()`
4. Install as new service

---

### Service Dependencies

Set service to start after another service:

```bash
sc config SentientStockMonitor depend=SentientDEXLaunch
```

---

## Best Practices

### ✅ DO

- Run as Administrator for installation/management
- Check Event Viewer for startup errors
- Set Recovery options for automatic restart
- Monitor logs regularly
- Use management script for bulk operations
- Test services with `status` command before relying on them
- Configure alerts to avoid spam (use cooldowns)

### ❌ DON'T

- Install services without Administrator privileges
- Ignore Event Log errors
- Run services with missing dependencies
- Delete log files while services are running
- Modify service files without reinstalling
- Run multiple services with the same name

---

## Security Considerations

### Service Account

By default, services run under Local System account. For production:

1. Create dedicated service account
2. Grant minimal required permissions
3. Configure service to use account:
   ```bash
   sc config SentientStockMonitor obj=.\ServiceAccount password=PASSWORD
   ```

### API Keys

- Store in `.env` file with restricted permissions
- Never commit `.env` to version control
- Rotate keys periodically
- Monitor usage in LLM dashboard

### Network Access

- Configure Windows Firewall for required access
- Use HTTPS for all external API calls
- Validate webhook URLs before deployment

---

## Performance Tuning

### Memory Usage

Monitor service memory:
```powershell
Get-Process | Where-Object {$_.ProcessName -like "*python*"}
```

Optimize if needed:
- Increase cache TTL to reduce LLM calls
- Reduce scan frequency
- Limit watchlist size

### CPU Usage

If high CPU usage:
- Increase scan intervals
- Use vectorized operations
- Enable caching more aggressively

### Disk I/O

Manage log file size:
- Current rotation: 50 MB per file
- Retention: 30 days
- Adjust in service `_setup_logging()` method

---

## Comparison: Services vs Manual Scripts

| Feature | Windows Service | Manual Script |
|---------|----------------|---------------|
| Auto-start on boot | ✅ Yes | ❌ No |
| Runs after logout | ✅ Yes | ❌ No |
| Windows Event Log | ✅ Yes | ❌ No |
| Service panel UI | ✅ Yes | ❌ No |
| Auto-restart on crash | ✅ Yes | ❌ No |
| Easy to stop/start | ✅ Yes | ⚠️ Manual |
| Terminal required | ❌ No | ✅ Yes |
| Setup complexity | ⚠️ Medium | ✅ Easy |

**Recommendation:** Use Windows services for production, manual scripts for development/testing.

---

## Migration from Manual Scripts

If currently running:
```bash
python run_launch_monitor_background.py
```

Migrate to service:

1. **Stop manual script** (Ctrl+C)
2. **Install service:**
   ```bash
   python windows_services\manage_services.py install dex
   ```
3. **Configure auto-start:**
   ```bash
   sc config SentientDEXLaunch start=auto
   ```
4. **Start service:**
   ```bash
   python windows_services\manage_services.py start dex
   ```
5. **Verify in Event Viewer** - Check for startup messages

---

## Integration with Streamlit App

Services run independently of the Streamlit app. Benefits:

- **Services:** Continuous monitoring 24/7
- **Streamlit:** Interactive analysis and manual trading
- **Both:** Share same database, configs, and logs

**Workflow:**
1. Services monitor and send Discord alerts
2. Review alerts in Discord
3. Open Streamlit app for detailed analysis
4. Execute trades manually or via auto-trader
5. Services continue monitoring positions

---

## Cost Optimization

Services use LLM API calls. Optimize costs:

### Stock Monitor
- Uses LOW priority with 15-min cache
- Estimated: $0.10-0.50/day for 50 symbols
- Reduce via: Longer intervals, smaller watchlist

### DEX Launch
- Uses HIGH priority for token analysis
- Estimated: $0.50-2.00/day depending on launches
- Optimize: Filter by score before LLM call

### Crypto Breakout
- Uses MEDIUM priority
- Estimated: $0.20-1.00/day
- Optimize: AI analysis only for high scores

**Monitor in:** Streamlit app → 🤖 LLM Usage tab

---

## Support & Troubleshooting

### Quick Reference

```bash
# Installation
python windows_services\manage_services.py install-all

# Status check
python windows_services\manage_services.py status

# Start services
python windows_services\manage_services.py start-all

# View logs
type logs\SentientStockMonitor.log

# Event Viewer
eventvwr → Application → Filter by "Sentient"

# Restart service
python windows_services\manage_services.py restart stock

# Uninstall
python windows_services\manage_services.py uninstall-all
```

### Common Commands

**PowerShell as Administrator:**
```powershell
# Check if service exists
Get-Service | Where-Object {$_.Name -like "Sentient*"}

# Start all Sentient services
Get-Service | Where-Object {$_.Name -like "Sentient*"} | Start-Service

# Stop all Sentient services
Get-Service | Where-Object {$_.Name -like "Sentient*"} | Stop-Service

# Get service status
Get-Service SentientStockMonitor
```

---

## Summary

**Phase 3 Complete! ✅**

You now have:
- ✅ 3 Windows services for monitoring
- ✅ Centralized management script
- ✅ Auto-start on boot capability
- ✅ Windows Event Log integration
- ✅ Service panel management
- ✅ Automatic restart on failure

**Next Steps:**
1. Install services: `python windows_services\manage_services.py install-all`
2. Configure auto-start: Set in Services panel
3. Start services: `python windows_services\manage_services.py start-all`
4. Monitor logs: Check `logs/` folder and Event Viewer
5. Verify alerts: Check Discord for notifications

**For Production:**
- Set all services to Automatic (Delayed Start)
- Configure Recovery options (restart on failure)
- Monitor costs in LLM Usage dashboard
- Review logs weekly
- Keep services updated with code changes

---

**Related Documentation:**
- `docs/LLM_REQUEST_MANAGER_GUIDE.md` - LLM cost optimization
- `config_stock_informational.py` - Stock monitor config
- `windows_services/manage_services.py` - Service management

**Service Files:**
- `windows_services/stock_monitor_service.py`
- `windows_services/dex_launch_service.py`
- `windows_services/crypto_breakout_service.py`
- `services/windows_service_base.py`

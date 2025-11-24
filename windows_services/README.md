# Windows Services - Sentient Trader

This directory contains Windows service wrappers for running monitoring services as native Windows services.

## Quick Start

### Installation (Run as Administrator)

```bash
# Install pywin32
pip install pywin32
python -m win32serviceutil

# Install all services
python windows_services\manage_services.py install-all

# Start all services
python windows_services\manage_services.py start-all

# Check status
python windows_services\manage_services.py status
```

## Services

| Service | Name | Description |
|---------|------|-------------|
| **Stock Monitor** | `SentientStockMonitor` | Monitors stocks for opportunities |
| **DEX Launch** | `SentientDEXLaunch` | Monitors crypto DEX launches |
| **Crypto Breakout** | `SentientCryptoBreakout` | Monitors crypto breakout patterns |

## Files

- **`windows_service_base.py`** - Base class for all services
- **`stock_monitor_service.py`** - Stock Informational Monitor service
- **`dex_launch_service.py`** - DEX Launch Monitor service
- **`crypto_breakout_service.py`** - Crypto Breakout Monitor service
- **`manage_services.py`** - Service management script

## Management Commands

```bash
# Status
python windows_services\manage_services.py status

# Install specific service
python windows_services\manage_services.py install stock
python windows_services\manage_services.py install dex
python windows_services\manage_services.py install crypto

# Start/Stop
python windows_services\manage_services.py start stock
python windows_services\manage_services.py stop stock
python windows_services\manage_services.py restart stock

# Bulk operations
python windows_services\manage_services.py install-all
python windows_services\manage_services.py start-all
python windows_services\manage_services.py stop-all
python windows_services\manage_services.py uninstall-all
```

## Direct Service Control

Each service can be controlled directly:

```bash
# Stock Monitor
python windows_services\stock_monitor_service.py install
python windows_services\stock_monitor_service.py start
python windows_services\stock_monitor_service.py stop
python windows_services\stock_monitor_service.py remove

# DEX Launch
python windows_services\dex_launch_service.py install
python windows_services\dex_launch_service.py start
python windows_services\dex_launch_service.py stop
python windows_services\dex_launch_service.py remove

# Crypto Breakout
python windows_services\crypto_breakout_service.py install
python windows_services\crypto_breakout_service.py start
python windows_services\crypto_breakout_service.py stop
python windows_services\crypto_breakout_service.py remove
```

## Windows Services Panel

Manage via Windows Services panel:
1. Press `Win + R`
2. Type `services.msc`
3. Find services starting with "Sentient"
4. Right-click for Start/Stop/Properties

## Auto-Start Configuration

Set services to start automatically on boot:

1. Open `services.msc`
2. Right-click service → Properties
3. Startup Type: **Automatic** (or **Automatic (Delayed Start)**)
4. Click OK
5. Start the service

Or via command line:
```bash
sc config SentientStockMonitor start=auto
sc config SentientDEXLaunch start=auto
sc config SentientCryptoBreakout start=auto
```

## Logs

Service logs are located in:
```
logs\SentientStockMonitor.log
logs\SentientDEXLaunch.log
logs\SentientCryptoBreakout.log
```

Windows Event Log:
- Open Event Viewer (`eventvwr`)
- Navigate to: Windows Logs → Application
- Filter by Source: "Sentient*"

## Troubleshooting

### Service won't install
- Run PowerShell/CMD as Administrator
- Verify pywin32 installed: `pip show pywin32`
- Run: `python -m win32serviceutil`

### Service won't start
- Check Event Viewer for errors
- Verify `.env` file exists with API keys
- Check log files for errors
- Test manually: `python windows_services\stock_monitor_service.py debug`

### Service crashes
- Check Windows Event Log
- Review service log files
- Configure automatic restart in Services panel → Recovery tab

## Documentation

For comprehensive guide, see:
**`docs\WINDOWS_SERVICES_GUIDE.md`**

Covers:
- Installation & setup
- Configuration
- Monitoring & logs
- Troubleshooting
- Best practices
- Security
- Performance tuning

## Support

- **Installation Guide:** `docs\WINDOWS_SERVICES_GUIDE.md`
- **LLM Cost Management:** `docs\LLM_REQUEST_MANAGER_GUIDE.md`
- **Service Logs:** `logs\` folder
- **Windows Event Log:** `eventvwr` → Application

## Requirements

- **Windows OS** - Services are Windows-specific
- **Python 3.8+** - Installed and in PATH
- **pywin32** - Automatically installed via `requirements.txt`
- **Administrator Access** - Required for service installation

## Phase 3 Status

✅ **COMPLETE**

All Phase 3 components implemented:
- Windows service base class
- 3 monitoring service wrappers
- Centralized management script
- Comprehensive documentation
- Auto-start capability
- Windows Event Log integration

# Troubleshooting Commands - Service Import Issues

## Current Situation

Services are starting but hanging during module imports. The `llm_helper` fix worked, but now they're hanging on imports of:
- `yfinance` (in top_trades_scanner)
- `requests` (in discord_webhook)

These network libraries can't initialize properly in Task Scheduler's environment.

---

## Step 1: Test Import Timing

Run this to see HOW LONG imports actually take (not if they timeout):

```powershell
cd "C:\Users\seaso\Sentient Trader"
.\venv\Scripts\Activate.ps1
python windows_services\test_import_timing.py
```

**What to look for:**
- If imports complete in < 30 seconds → Use **simple runners** (no timeout)
- If imports hang forever → Need to make more imports lazy
- If imports fail with errors → Fix the underlying issue

---

## Step 2: Test Simple Runner Manually

Try running the stock monitor directly to see if it works WITHOUT Task Scheduler:

```powershell
python windows_services\runners\run_stock_monitor_simple.py
```

**Expected behavior:**
- Takes 30-60 seconds to import (that's OK!)
- Eventually shows "SERVICE READY"
- Starts running scans

**If it works manually:** Services will work in Task Scheduler with simple runners.

**If it hangs forever:** Need to make more imports lazy (see Step 4).

---

## Step 3: Switch to Simple Runners in Task Scheduler

If Step 2 works (even if slow), update Task Scheduler to use simple runners:

```powershell
# Stop current tasks
Stop-ScheduledTask -TaskName "SentientStockMonitor"
Stop-ScheduledTask -TaskName "SentientCryptoBreakout"  
Stop-ScheduledTask -TaskName "SentientDEXLaunch"

# Remove old tasks
Unregister-ScheduledTask -TaskName "SentientStockMonitor" -Confirm:$false
Unregister-ScheduledTask -TaskName "SentientCryptoBreakout" -Confirm:$false
Unregister-ScheduledTask -TaskName "SentientDEXLaunch" -Confirm:$false

# Create new tasks with simple runners
$venvPython = "C:\Users\seaso\Sentient Trader\venv\Scripts\pythonw.exe"
$projectRoot = "C:\Users\seaso\Sentient Trader"
$username = "SEANBC\seaso"

# Stock Monitor
$action = New-ScheduledTaskAction `
    -Execute $venvPython `
    -Argument "windows_services\runners\run_stock_monitor_simple.py" `
    -WorkingDirectory $projectRoot

$trigger = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)

$principal = New-ScheduledTaskPrincipal `
    -UserID $username `
    -LogonType S4U `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName "SentientStockMonitor" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force

# Crypto Breakout
$action = New-ScheduledTaskAction `
    -Execute $venvPython `
    -Argument "windows_services\runners\run_crypto_breakout_simple.py" `
    -WorkingDirectory $projectRoot

Register-ScheduledTask `
    -TaskName "SentientCryptoBreakout" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force

# DEX Launch
$action = New-ScheduledTaskAction `
    -Execute $venvPython `
    -Argument "windows_services\runners\run_dex_launch_simple.py" `
    -WorkingDirectory $projectRoot

Register-ScheduledTask `
    -TaskName "SentientDEXLaunch" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force

# Start all
Start-ScheduledTask -TaskName "SentientStockMonitor"
Start-ScheduledTask -TaskName "SentientCryptoBreakout"
Start-ScheduledTask -TaskName "SentientDEXLaunch"
```

Then wait 2-3 minutes and check logs:

```powershell
Get-Content logs\stock_monitor_service.log -Tail 30
```

Look for "SERVICE READY" message.

---

## Step 4: If Still Hanging - Make More Imports Lazy

If imports never complete, we need to make the network library imports lazy.

### Option A: Quick Fix - Disable Features

Edit `services/alert_system.py` to make discord import lazy:

**BEFORE:**
```python
from src.integrations.discord_webhook import send_discord_alert
```

**AFTER:**
```python
# Lazy import to avoid Task Scheduler hang
def send_discord_alert(*args, **kwargs):
    from src.integrations.discord_webhook import send_discord_alert as _send
    return _send(*args, **kwargs)
```

### Option B: Deeper Fix - Lazy Import in Top Trades Scanner

Edit `services/top_trades_scanner.py`:

**BEFORE:**
```python
import yfinance as yf
```

**AFTER:**
```python
# Lazy import
yf = None

def _get_yf():
    global yf
    if yf is None:
        import yfinance as _yf
        yf = _yf
    return yf
```

Then replace all `yf.` with `_get_yf().` in the file.

---

## Step 5: Check What's Actually Running

View detailed task status:

```powershell
Get-ScheduledTask -TaskName 'Sentient*' | ForEach-Object {
    $info = Get-ScheduledTaskInfo $_.TaskName
    [PSCustomObject]@{
        Name = $_.TaskName
        State = $_.State
        LastRun = $info.LastRunTime
        LastResult = $info.LastTaskResult
        NextRun = $info.NextRunTime
    }
} | Format-Table -AutoSize
```

**LastResult codes:**
- `0` = Success
- `1` = Still running
- `267009` = Currently executing (normal)
- `Other` = Error code

---

## Step 6: Watch Logs Live

Open separate terminals:

```powershell
# Terminal 1
Get-Content logs\stock_monitor_service.log -Tail 20 -Wait

# Terminal 2  
Get-Content logs\crypto_breakout_service.log -Tail 20 -Wait

# Terminal 3
Get-Content logs\dex_launch_service.log -Tail 20 -Wait
```

---

## Quick Decision Tree

```
Are imports slow but eventually complete?
├─ YES → Use simple runners (Step 3)
│
└─ NO, they hang forever
   ├─ Can you wait 2-5 minutes?
   │  └─ YES → Try simple runners anyway (might just be REALLY slow)
   │
   └─ Definitely hanging (10+ minutes)
      └─ Make imports lazy (Step 4)
         ├─ Option A: Disable discord alerts temporarily
         └─ Option B: Fix yfinance imports properly
```

---

## Verification Checklist

After applying fixes:

```powershell
# 1. Test import timing
python windows_services\test_import_timing.py
# Should complete in < 60 seconds

# 2. Test runner manually
python windows_services\runners\run_stock_monitor_simple.py
# Should show "SERVICE READY" within 2 minutes
# Press Ctrl+C to stop

# 3. Deploy to Task Scheduler
# (Use commands from Step 3)

# 4. Wait 3 minutes, then check logs
Get-Content logs\stock_monitor_service.log -Tail 30
# Should see "SERVICE READY" and scan activity

# 5. Check task is actually running
Get-ScheduledTask -TaskName "SentientStockMonitor" | Select TaskName, State
# Should show "Running"
```

---

## What "Working" Looks Like

### Good Log Output:
```
2025-11-24 03:35:00 | INFO | 📊 STOCK MONITOR - SIMPLE RUNNER
2025-11-24 03:35:00 | INFO | Starting imports (be patient)...
2025-11-24 03:35:45 | INFO | ✓ Import completed in 45.2s
2025-11-24 03:35:46 | INFO | ✓ Instance created in 0.8s
2025-11-24 03:35:46 | INFO | ✓ Watchlist: 50 symbols
2025-11-24 03:35:46 | INFO | 🚀 SERVICE READY (total startup: 46.0s)
2025-11-24 03:35:46 | INFO | Starting scan #1...
2025-11-24 03:36:10 | INFO | Scan #1 complete (24.3s)
2025-11-24 03:36:10 | INFO | Next scan in 30 minutes...
```

### Bad Log Output (Still Hanging):
```
2025-11-24 03:35:00 | INFO | 📊 STOCK MONITOR - SIMPLE RUNNER
2025-11-24 03:35:00 | INFO | Starting imports (be patient)...
[... nothing else for 10+ minutes ...]
```

If you see the bad output, proceed to Step 4 (make more imports lazy).

---

## Emergency: Disable All Services

If you need to stop everything:

```powershell
Stop-ScheduledTask -TaskName "SentientStockMonitor"
Stop-ScheduledTask -TaskName "SentientCryptoBreakout"
Stop-ScheduledTask -TaskName "SentientDEXLaunch"
```

Or disable them permanently:

```powershell
Disable-ScheduledTask -TaskName "SentientStockMonitor"
Disable-ScheduledTask -TaskName "SentientCryptoBreakout"
Disable-ScheduledTask -TaskName "SentientDEXLaunch"
```

To re-enable:

```powershell
Enable-ScheduledTask -TaskName "SentientStockMonitor"
```

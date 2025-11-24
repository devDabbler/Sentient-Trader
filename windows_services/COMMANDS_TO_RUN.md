# Commands to Run - Task Scheduler Fix

Run these commands in **PowerShell Admin** from the project root directory.

## Step 1: Activate Virtual Environment
```powershell
cd "C:\Users\seaso\Sentient Trader"
.\venv\Scripts\Activate.ps1
```

## Step 2: Test the Lazy Import Fix
This will verify that all imports now work without hanging:
```powershell
python windows_services\test_lazy_imports.py
```

**Expected**: All tests pass, imports complete in < 1 second each.

## Step 3: Test the Debug Script
This will verify the specific import that was hanging:
```powershell
python windows_services\runners\run_stock_monitor_debug.py
```

**Expected**: Should see `✓ services.llm_helper - SHOULD NOW WORK` and complete all imports.

## Step 4: Remove Old Tasks (if any exist)
```powershell
Get-ScheduledTask -TaskName "Sentient*" | Unregister-ScheduledTask -Confirm:$false
```

## Step 5: Setup New Task Scheduler Services
```powershell
.\windows_services\setup_task_scheduler_v2.ps1
```

This will:
- Create 3 scheduled tasks (Stock Monitor, Crypto Breakout, DEX Launch)
- Start all services
- Show initial status and logs

**Expected**: 
- Tasks created successfully
- Services start and show "Running" state
- Logs show "SERVICE READY" messages

## Step 6: Verify Services Are Running
```powershell
# Check task status
Get-ScheduledTask -TaskName 'Sentient*' | Select TaskName, State

# Check logs for successful startup
Get-Content logs\stock_monitor_service.log -Tail 30
Get-Content logs\crypto_breakout_service.log -Tail 30
Get-Content logs\dex_launch_service.log -Tail 30
```

**Look for**:
- `📊 STOCK MONITOR SERVICE V5 - LAZY IMPORT FIX`
- `✓ Import successful!`
- `🚀 SERVICE READY - STARTING MONITOR`
- Scan cycles beginning

## Step 7: Watch Logs in Real-Time (Optional)
Open separate PowerShell windows for each:
```powershell
# Stock Monitor
Get-Content logs\stock_monitor_service.log -Tail 20 -Wait

# Crypto Breakout
Get-Content logs\crypto_breakout_service.log -Tail 20 -Wait

# DEX Launch
Get-Content logs\dex_launch_service.log -Tail 20 -Wait
```

## Useful Management Commands

### View Task Status with Last Run Info
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

### Start a Specific Service
```powershell
Start-ScheduledTask -TaskName "SentientStockMonitor"
# or
Start-ScheduledTask -TaskName "SentientCryptoBreakout"
# or
Start-ScheduledTask -TaskName "SentientDEXLaunch"
```

### Stop a Specific Service
```powershell
Stop-ScheduledTask -TaskName "SentientStockMonitor"
```

### Stop All Services
```powershell
Get-ScheduledTask -TaskName "Sentient*" | Stop-ScheduledTask
```

### Start All Services
```powershell
Get-ScheduledTask -TaskName "Sentient*" | Start-ScheduledTask
```

### View Detailed Task Info
```powershell
Get-ScheduledTaskInfo -TaskName "SentientStockMonitor" | Format-List
```

### Check Last Run Result Codes
```powershell
$tasks = Get-ScheduledTask -TaskName "Sentient*"
foreach ($task in $tasks) {
    $info = Get-ScheduledTaskInfo $task.TaskName
    Write-Host "$($task.TaskName): " -NoNewline
    if ($info.LastTaskResult -eq 0) {
        Write-Host "SUCCESS (0)" -ForegroundColor Green
    } elseif ($info.LastTaskResult -eq 1) {
        Write-Host "RUNNING (1)" -ForegroundColor Yellow
    } else {
        Write-Host "ERROR ($($info.LastTaskResult))" -ForegroundColor Red
    }
}
```

## Troubleshooting Commands

### If Service Won't Start
```powershell
# Try running manually to see errors
python windows_services\runners\run_stock_monitor_v5.py
```

### If Imports Still Hang
```powershell
# Run debug script to identify where
python windows_services\runners\run_stock_monitor_debug.py
```

### Check Python Path in Task
```powershell
(Get-ScheduledTask -TaskName "SentientStockMonitor").Actions[0].Execute
# Should show: C:\Users\seaso\Sentient Trader\venv\Scripts\pythonw.exe
```

### View Task XML Configuration
```powershell
Export-ScheduledTask -TaskName "SentientStockMonitor"
```

### Manually Create Test Task
```powershell
$action = New-ScheduledTaskAction `
    -Execute "C:\Users\seaso\Sentient Trader\venv\Scripts\pythonw.exe" `
    -Argument "windows_services\runners\run_stock_monitor_v5.py" `
    -WorkingDirectory "C:\Users\seaso\Sentient Trader"
    
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(10)

$principal = New-ScheduledTaskPrincipal `
    -UserID "SEANBC\seaso" `
    -LogonType S4U `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName "TestStockMonitor" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Force

Start-ScheduledTask -TaskName "TestStockMonitor"
Start-Sleep 20
Get-Content logs\stock_monitor_service.log -Tail 30
```

## Quick Health Check Script
Save and run this to check all services at once:
```powershell
Write-Host "`n=== SENTIENT TRADER SERVICE STATUS ===" -ForegroundColor Cyan
Write-Host ""

# Task status
Write-Host "Task Status:" -ForegroundColor Yellow
Get-ScheduledTask -TaskName 'Sentient*' | Select TaskName, State | Format-Table

# Last run times
Write-Host "`nLast Run Times:" -ForegroundColor Yellow
Get-ScheduledTask -TaskName 'Sentient*' | ForEach-Object {
    $info = Get-ScheduledTaskInfo $_.TaskName
    "$($_.TaskName): $($info.LastRunTime)"
}

# Check for errors in logs
Write-Host "`nRecent Errors:" -ForegroundColor Yellow
$logs = @(
    "logs\stock_monitor_service.log",
    "logs\crypto_breakout_service.log", 
    "logs\dex_launch_service.log"
)

foreach ($log in $logs) {
    if (Test-Path $log) {
        $errors = Get-Content $log | Select-String "ERROR" | Select-Object -Last 3
        if ($errors) {
            Write-Host "`n$log" -ForegroundColor Red
            $errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        }
    }
}

Write-Host "`n" -NoNewline
```

## System Reboot Test
After everything is working:
```powershell
# Verify tasks will start at boot
Get-ScheduledTask -TaskName 'Sentient*' | Select-Object TaskName, @{
    Name='StartsAtBoot'
    Expression={$_.Triggers[0].CimClass.CimClassName -eq 'MSFT_TaskBootTrigger' -or 
                $_.Triggers[0].CimClass.CimClassName -eq 'MSFT_TaskSessionStateChangeTrigger'}
}

# Then reboot and check logs after restart
Restart-Computer -Confirm
```

After reboot:
```powershell
cd "C:\Users\seaso\Sentient Trader"
Get-ScheduledTask -TaskName 'Sentient*' | Select TaskName, State
Get-Content logs\stock_monitor_service.log -Tail 20
```

## Expected Timeline
- **Step 1-2**: < 1 minute (test imports)
- **Step 3**: < 1 minute (verify fix)
- **Step 4-5**: 2-3 minutes (setup services)
- **Step 6**: 1 minute (verify)
- **Total**: ~5-10 minutes to full deployment

## Success Indicators
✅ All imports complete without timeout  
✅ Tasks show "Running" state  
✅ Logs show "SERVICE READY" messages  
✅ Scan cycles appear in logs  
✅ No error messages in logs  
✅ Services survive reboot (test after)  

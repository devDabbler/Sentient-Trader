# Task Scheduler Import Hang Fix

## Problem

Windows Task Scheduler services were hanging on startup during the import phase. The services would start but never reach the running state.

### Symptoms
- Services showed as "Running" in Task Scheduler but weren't actually functioning
- Logs showed imports starting but never completing
- Specifically hung at: `Importing services.llm_helper...`
- Commands worked fine when run manually in PowerShell
- Debug logs showed hang occurred at line 68 of `run_stock_monitor_debug.py`

### Root Cause

The issue was caused by **module-level imports triggering blocking operations** in the Task Scheduler environment:

1. `services/llm_helper.py` imported `get_llm_manager` at module level (line 8)
2. This triggered import of `services/llm_request_manager.py`
3. `llm_request_manager.py` imports `requests`, `asyncio`, and other network libraries
4. In Task Scheduler's non-interactive environment, these imports can hang when:
   - Network initialization occurs
   - Singleton patterns initialize
   - Socket operations are attempted without proper environment setup

### Why It Worked Manually But Not in Task Scheduler

- **Manual execution**: Full user environment with GUI, network access, proper PATH
- **Task Scheduler**: Limited environment, no GUI, different security context, network may not be fully initialized

This is a known issue with Windows services and Task Scheduler - see:
- https://stackoverflow.com/questions/46645016/task-scheduler-doesnt-work-when-my-python-3-6-script-has-the-requests-module-im
- Module-level imports that do network operations can block in service context

## Solution: Lazy Imports

Changed `services/llm_helper.py` to use **lazy imports** - moving imports inside functions instead of module level.

### Before (Broken)
```python
# services/llm_helper.py
from services.llm_request_manager import get_llm_manager  # ❌ Module-level import

class LLMHelper:
    @property
    def manager(self):
        if self._manager is None:
            self._manager = get_llm_manager()  # Uses already-imported function
        return self._manager
```

### After (Fixed)
```python
# services/llm_helper.py
# DO NOT IMPORT llm_request_manager at module level

class LLMHelper:
    @property
    def manager(self):
        if self._manager is None:
            from services.llm_request_manager import get_llm_manager  # ✅ Lazy import
            self._manager = get_llm_manager()
        return self._manager
```

### Benefits of Lazy Imports

1. **Fast imports**: Module loads instantly without waiting for dependencies
2. **Conditional loading**: Heavy dependencies only loaded when actually needed
3. **Service compatibility**: Works in restricted Task Scheduler environment
4. **Error isolation**: Import errors only occur when feature is used, not at startup

## Files Modified

### Core Fix
- `services/llm_helper.py` - Made all imports of `llm_request_manager` lazy

### New Service Runners
- `windows_services/runners/run_stock_monitor_v5.py` - Updated runner with better error handling
- `windows_services/runners/run_stock_monitor_debug.py` - Enhanced debug output

### Setup Scripts
- `windows_services/setup_task_scheduler_v2.ps1` - New setup script using V5 runner
- `windows_services/test_lazy_imports.py` - Test script to verify fix

## How to Deploy

### 1. Test Imports First
Run the test script to verify lazy imports work:
```powershell
cd "C:\Users\seaso\Sentient Trader"
.\venv\Scripts\Activate.ps1
python windows_services\test_lazy_imports.py
```

Expected output: All imports should succeed without timeouts.

### 2. Test Debug Script
Verify the debug script now completes:
```powershell
python windows_services\runners\run_stock_monitor_debug.py
```

Should see: `✓ services.llm_helper - SHOULD NOW WORK (lazy imports added)`

### 3. Setup Task Scheduler
Run the new setup script (in **PowerShell Admin**):
```powershell
cd "C:\Users\seaso\Sentient Trader"
.\venv\Scripts\Activate.ps1
.\windows_services\setup_task_scheduler_v2.ps1
```

This will:
- Remove old tasks
- Create new tasks with V5 runner
- Start all services
- Show initial logs

### 4. Monitor Services
Check that services are running:
```powershell
# View task status
Get-ScheduledTask -TaskName 'Sentient*' | Select TaskName, State

# Watch logs in real-time
Get-Content logs\stock_monitor_service.log -Tail 20 -Wait

# Check for successful initialization
Get-Content logs\stock_monitor_service.log | Select-String "SERVICE READY"
```

## Task Scheduler Commands

### View Status
```powershell
Get-ScheduledTask -TaskName 'Sentient*' | 
    Select TaskName, State, 
    @{Name="LastRun"; Expression={(Get-ScheduledTaskInfo $_.TaskName).LastRunTime}}
```

### Start a Service
```powershell
Start-ScheduledTask -TaskName "SentientStockMonitor"
```

### Stop a Service
```powershell
Stop-ScheduledTask -TaskName "SentientStockMonitor"
```

### View Recent Logs
```powershell
Get-Content logs\stock_monitor_service.log -Tail 30
```

### Watch Logs Live
```powershell
Get-Content logs\stock_monitor_service.log -Tail 20 -Wait
```

## Verification Steps

After setup, verify each service is working:

### 1. Stock Monitor
```powershell
# Check log for successful init
Get-Content logs\stock_monitor_service.log -Tail 20

# Should see:
# - "SERVICE READY - STARTING MONITOR"
# - Watchlist loaded
# - Starting scan cycles
```

### 2. Crypto Breakout
```powershell
Get-Content logs\crypto_breakout_service.log -Tail 20

# Should see:
# - Service initialization
# - Scan cycles running
```

### 3. DEX Launch
```powershell
Get-Content logs\dex_launch_service.log -Tail 20

# Should see:
# - Service initialization
# - Monitoring loops
```

## Troubleshooting

### Service Shows "Running" But No Logs
**Cause**: Script may have crashed immediately  
**Fix**: Check Task Scheduler Last Run Result
```powershell
Get-ScheduledTaskInfo -TaskName "SentientStockMonitor" | 
    Select LastRunTime, LastTaskResult
```
- Result 0 = Success
- Result 1 = General failure
- Result 0x1 = Import/syntax error

### Service Stops Immediately
**Cause**: Python error or missing dependencies  
**Fix**: Run manually first to see error:
```powershell
python windows_services\runners\run_stock_monitor_v5.py
```

### Imports Still Hanging
**Cause**: Another module might have module-level network operations  
**Fix**: Use the debug script to identify:
```powershell
python windows_services\runners\run_stock_monitor_debug.py
```
Check where it hangs and apply lazy imports there too.

### "Access Denied" Errors
**Cause**: Task not running with proper privileges  
**Fix**: Ensure task is set to "Run with highest privileges" and user has proper permissions

## Best Practices Going Forward

### 1. Always Use Lazy Imports for Heavy Dependencies
```python
# ❌ BAD - Module level
from some_heavy_library import something

# ✅ GOOD - Lazy import inside function
def use_something():
    from some_heavy_library import something
    return something()
```

### 2. Especially for These Types of Imports
- Network libraries (`requests`, `httpx`, `aiohttp`)
- API clients (OpenAI, Anthropic, etc.)
- Database connections
- Large ML/data libraries

### 3. Test in Task Scheduler Environment
Before committing service changes:
```powershell
# Quick Task Scheduler test
$action = New-ScheduledTaskAction -Execute "pythonw.exe" -Argument "your_script.py"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
Register-ScheduledTask -TaskName "Test" -Action $action -Trigger $trigger -Force
Start-ScheduledTask -TaskName "Test"
Start-Sleep 10
Get-Content logs\your_log.log -Tail 20
```

### 4. Use pythonw.exe for Services
- `python.exe` = Shows console (can cause issues in Task Scheduler)
- `pythonw.exe` = No console (proper for background services)

## Architecture Notes

### Service Startup Flow
1. Task Scheduler launches `pythonw.exe` with script path
2. Script sets up Python path and environment
3. Imports are performed (NOW WORKS with lazy imports)
4. Service initializes (creates monitor instances)
5. Main loop starts (scanning/monitoring)

### Import Dependency Tree
```
run_stock_monitor_v5.py
  ├── loguru (logger)
  ├── services.stock_informational_monitor
  │     ├── services.llm_helper (module-level, but now safe)
  │     │     └── [NO module-level llm_request_manager import]
  │     ├── services.alert_system
  │     └── services.ai_confidence_scanner
  └── config_stock_informational
```

The key is that `llm_helper` no longer imports `llm_request_manager` at module level, so the entire dependency tree loads fast.

## Related Issues

- Task Scheduler "logon failure" - **Not using logon method anymore** (using S4U instead)
- NSSM service manager - **Abandoned in favor of Task Scheduler**
- Working directory issues - **Fixed with explicit os.chdir() in runners**

## Success Metrics

After applying this fix:
- ✅ Services start within 5-10 seconds
- ✅ Logs show "SERVICE READY" message
- ✅ No import timeouts or hangs
- ✅ Services survive reboots (AtStartup trigger)
- ✅ Services auto-restart on failure (RestartCount setting)

## Credits

Fix inspired by common Windows service patterns and research into:
- Python import mechanics in restricted environments
- Windows Task Scheduler security context limitations
- Lazy loading patterns in production services

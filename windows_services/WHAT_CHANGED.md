# What Changed - Task Scheduler Fix

## The Core Problem

Task Scheduler services were **hanging during import** of `services.llm_helper`.

### Import Chain That Caused the Hang
```
run_stock_monitor.py
  └── import services.stock_informational_monitor
        └── import services.llm_helper  ⚠️ HANGS HERE
              └── import services.llm_request_manager (module-level)
                    ├── import requests  ❌ Network library
                    ├── import asyncio   ❌ Event loop
                    └── import socket    ❌ Socket operations
```

**Why it hung**: In Task Scheduler's restricted environment, these network libraries couldn't initialize properly at module-level import time.

---

## The Fix: Lazy Imports

### Changed File: `services/llm_helper.py`

#### BEFORE (Broken) ❌
```python
"""
LLM Helper
"""
import logging
from typing import Optional, Dict, Any

# This line caused the hang:
from services.llm_request_manager import get_llm_manager  # ❌ Module-level import
from models.llm_models import LLMPriority

logger = logging.getLogger(__name__)


class LLMHelper:
    def __init__(self, service_name: str, default_priority: str = "MEDIUM"):
        self.service_name = service_name
        self.default_priority = default_priority
        self._manager = None
    
    @property
    def manager(self):
        """Lazy-load the LLM manager"""
        if self._manager is None:
            self._manager = get_llm_manager()  # Uses already-imported function
        return self._manager
```

#### AFTER (Fixed) ✅
```python
"""
LLM Helper - Easy integration adapter for existing services

IMPORTANT: This module uses LAZY IMPORTS to prevent hangs when running 
as Windows Service/Task Scheduler. DO NOT add module-level imports of 
llm_request_manager - it must be imported inside functions only.
"""
import logging
from typing import Optional, Dict, Any

# DO NOT IMPORT llm_request_manager at module level - causes Task Scheduler hangs
# from services.llm_request_manager import get_llm_manager  # MOVED TO FUNCTION LEVEL

logger = logging.getLogger(__name__)


class LLMHelper:
    def __init__(self, service_name: str, default_priority: str = "MEDIUM"):
        self.service_name = service_name
        self.default_priority = default_priority
        self._manager = None
    
    @property
    def manager(self):
        """Lazy-load the LLM manager only when actually needed"""
        if self._manager is None:
            # Import only when needed, not at module level
            from services.llm_request_manager import get_llm_manager  # ✅ Lazy import
            self._manager = get_llm_manager()
        return self._manager
```

### Also Fixed: Convenience Functions

```python
# BEFORE ❌
def llm_request(prompt: str, service_name: str, ...):
    manager = get_llm_manager()  # Would fail - not imported
    return manager.request(...)

# AFTER ✅  
def llm_request(prompt: str, service_name: str, ...):
    from services.llm_request_manager import get_llm_manager  # Lazy import
    manager = get_llm_manager()
    return manager.request(...)
```

---

## New Files Created

### 1. Updated Service Runner
**`windows_services/runners/run_stock_monitor_v5.py`**
- Cleaner error handling
- Better logging
- Explicit working directory management
- Uses the fixed lazy imports

### 2. Test Script
**`windows_services/test_lazy_imports.py`**
- Tests that all imports work without hanging
- Verifies lazy loading behavior
- Confirms services can be instantiated

### 3. Setup Script
**`windows_services/setup_task_scheduler_v2.ps1`**
- Automated Task Scheduler setup
- Uses V5 runner
- Configures all 3 services
- Shows status and logs

### 4. Documentation
**`docs/TASK_SCHEDULER_FIX.md`**
- Complete explanation of problem and solution
- Deployment guide
- Troubleshooting steps
- Best practices

---

## How Lazy Imports Fixed It

### Import Timeline Comparison

#### BEFORE (Module-level imports) ❌
```
0.00s: Start importing llm_helper
0.01s: Import llm_request_manager
0.02s:   - Import requests
0.03s:   - Import asyncio  
0.XX s:   - Network init... ⏱️ HANGS FOREVER
```

#### AFTER (Lazy imports) ✅
```
0.00s: Start importing llm_helper
0.01s: Import complete! ✅
...
Later, when actually used:
0.00s: First LLM request made
0.01s: Import llm_request_manager NOW
0.05s: Successfully initialized
0.10s: Request sent
```

---

## Why This Works

### Module-level Import (Bad for Services)
```python
# At module level - runs when file is imported
from services.llm_request_manager import get_llm_manager

# Problem: This import chain happens IMMEDIATELY when someone does:
from services.llm_helper import get_llm_helper
# Even if they never call get_llm_helper()!
```

### Function-level Import (Good for Services)
```python
# Inside function - only runs when function is called
def manager(self):
    if self._manager is None:
        from services.llm_request_manager import get_llm_manager
        self._manager = get_llm_manager()
    return self._manager

# Benefit: Import only happens when manager property is accessed
# If service never uses LLM features, import never happens!
```

---

## Impact on Other Services

### Stock Monitor ✅
- **Uses**: `llm_helper` → Fixed
- **Status**: Now works in Task Scheduler

### Crypto Breakout ✅
- **Uses**: Direct imports, no `llm_helper`
- **Status**: Already worked (now verified with better runner)

### DEX Launch ✅
- **Uses**: Direct imports, no `llm_helper`
- **Status**: Already worked (now verified with better runner)

---

## No Functional Changes

**Important**: This fix only changes WHEN imports happen, not WHAT the code does.

- ✅ All functionality preserved
- ✅ No API changes
- ✅ No behavior changes
- ✅ Just faster, non-blocking imports

The services work exactly the same, they just start faster and don't hang in Task Scheduler.

---

## Before/After Comparison

### Service Startup Behavior

#### BEFORE ❌
```
Task Scheduler starts service...
  Importing modules...
  Loading llm_helper...
    Loading llm_request_manager...
      Loading requests...
        Initializing network... ⏱️
        ⏱️ Hang... waiting... timeout...
  ❌ Service appears running but isn't
```

#### AFTER ✅
```
Task Scheduler starts service...
  Importing modules... ✅ (< 1 second)
  Creating monitor... ✅
  Starting scan loop... ✅
  🚀 SERVICE READY
  
Later, when LLM is actually needed:
  First LLM request...
  Loading llm_request_manager now... ✅
  Request processed... ✅
```

---

## Testing the Fix

### Quick Test
```powershell
# This should complete in < 1 second and show all tests passing:
python windows_services\test_lazy_imports.py
```

### Detailed Test
```powershell
# This should show each import completing without hang:
python windows_services\runners\run_stock_monitor_debug.py

# Look for:
# ✓ services.llm_helper - SHOULD NOW WORK (lazy imports added)
```

### Full Integration Test
```powershell
# Setup and run in Task Scheduler:
.\windows_services\setup_task_scheduler_v2.ps1

# Check logs - should see "SERVICE READY" within seconds:
Get-Content logs\stock_monitor_service.log -Tail 20
```

---

## Lessons Learned

### 1. Services Are Different From Interactive Scripts
- Interactive: Full environment, GUI, user context
- Service: Limited environment, no GUI, system context

### 2. Module-level Imports Can Be Dangerous
- Fine for scripts
- Problematic for services
- Can hang on network/GUI operations

### 3. Lazy Imports Are Better for Services
- Faster startup
- Conditional loading
- Isolate import errors
- Work in restricted environments

### 4. Test in Target Environment
- Don't assume Task Scheduler = PowerShell
- Always test services as they'll run in production
- Use debug scripts to identify bottlenecks

---

## Summary

**One simple change** (moving imports inside functions) **fixed all services**.

The fix is elegant, minimal, and follows Python best practices for production services.

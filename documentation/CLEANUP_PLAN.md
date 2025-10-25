# Project Root Cleanup Plan

## Current Issues
- 30+ files in root directory
- Multiple app versions (app.py, app_backup.py, app_modular.py, app_new.py, app_tabs_new.py)
- Demo scripts scattered
- No clear separation between production and development files

## Recommended Structure

```
AI Options Trader/
├── 📂 src/                          # Main source code
│   ├── models/                      ✅ Already created
│   ├── analyzers/                   ✅ Already created
│   ├── clients/                     ✅ Already created
│   ├── utils/                       ✅ Already created
│   └── integrations/                🆕 Move integration clients here
│       ├── tradier_client.py
│       ├── ibkr_client.py
│       ├── discord_alert_listener.py
│       ├── discord_config.py
│       └── discord_ui_tab.py
│
├── 📂 services/                     🆕 Business logic services
│   ├── ai_confidence_scanner.py
│   ├── ai_trading_signals.py
│   ├── penny_stock_analyzer.py
│   ├── ticker_manager.py
│   ├── top_trades_scanner.py
│   ├── watchlist_manager.py
│   ├── llm_strategy_analyzer.py
│   └── options_pricing.py
│
├── 📂 demos/                        🆕 Demo and example scripts
│   ├── demo_ai_confidence.py
│   ├── demo_new_features.py
│   └── discord_integration_example.py
│
├── 📂 scripts/                      🆕 Utility scripts
│   ├── check_env.py
│   ├── fix_app.py
│   └── test_modules.py
│
├── 📂 data/                         🆕 Databases and data files
│   ├── tickers.db
│   └── watchlist.db
│
├── 📂 logs/                         🆕 Log files
│   └── trading_signals.log
│
├── 📂 archive/                      🆕 Old/backup versions
│   ├── app_backup.py
│   ├── app_modular.py
│   ├── app_new.py
│   └── app_tabs_new.py
│
├── 📂 tests/                        ✅ Already exists
│   └── ...
│
├── 📂 documentation/                ✅ Already exists
│   └── ...
│
├── app.py                           ✅ Main application (keep in root)
├── requirements.txt                 ✅ Keep in root
├── .env                            ✅ Keep in root
├── .gitignore                      ✅ Keep in root
├── README.md                       ✅ Keep in root
├── MODULAR_ARCHITECTURE.md         ✅ Keep in root
└── CLEANUP_PLAN.md                 📄 This file

```

## Migration Commands

### Step 1: Create new directories
```powershell
mkdir src\integrations, services, demos, scripts, data, logs, archive
```

### Step 2: Move integration clients
```powershell
Move-Item tradier_client.py src\integrations\
Move-Item ibkr_client.py src\integrations\
Move-Item discord_alert_listener.py src\integrations\
Move-Item discord_config.py src\integrations\
Move-Item discord_ui_tab.py src\integrations\
Move-Item discord_integration_example.py demos\
```

### Step 3: Move service modules
```powershell
Move-Item ai_confidence_scanner.py services\
Move-Item ai_trading_signals.py services\
Move-Item penny_stock_analyzer.py services\
Move-Item ticker_manager.py services\
Move-Item top_trades_scanner.py services\
Move-Item watchlist_manager.py services\
Move-Item llm_strategy_analyzer.py services\
Move-Item options_pricing.py services\
```

### Step 4: Move demos and scripts
```powershell
Move-Item demo_ai_confidence.py demos\
Move-Item demo_new_features.py demos\
Move-Item check_env.py scripts\
Move-Item fix_app.py scripts\
Move-Item test_modules.py scripts\
```

### Step 5: Move data and logs
```powershell
Move-Item tickers.db data\
Move-Item watchlist.db data\
Move-Item trading_signals.log logs\
```

### Step 6: Archive old versions
```powershell
Move-Item app_backup.py archive\
Move-Item app_modular.py archive\
Move-Item app_new.py archive\
Move-Item app_tabs_new.py archive\
```

## After Migration - Update Imports

You'll need to update imports in `app.py`:

### Before:
```python
from tradier_client import TradierClient
from ibkr_client import IBKRClient
from penny_stock_analyzer import PennyStockAnalyzer
from ticker_manager import TickerManager
# etc.
```

### After:
```python
from src.integrations.tradier_client import TradierClient
from src.integrations.ibkr_client import IBKRClient
from services.penny_stock_analyzer import PennyStockAnalyzer
from services.ticker_manager import TickerManager
# etc.
```

## Benefits

✅ **Clear organization** - Production vs development files separated
✅ **Easier navigation** - Logical grouping of related files
✅ **Professional structure** - Standard Python project layout
✅ **Cleaner root** - Only essential config files at root
✅ **Better git workflow** - Easier to see what's changed

## Files to Keep in Root

- `app.py` - Main entry point
- `requirements.txt` - Dependencies
- `.env` - Environment variables
- `.gitignore` - Git configuration
- `README.md` - Project documentation
- `MODULAR_ARCHITECTURE.md` - Architecture guide

## Optional: Create src/ Package

Add `src/__init__.py` to make it a proper package, then you can install it:
```powershell
pip install -e .
```

This allows cleaner imports from anywhere!

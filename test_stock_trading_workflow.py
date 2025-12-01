#!/usr/bin/env python3
"""
Stock Trading Workflow Test Suite
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(test_name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"       {details}")

def test_imports():
    print_header("TEST 1: Import Verification")
    modules = [
        ("services.ai_stock_position_manager", "AIStockPositionManager"),
        ("services.stock_informational_monitor", "StockInformationalMonitor"),
        ("services.discord_trade_approval", "DiscordApprovalManager"),
        ("services.service_orchestrator", "ServiceOrchestrator"),
        ("windows_services.runners.service_config_loader", "queue_analysis_request"),
    ]
    all_passed = True
    for mod, cls in modules:
        try:
            m = __import__(mod, fromlist=[cls])
            getattr(m, cls)
            print_result(f"Import {mod}.{cls}", True)
        except Exception as e:
            print_result(f"Import {mod}.{cls}", False, str(e))
            all_passed = False
    return all_passed

def test_presets():
    print_header("TEST 2: Analysis Presets")
    from windows_services.runners.service_config_loader import ANALYSIS_PRESETS
    presets = ["stock_standard", "stock_multi", "stock_ultimate"]
    all_passed = True
    for p in presets:
        if p in ANALYSIS_PRESETS:
            print_result(f"Preset '{p}'", True, f"mode={ANALYSIS_PRESETS[p].get('analysis_mode')}")
        else:
            print_result(f"Preset '{p}'", False, "Not found")
            all_passed = False
    return all_passed

def test_position_manager():
    print_header("TEST 3: AI Stock Position Manager")
    from services.ai_stock_position_manager import get_ai_stock_position_manager
    mgr = get_ai_stock_position_manager()
    if mgr:
        print_result("Manager initialized", True)
        print_result("Paper mode", mgr.paper_mode, f"paper_mode={mgr.paper_mode}")
        print_result("Broker", mgr.broker_adapter is not None, "Configured" if mgr.broker_adapter else "Not set")
        return True
    print_result("Manager", False)
    return False

def test_orchestrator():
    print_header("TEST 4: Service Orchestrator")
    from services.service_orchestrator import get_orchestrator
    orch = get_orchestrator()
    if orch:
        print_result("Orchestrator", True)
        alert = orch.add_alert(symbol="TEST", alert_type="TEST", source="test", asset_type="stock", price=100.0, reasoning="Test", confidence="HIGH", expires_minutes=1)
        print_result("Add alert", alert is not None)
        orch.reject_alert(alert.id)
        return True
    return False

def test_env():
    print_header("TEST 5: Environment")
    vars = ['BROKER_TYPE', 'STOCK_PAPER_MODE', 'DISCORD_BOT_TOKEN', 'DISCORD_WEBHOOK_URL']
    for v in vars:
        val = os.getenv(v)
        masked = "***" + val[-4:] if val and ('KEY' in v or 'TOKEN' in v) else (val or "NOT SET")
        print_result(v, bool(val), masked)
    return True

def main():
    print("\n" + "=" * 70)
    print("  STOCK TRADING WORKFLOW TEST")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    results = {
        'imports': test_imports(),
        'presets': test_presets(),
        'position_manager': test_position_manager(),
        'orchestrator': test_orchestrator(),
        'environment': test_env(),
    }
    
    print_header("SUMMARY")
    passed = sum(results.values())
    for k, v in results.items():
        print_result(k, v)
    print(f"\n  {passed}/{len(results)} tests passed")
    return passed == len(results)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)

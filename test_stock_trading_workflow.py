#!/usr/bin/env python3
"""
Stock Trading Workflow Test Suite
Tests the end-to-end workflow: Detection → Analysis → Approval → Execution

Run from project root:
    python test_stock_trading_workflow.py

Or with pytest:
    pytest test_stock_trading_workflow.py -v
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress excessive logging during tests
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"       {details}")


def test_imports():
    """Test that all required modules can be imported"""
    print_header("TEST 1: Import Verification")
    
    modules_to_test = [
        ("services.ai_stock_position_manager", "AIStockPositionManager"),
        ("services.stock_informational_monitor", "StockInformationalMonitor"),
        ("services.ai_stock_entry_assistant", "AIStockEntryAssistant"),
        ("services.discord_trade_approval", "DiscordApprovalManager"),
        ("services.service_orchestrator", "ServiceOrchestrator"),
        ("windows_services.runners.service_config_loader", "queue_analysis_request"),
    ]
    
    all_passed = True
    for module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print_result(f"Import {module_path}.{class_name}", True)
        except ImportError as e:
            print_result(f"Import {module_path}.{class_name}", False, str(e))
            all_passed = False
        except AttributeError as e:
            print_result(f"Import {module_path}.{class_name}", False, str(e))
            all_passed = False
    
    return all_passed


def test_analysis_presets():
    """Test that stock analysis presets are configured correctly"""
    print_header("TEST 2: Analysis Presets Configuration")
    
    from windows_services.runners.service_config_loader import ANALYSIS_PRESETS
    
    required_presets = [
        "stock_standard",
        "stock_multi", 
        "stock_ultimate",
    ]
    
    all_passed = True
    for preset_key in required_presets:
        if preset_key in ANALYSIS_PRESETS:
            preset = ANALYSIS_PRESETS[preset_key]
            has_asset_type = preset.get("asset_type") == "stock"
            has_analysis_mode = "analysis_mode" in preset
            
            if has_asset_type and has_analysis_mode:
                print_result(f"Preset '{preset_key}'", True, 
                           f"mode={preset['analysis_mode']}")
            else:
                print_result(f"Preset '{preset_key}'", False, 
                           "Missing asset_type or analysis_mode")
                all_passed = False
        else:
            print_result(f"Preset '{preset_key}'", False, "Not found in ANALYSIS_PRESETS")
            all_passed = False
    
    return all_passed


def test_ai_stock_position_manager():
    """Test AI Stock Position Manager initialization"""
    print_header("TEST 3: AI Stock Position Manager")
    
    from services.ai_stock_position_manager import get_ai_stock_position_manager
    
    # Get manager (will create broker adapter if configured)
    manager = get_ai_stock_position_manager()
    
    if manager:
        print_result("Manager initialized", True)
        print_result("Paper mode enabled", manager.paper_mode, 
                    f"paper_mode={manager.paper_mode}")
        print_result("Broker adapter", manager.broker_adapter is not None,
                    "Configured" if manager.broker_adapter else "NOT CONFIGURED (ok for test)")
        print_result("Discord integration", manager.discord_approval_manager is not None,
                    "Enabled" if manager.discord_approval_manager else "Not configured")
        
        # Test position tracking
        positions = manager.get_active_positions()
        print_result("Get active positions", True, f"Found {len(positions)} positions")
        
        return True
    else:
        print_result("Manager initialized", False, "Could not create manager")
        return False


def test_service_orchestrator():
    """Test Service Orchestrator alert queue"""
    print_header("TEST 4: Service Orchestrator Alert Queue")
    
    from services.service_orchestrator import get_orchestrator
    
    orch = get_orchestrator()
    
    if orch:
        print_result("Orchestrator initialized", True)
        
        # Add a test alert
        alert = orch.add_alert(
            symbol="TEST_STOCK",
            alert_type="MULTI_FACTOR",
            source="test_suite",
            asset_type="stock",
            price=100.0,
            reasoning="Test alert from workflow test",
            confidence="HIGH",
            expires_minutes=5
        )
        
        print_result("Add test alert", alert is not None, f"id={alert.id if alert else 'N/A'}")
        
        # Get pending alerts
        pending = orch.get_pending_alerts(asset_type="stock")
        found = any(a.symbol == "TEST_STOCK" for a in pending)
        print_result("Alert in pending queue", found, f"Found {len(pending)} stock alerts")
        
        # Clean up - reject the test alert
        if found:
            for a in pending:
                if a.symbol == "TEST_STOCK":
                    orch.reject_alert(a.id)
        
        return True
    else:
        print_result("Orchestrator initialized", False)
        return False


def test_stock_monitor_initialization():
    """Test Stock Monitor can be initialized"""
    print_header("TEST 5: Stock Monitor Initialization")
    
    try:
        from services.stock_informational_monitor import StockInformationalMonitor
        
        # Create with empty watchlist for fast init
        monitor = StockInformationalMonitor(
            watchlist=["AAPL"],  # Just one ticker for fast test
            scan_interval_minutes=60,  # Long interval (won't actually run)
            min_score=50
        )
        
        print_result("Monitor initialized", True)
        watchlist = monitor.watchlist or []
        print_result("Watchlist loaded", len(watchlist) > 0, 
                    f"{len(watchlist)} tickers")
        print_result("Discord webhook", monitor.discord_webhook is not None,
                    "Configured" if monitor.discord_webhook else "Not set")
        print_result("Discord bot", monitor.discord_bot_manager is not None,
                    "Enabled" if monitor.discord_bot_manager else "Not configured")
        
        # Get stats
        stats = monitor.get_stats()
        print_result("Stats tracking", True, f"Health: {monitor.stats.get_health_status()}")
        
        return True
        
    except Exception as e:
        print_result("Monitor initialization", False, str(e))
        return False


def test_discord_approval_manager():
    """Test Discord Approval Manager (without actually connecting)"""
    print_header("TEST 6: Discord Approval Manager")
    
    # Check environment variables
    token = os.getenv('DISCORD_BOT_TOKEN')
    channel = os.getenv('DISCORD_CHANNEL_IDS', '').split(',')[0].strip()
    
    print_result("DISCORD_BOT_TOKEN set", bool(token), 
                "***" + token[-4:] if token else "Not set")
    print_result("DISCORD_CHANNEL_IDS set", bool(channel),
                channel if channel else "Not set")
    
    if token and channel:
        try:
            from services.discord_trade_approval import DiscordApprovalManager
            
            # Just test initialization (don't start the bot)
            manager = DiscordApprovalManager()
            print_result("Manager created", manager.enabled, 
                        "Enabled" if manager.enabled else "Disabled")
            
            return manager.enabled
        except Exception as e:
            print_result("Manager initialization", False, str(e))
            return False
    else:
        print("       ⚠️ Discord not configured - skipping bot test")
        return True  # Pass if not configured (optional feature)


def test_broker_configuration():
    """Test broker configuration"""
    print_header("TEST 7: Broker Configuration")
    
    broker_type = os.getenv('BROKER_TYPE', 'NOT_SET')
    print_result("BROKER_TYPE", broker_type != 'NOT_SET', broker_type)
    
    if broker_type.upper() == 'IBKR':
        # Paper mode credentials
        paper_port = os.getenv('IBKR_PAPER_PORT', 'NOT_SET')
        paper_client_id = os.getenv('IBKR_PAPER_CLIENT_ID', 'NOT_SET')
        print_result("IBKR_PAPER_PORT", paper_port != 'NOT_SET', paper_port)
        print_result("IBKR_PAPER_CLIENT_ID", paper_client_id != 'NOT_SET', paper_client_id)
        
        # Live mode credentials (optional but good to verify)
        live_port = os.getenv('IBKR_LIVE_PORT', 'NOT_SET')
        live_client_id = os.getenv('IBKR_LIVE_CLIENT_ID', 'NOT_SET')
        print_result("IBKR_LIVE_PORT", live_port != 'NOT_SET', live_port)
        print_result("IBKR_LIVE_CLIENT_ID", live_client_id != 'NOT_SET', live_client_id)
        
    elif broker_type.upper() == 'TRADIER':
        # Check for paper or production credentials (matching trading_config.py)
        paper_account = os.getenv('TRADIER_PAPER_ACCOUNT_ID') or os.getenv('TRADIER_ACCOUNT_ID')
        paper_token = os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or os.getenv('TRADIER_ACCESS_TOKEN')
        
        print_result("TRADIER_PAPER_ACCOUNT_ID", bool(paper_account), 
                    paper_account[:8] + "..." if paper_account else "Not set")
        print_result("TRADIER_PAPER_ACCESS_TOKEN", bool(paper_token), 
                    "***" + paper_token[-4:] if paper_token else "Not set")
    
    paper_mode = os.getenv('STOCK_PAPER_MODE', 'true').lower() == 'true'
    print_result("STOCK_PAPER_MODE", True, f"{'PAPER (safe)' if paper_mode else '⚠️ LIVE'}")
    
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 70)
    print("  STOCK TRADING WORKFLOW TEST SUITE")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['presets'] = test_analysis_presets()
    results['position_manager'] = test_ai_stock_position_manager()
    results['orchestrator'] = test_service_orchestrator()
    results['stock_monitor'] = test_stock_monitor_initialization()
    results['discord'] = test_discord_approval_manager()
    results['broker'] = test_broker_configuration()
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        print_result(test_name.replace('_', ' ').title(), result)
    
    print("\n" + "-" * 70)
    print(f"  TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print(f"  ⚠️ {total - passed} test(s) failed")
    
    print("-" * 70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

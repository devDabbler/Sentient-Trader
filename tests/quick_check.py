import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# quick_check.py
print("Checking Phase 3 components...")

try:
    from services.alert_system import get_alert_system
    print("✓ Alert system")
except Exception as e:
    print(f"✗ Alert system: {e}")

try:
    from services.backtest_ema_fib import EMAFibonacciBacktester
    print("✓ Backtester")
except Exception as e:
    print(f"✗ Backtester: {e}")

try:
    from services.preset_scanners import PresetScanner
    print("✓ Scanner")
except Exception as e:
    print(f"✗ Scanner: {e}")

try:
    from services.options_chain_fib import FibonacciOptionsChain
    print("✓ Options chain")
except Exception as e:
    print(f"✗ Options chain: {e}")

print("\nTesting SOFI analysis...")
from analyzers.comprehensive import ComprehensiveAnalyzer
analysis = ComprehensiveAnalyzer.analyze_stock("SOFI", "SWING_TRADE")
print(f"Analysis result: {analysis is not None}")
if analysis:
    print(f"  EMA8: {analysis.ema8}")
    print(f"  EMA21: {analysis.ema21}")
    print(f"  DeMarker: {analysis.demarker}")
    print(f"  Fib Targets: {analysis.fib_targets is not None}")
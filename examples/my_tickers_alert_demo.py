"""
My Tickers Alert Demo

Demonstrates how alerts work with your saved "My Tickers" (TSLA, AMC, AMD, COIN, etc.)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.ticker_manager import TickerManager
from services.preset_scanners import PresetScanner, ScanPreset
from services.alert_system import get_alert_system, console_callback
from analyzers.comprehensive import ComprehensiveAnalyzer


def demo_my_tickers_alerts():
    """Demonstrate alert filtering based on My Tickers"""
    
    print("=" * 80)
    print("MY TICKERS ALERT DEMO")
    print("=" * 80)
    print()
    
    # 1. Initialize ticker manager and show current My Tickers
    print("1. Loading your saved tickers from My Tickers...")
    ticker_manager = TickerManager()
    my_tickers = ticker_manager.get_all_tickers()
    
    if not my_tickers:
        print("   ⚠️  No tickers found in My Tickers!")
        print("   Add some tickers first using your app or run:")
        print()
        print("   from services.ticker_manager import TickerManager")
        print("   tm = TickerManager()")
        print("   tm.add_ticker('TSLA', name='Tesla Inc', ticker_type='stock')")
        print()
        return
    
    print(f"   ✓ Found {len(my_tickers)} tickers in My Tickers:")
    for ticker in my_tickers:
        ml_score = ticker.get('ml_score', 'N/A')
        print(f"     - {ticker['ticker']}: {ticker.get('name', 'N/A')} (ML Score: {ml_score})")
    print()
    
    # 2. Setup alert system
    print("2. Setting up alert system...")
    alert_system = get_alert_system()
    alert_system.add_callback(console_callback)
    print("   ✓ Alert system configured with console output")
    print()
    
    # 3. Create scanner WITH My Tickers filtering
    print("3. Creating scanner with 'My Tickers Only' filter...")
    scanner_filtered = PresetScanner(
        alert_system=alert_system,
        my_tickers_only=True,
        ticker_manager=ticker_manager
    )
    print("   ✓ Scanner will only alert for your saved tickers")
    print()
    
    # 4. Compare: Scanner WITHOUT filtering
    print("4. Creating scanner WITHOUT filter (for comparison)...")
    scanner_all = PresetScanner(alert_system=alert_system)
    print("   ✓ Scanner will alert for any ticker")
    print()
    
    # 5. Scan a mixed list (includes your tickers + others)
    print("5. Scanning a mixed list of tickers...")
    print()
    
    # Get your ticker symbols
    my_ticker_symbols = [t['ticker'] for t in my_tickers]
    
    # Create test list: your tickers + some random ones
    test_tickers = my_ticker_symbols + ["AAPL", "MSFT", "GOOGL", "NVDA"]
    
    print(f"   Scanning: {', '.join(test_tickers)}")
    print(f"   Your My Tickers: {', '.join(my_ticker_symbols)}")
    print(f"   Others: AAPL, MSFT, GOOGL, NVDA")
    print()
    
    # 6. Demonstrate the difference
    print("=" * 80)
    print("SCENARIO 1: Scanning WITH 'My Tickers Only' filter")
    print("=" * 80)
    print()
    
    # Manually analyze a couple tickers to trigger alerts
    for ticker in test_tickers[:4]:  # Just scan first 4 to keep it quick
        print(f"Analyzing {ticker}...")
        try:
            analysis = ComprehensiveAnalyzer.analyze_stock(ticker, "SWING_TRADE")
            if analysis:
                # This will only alert if ticker is in My Tickers
                alerts = scanner_filtered.setup_detector.analyze_for_alerts(analysis)
                
                if alerts:
                    print(f"  ✓ {ticker}: Generated {len(alerts)} alert(s)")
                else:
                    if ticker in my_ticker_symbols:
                        print(f"  - {ticker}: In My Tickers, but no alert conditions met")
                    else:
                        print(f"  - {ticker}: NOT in My Tickers, alerts skipped")
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")
    
    print()
    print("=" * 80)
    print("WHAT HAPPENED?")
    print("=" * 80)
    print()
    print("✅ Tickers in My Tickers → Analyzed and alerted (if conditions met)")
    print("❌ Other tickers → Analyzed but NO alerts sent")
    print()
    print("This means Discord will ONLY notify you about stocks you're tracking!")
    print()
    
    # 7. Show alert summary
    recent_alerts = alert_system.get_recent_alerts(count=10)
    
    if recent_alerts:
        print("=" * 80)
        print(f"RECENT ALERTS ({len(recent_alerts)} total)")
        print("=" * 80)
        print()
        
        for alert in recent_alerts:
            print(f"[{alert.priority.value}] {alert.ticker}: {alert.message}")
        
        print()
        print("Notice: All alerts are for YOUR tickers only!")
    else:
        print("No alerts generated (tickers didn't meet alert conditions)")
    
    print()


def demo_add_ticker_for_alerts():
    """Show how to add a ticker to start receiving alerts"""
    
    print("=" * 80)
    print("HOW TO ADD TICKERS FOR ALERTS")
    print("=" * 80)
    print()
    
    ticker_manager = TickerManager()
    
    # Example: Add NVDA
    print("Example: Adding NVDA to My Tickers...")
    
    success = ticker_manager.add_ticker(
        ticker="NVDA",
        name="NVIDIA Corporation",
        sector="Technology",
        ticker_type="stock",
        notes="AI chip leader - watching for entries"
    )
    
    if success:
        print("✓ NVDA added to My Tickers!")
        print()
        print("Now NVDA will trigger alerts when:")
        print("  - EMA reclaim detected")
        print("  - Triple threat setup found")
        print("  - High confidence score (85+)")
        print("  - Any other technical setup conditions")
        print()
    
    # Show current list
    all_tickers = ticker_manager.get_all_tickers()
    print(f"Your current My Tickers ({len(all_tickers)} total):")
    for t in all_tickers:
        print(f"  • {t['ticker']}: {t.get('name', 'N/A')}")
    
    print()
    print("=" * 80)
    print("To remove a ticker from alerts:")
    print("=" * 80)
    print()
    print("  ticker_manager.remove_ticker('NVDA')")
    print()


def show_current_my_tickers():
    """Display current My Tickers configuration"""
    
    print("=" * 80)
    print("YOUR CURRENT MY TICKERS")
    print("=" * 80)
    print()
    
    ticker_manager = TickerManager()
    my_tickers = ticker_manager.get_all_tickers()
    
    if not my_tickers:
        print("❌ No tickers saved yet!")
        print()
        print("Add tickers in your app (⭐ My Tickers tab) or via code:")
        print()
        print("  from services.ticker_manager import TickerManager")
        print("  tm = TickerManager()")
        print("  tm.add_ticker('TSLA', name='Tesla Inc', ticker_type='stock')")
        print()
        return
    
    print(f"Total: {len(my_tickers)} tickers\n")
    
    for ticker in my_tickers:
        print(f"📌 {ticker['ticker']}")
        if ticker.get('name'):
            print(f"   Name: {ticker['name']}")
        if ticker.get('sector'):
            print(f"   Sector: {ticker['sector']}")
        if ticker.get('ml_score'):
            print(f"   ML Score: {ticker['ml_score']}")
        if ticker.get('notes'):
            print(f"   Notes: {ticker['notes']}")
        print()
    
    print("=" * 80)
    print("These tickers will receive alerts when my_tickers_only=True")
    print("=" * 80)
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="My Tickers Alert Demo")
    parser.add_argument(
        '--mode',
        choices=['demo', 'add', 'show'],
        default='show',
        help='Mode: demo (test alerts), add (add ticker example), show (display current)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_my_tickers_alerts()
    elif args.mode == 'add':
        demo_add_ticker_for_alerts()
    else:
        show_current_my_tickers()

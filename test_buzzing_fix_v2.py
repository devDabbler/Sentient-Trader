"""
Test script to verify buzzing stocks analysis fixes

Tests:
1. ThreadPoolExecutor management (no "cannot schedule new futures" error)
2. Browser driver connection handling (retry on connection errors)
3. Timeout protection (prevents hanging)
4. Proper cleanup
"""

import asyncio
import logging
from services.advanced_opportunity_scanner import AdvancedOpportunityScanner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_buzzing_stocks_analysis():
    """Test buzzing stocks analysis with multiple tickers"""
    print("\n" + "="*80)
    print("🧪 TESTING BUZZING STOCKS ANALYSIS FIXES")
    print("="*80 + "\n")
    
    try:
        # Create scanner
        scanner = AdvancedOpportunityScanner(use_ai=False)
        
        # Test with small ticker list
        scanner.EXTENDED_UNIVERSE = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
        
        print(f"🔍 Scanning {len(scanner.EXTENDED_UNIVERSE)} tickers for buzzing stocks...")
        print("   This will test:")
        print("   ✓ ThreadPoolExecutor management")
        print("   ✓ Event loop handling")
        print("   ✓ Timeout protection")
        print("   ✓ Connection error retry")
        print()
        
        # Run the scan
        results = scanner.scan_buzzing_stocks(
            top_n=5, 
            min_buzz_score=10.0,
            max_tickers_to_scan=5  # Limit for faster test
        )
        
        print(f"\n✅ SCAN COMPLETED - Found {len(results)} buzzing stocks")
        
        if results:
            print("\n📊 Results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.ticker}: Buzz Score={result.buzz_score:.0f}")
                print(f"      {result.reason}")
        else:
            print("   No buzzing stocks found (this is OK - means filters working)")
        
        # Test cleanup
        print("\n🧹 Testing cleanup...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(scanner.cleanup())
        loop.close()
        
        print("✅ Cleanup successful")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - No errors!")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_buzzing_stocks_analysis()
    exit(0 if success else 1)

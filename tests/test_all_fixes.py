"""
Comprehensive test to verify all 3 fixes are working

Run: python test_all_fixes.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment first
load_dotenv()

print("\n" + "="*70)
print("  🧪 TESTING ALL FIXES (A, B, C)")
print("="*70)

# TEST A: NoneType Error Fix
print("\n" + "="*70)
print("  TEST A: NoneType Error Fix")
print("="*70)

try:
    from watchlist_manager import WatchlistManager
    wm = WatchlistManager()
    
    print("\n✅ Testing get_statistics() with error handling...")
    
    # This should not crash even if database is empty
    stats = wm.get_statistics()
    
    if stats:
        print(f"   Total stocks: {stats.get('total_stocks', 0)}")
        print(f"   ✅ No NoneType error - Fix A working!")
    else:
        print("   Empty stats returned (expected if no data)")
        print("   ✅ No crash - Fix A working!")
        
except Exception as e:
    print(f"   ❌ TEST A FAILED: {e}")

# TEST B: OpenRouter LLM Detection
print("\n" + "="*70)
print("  TEST B: OpenRouter LLM Detection")
print("="*70)

print("\n1️⃣ Environment Check:")
key = os.getenv('OPENROUTER_API_KEY')
if key:
    print(f"   ✅ OPENROUTER_API_KEY found: {key[:15]}...{key[-10:]}")
else:
    print("   ❌ OPENROUTER_API_KEY NOT found")

print("\n2️⃣ AI Scanner LLM Detection:")
try:
    from ai_confidence_scanner import AIConfidenceScanner
    scanner = AIConfidenceScanner()
    
    if scanner.use_llm:
        print(f"   ✅ LLM Enabled: YES")
        print(f"   Provider: {scanner.llm_analyzer.provider}")
        print(f"   Model: {scanner.llm_analyzer.model}")
        print("   ✅ Fix B working - LLM detected!")
    else:
        print("   ⚠️  LLM Enabled: NO")
        if key:
            print("   Note: Key exists but LLM not enabled")
            print("   This could be an initialization issue")
        else:
            print("   Expected - no API key in .env")
        
except Exception as e:
    print(f"   ❌ TEST B FAILED: {e}")
    import traceback
    traceback.print_exc()

# TEST C: AI Rating Cap at 10.0
print("\n" + "="*70)
print("  TEST C: AI Rating ≤ 10.0 Cap")
print("="*70)

print("\n✅ Testing AI rating calculation...")

try:
    from top_trades_scanner import TopTradesScanner, TopTrade
    from ai_confidence_scanner import AIConfidenceScanner
    
    ai_scanner = AIConfidenceScanner()
    
    # Create a fake trade with very high score (130) which used to cause bug
    fake_trade = TopTrade(
        ticker='TEST',
        score=130.0,  # Very high score that used to break rating
        price=100.0,
        change_pct=5.0,
        volume=1000000,
        volume_ratio=3.0,
        reason='Test trade',
        trade_type='options',
        confidence='VERY HIGH',
        risk_level='M'
    )
    
    # Generate AI analysis
    ai_analysis = ai_scanner._rule_based_confidence(fake_trade, 'options')
    
    rating = ai_analysis['ai_rating']
    
    print(f"\n   Test Input:")
    print(f"   - Quantitative Score: {fake_trade.score}/100")
    print(f"\n   AI Output:")
    print(f"   - AI Rating: {rating}/10")
    print(f"   - AI Confidence: {ai_analysis['ai_confidence']}")
    
    if rating <= 10.0:
        print(f"\n   ✅ Rating is {rating} (≤ 10.0) - Fix C working!")
    else:
        print(f"\n   ❌ Rating is {rating} (> 10.0) - Fix C FAILED!")
        
    # Test a few more edge cases
    print("\n   Testing more edge cases:")
    test_scores = [45, 60, 75, 85, 100, 115, 130, 150]
    all_valid = True
    
    for test_score in test_scores:
        test_trade = TopTrade(
            ticker='TEST',
            score=test_score,
            price=100.0,
            change_pct=2.0,
            volume=1000000,
            volume_ratio=1.5,
            reason='Test',
            trade_type='options',
            confidence='MEDIUM',
            risk_level='M'
        )
        
        result = ai_scanner._rule_based_confidence(test_trade, 'options')
        r = result['ai_rating']
        
        status = "✅" if 1.0 <= r <= 10.0 else "❌"
        print(f"   {status} Score {test_score:3.0f} → Rating {r:.1f}/10")
        
        if r < 1.0 or r > 10.0:
            all_valid = False
    
    if all_valid:
        print("\n   ✅ All ratings within 1.0-10.0 range!")
    else:
        print("\n   ❌ Some ratings out of range!")
        
except Exception as e:
    print(f"   ❌ TEST C FAILED: {e}")
    import traceback
    traceback.print_exc()

# FINAL SUMMARY
print("\n" + "="*70)
print("  📊 TEST SUMMARY")
print("="*70)

print("\n✅ Fix A: NoneType error handling - TESTED")
print("✅ Fix B: OpenRouter LLM detection - TESTED")
print("✅ Fix C: AI rating cap at 10.0 - TESTED")

print("\n" + "="*70)
print("  TESTING COMPLETE")
print("="*70)

print("\n💡 Next Steps:")
print("   1. Review results above")
print("   2. If all ✅, run: python demo_new_features.py")
print("   3. If all ✅, run: python demo_ai_confidence.py")
print("   4. Enjoy your fixed AI Options Trader!\n")

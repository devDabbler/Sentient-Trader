"""
Clean AI Confidence Test - Easier to read output

Run: python test_ai_clean.py
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

from ai_confidence_scanner import AIConfidenceScanner
from top_trades_scanner import TopTradesScanner

print("\n" + "="*70)
print("  🤖 AI CONFIDENCE SCANNER - CLEAN TEST")
print("="*70)

# Check setup
print("\n📋 Setup Check:")
print(f"   OpenRouter Key: {'✅ Found' if os.getenv('OPENROUTER_API_KEY') else '❌ Missing'}")

# Initialize
ai_scanner = AIConfidenceScanner()
print(f"   LLM Enabled: {'✅ YES' if ai_scanner.use_llm else '❌ NO'}")

if ai_scanner.use_llm:
    print(f"   Provider: {ai_scanner.llm_analyzer.provider}")
    print(f"   Model: {ai_scanner.llm_analyzer.model}")
else:
    print("\n⚠️  LLM not available - check your .env file")
    exit()

# Test with ONE stock to see clear output
test_ticker = 'AAPL'

print("\n" + "="*70)
print(f"  📊 ANALYZING {test_ticker} WITH AI")
print("="*70)

print(f"\n🔍 Step 1: Getting quantitative analysis...")

scanner = TopTradesScanner()
trade = scanner._analyze_options_opportunity(test_ticker)

if not trade:
    print(f"❌ Could not analyze {test_ticker}")
    exit()

print(f"✅ Quantitative Score: {trade.score}/100")
print(f"   Price: ${trade.price:.2f}")
print(f"   Change: {trade.change_pct:+.2f}%")
print(f"   Volume Ratio: {trade.volume_ratio:.2f}x")
print(f"   Confidence: {trade.confidence}")

print(f"\n🤖 Step 2: Getting AI analysis (this takes 3-5 seconds)...")

ai_analysis = ai_scanner._generate_ai_confidence(trade, 'options')

print(f"\n" + "="*70)
print(f"  ✨ AI ANALYSIS RESULTS FOR {test_ticker}")
print("="*70)

print(f"\n📊 SCORES:")
print(f"   Quantitative Score: {trade.score}/100")
print(f"   AI Rating: {ai_analysis['ai_rating']}/10 ⭐")
print(f"   AI Confidence: {ai_analysis['ai_confidence']}")

print(f"\n💡 AI REASONING:")
print(f"   {ai_analysis['ai_reasoning']}")

print(f"\n⚠️  AI RISKS:")
print(f"   {ai_analysis['ai_risks']}")

print("\n" + "="*70)
print("  ✅ TEST COMPLETE")
print("="*70)

print("\n📝 Interpretation:")
if "Standard" in ai_analysis['ai_reasoning'] or "standard" in ai_analysis['ai_reasoning']:
    print("   ⚠️  You saw generic 'standard analysis' - LLM may have failed")
    print("   This could be an API issue or rate limit")
else:
    print("   ✅ You saw REAL AI analysis with specific insights!")
    print("   The AI provided custom reasoning for this exact trade")

print("\n💡 Next Step:")
print("   Run: python demo_ai_confidence.py")
print("   This will analyze multiple stocks with AI\n")

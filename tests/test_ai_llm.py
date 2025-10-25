"""
Quick test of AI Confidence Scanner with LLM

Run: python test_ai_llm.py
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ai_confidence_scanner import AIConfidenceScanner

print("="*60)
print("  🤖 TESTING AI CONFIDENCE WITH REAL LLM")
print("="*60)

# Check API key
print(f"\nOpenRouter API Key: {'✅ Found' if os.getenv('OPENROUTER_API_KEY') else '❌ Not found'}")

# Initialize scanner
print("\n🔧 Initializing AI Confidence Scanner...")
ai_scanner = AIConfidenceScanner()

print(f"   LLM Enabled: {'✅ YES' if ai_scanner.use_llm else '❌ NO'}")

if ai_scanner.use_llm:
    print(f"   Provider: {ai_scanner.llm_analyzer.provider}")
    print(f"   Model: {ai_scanner.llm_analyzer.model}")
else:
    print("   ⚠️ Running in rule-based mode\n")
    exit()

# Test with a few stocks
print("\n" + "="*60)
print("  🔍 TESTING WITH SAMPLE STOCKS")
print("="*60)

test_tickers = ['NVDA', 'TSLA', 'AMD']

for ticker in test_tickers:
    print(f"\n📊 Testing {ticker}...")
    try:
        # Scan just this one ticker
        from top_trades_scanner import TopTradesScanner
        scanner = TopTradesScanner()
        
        # Get regular analysis
        trade = scanner._analyze_options_opportunity(ticker)
        
        if trade:
            print(f"   Quant Score: {trade.score}/100")
            print(f"   Confidence: {trade.confidence}")
            
            # Now get AI analysis
            print(f"\n   🤖 Getting AI analysis...")
            ai_analysis = ai_scanner._generate_ai_confidence(trade, 'options')
            
            print(f"\n   ✨ AI RESULTS:")
            print(f"   ├─ AI Rating: {ai_analysis['ai_rating']}/10 ⭐")
            print(f"   ├─ AI Confidence: {ai_analysis['ai_confidence']}")
            print(f"   │")
            print(f"   ├─ 💡 AI Reasoning:")
            print(f"   │  {ai_analysis['ai_reasoning']}")
            print(f"   │")
            print(f"   └─ ⚠️ AI Risks:")
            print(f"      {ai_analysis['ai_risks']}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("  ✅ TEST COMPLETE")
print("="*60)
print("\nIf you saw real AI analysis above, LLM is working!")
print("If you saw 'standard analysis', it fell back to rules.\n")

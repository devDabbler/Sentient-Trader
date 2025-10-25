"""
Test LLM with a single trade - Debug version

Run: python test_single_llm.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("\n" + "="*70)
print("  🔍 DEBUG: Testing Single LLM Call")
print("="*70)

# Check environment
print("\n1️⃣ Environment Check:")
openrouter_key = os.getenv('OPENROUTER_API_KEY')
print(f"   OpenRouter Key: {'✅ Found' if openrouter_key else '❌ Missing'}")
if openrouter_key:
    print(f"   Key preview: {openrouter_key[:15]}...{openrouter_key[-10:]}")

# Test LLM directly
print("\n2️⃣ Testing LLM Analyzer directly...")
try:
    from llm_strategy_analyzer import LLMStrategyAnalyzer
    
    llm = LLMStrategyAnalyzer(provider='openrouter', model='meta-llama/llama-3.3-70b-instruct')
    print(f"   ✅ LLM Analyzer created")
    print(f"   Provider: {llm.provider}")
    print(f"   Model: {llm.model}")
    print(f"   API Key set: {bool(llm.api_key)}")
    
    # Make a simple test call
    print("\n3️⃣ Making test LLM call...")
    test_prompt = """Analyze this trade:
Ticker: AAPL
Price: $220.50
Change: +2.5%
Volume: 2.5x average

Provide:
1. AI Confidence: [VERY HIGH/HIGH/MEDIUM/LOW]
2. AI Rating: [0-10]
3. AI Reasoning: [Brief why]
4. AI Risks: [Brief risks]
"""
    
    print("   Sending request to LLM... (may take 5-10 seconds)")
    
    response = llm._call_llm(test_prompt)
    
    if response:
        print(f"   ✅ Got response ({len(response)} chars)")
        print("\n4️⃣ LLM Response:")
        print("   " + "="*66)
        print("   " + response.replace("\n", "\n   "))
        print("   " + "="*66)
    else:
        print("   ❌ Empty response from LLM")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("  DIAGNOSTIC COMPLETE")
print("="*70)
print("\n💡 If you saw a detailed response above, LLM is working!")
print("   If you saw an error, check your OpenRouter API key.\n")

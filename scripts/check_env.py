"""
Quick check of environment variables

Run: python check_env.py
"""

import os
from dotenv import load_dotenv

print("="*60)
print("  🔍 CHECKING ENVIRONMENT VARIABLES")
print("="*60)

print("\n📁 Current directory:", os.getcwd())
print("📄 .env file exists:", os.path.exists('.env'))

if os.path.exists('.env'):
    with open('.env', 'r') as f:
        lines = f.readlines()
    print(f"📝 .env file has {len(lines)} lines")

print("\n🔧 Loading .env file...")
load_dotenv()

print("\n🔑 API Keys Status:")
print("="*60)

keys = {
    'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
}

for key_name, key_value in keys.items():
    if key_value:
        # Show first 10 and last 10 chars
        masked = f"{key_value[:10]}...{key_value[-10:]}" if len(key_value) > 20 else key_value
        print(f"✅ {key_name}: {masked}")
    else:
        print(f"❌ {key_name}: Not found")

print("\n" + "="*60)

# Test with AI scanner
print("\n🤖 Testing AI Confidence Scanner...")
from ai_confidence_scanner import AIConfidenceScanner

ai_scanner = AIConfidenceScanner()
print(f"   LLM Enabled: {'✅ YES' if ai_scanner.use_llm else '❌ NO'}")

if ai_scanner.use_llm:
    print(f"   Provider: {ai_scanner.llm_analyzer.provider}")
    print(f"   Model: {ai_scanner.llm_analyzer.model}")
    print("\n🎉 SUCCESS! LLM is ready to use!")
else:
    print("\n⚠️ LLM not available - using rule-based mode")
    print("\n💡 Make sure you have a valid API key in .env file:")
    print("   OPENROUTER_API_KEY=sk-or-v1-...")

print("\n" + "="*60)

"""
Test if .env file loads correctly

Run: python test_env_loading.py
"""

import os
import sys

print("="*70)
print("  🔍 TESTING .ENV FILE LOADING")
print("="*70)

print("\n1️⃣ Before loading .env:")
print(f"   OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')}")

print("\n2️⃣ Loading .env...")
from dotenv import load_dotenv
result = load_dotenv()
print(f"   load_dotenv() returned: {result}")

print("\n3️⃣ After loading .env:")
key = os.getenv('OPENROUTER_API_KEY')
if key:
    print(f"   OPENROUTER_API_KEY: {key[:15]}...{key[-10:]}")
    print("   ✅ Key loaded successfully!")
else:
    print("   ❌ Key NOT loaded")

print("\n4️⃣ Checking .env file:")
if os.path.exists('.env'):
    print("   ✅ .env file exists")
    with open('.env', 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    print(f"   Found {len(lines)} non-comment lines")
    
    # Check if OPENROUTER_API_KEY is in the file
    has_openrouter = any('OPENROUTER_API_KEY' in line for line in lines)
    print(f"   OPENROUTER_API_KEY in file: {'✅ Yes' if has_openrouter else '❌ No'}")
else:
    print("   ❌ .env file NOT found")

print("\n5️⃣ Testing AI scanner import:")
try:
    from ai_confidence_scanner import AIConfidenceScanner
    scanner = AIConfidenceScanner()
    print(f"   LLM enabled: {'✅ YES' if scanner.use_llm else '❌ NO'}")
    if scanner.use_llm:
        print(f"   Provider: {scanner.llm_analyzer.provider}")
        print(f"   Model: {scanner.llm_analyzer.model}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("  DIAGNOSTIC COMPLETE")
print("="*70)

if not key:
    print("\n💡 SOLUTION: Make sure .env file has:")
    print("   OPENROUTER_API_KEY=sk-or-v1-...")
    print("\n   Then run this test again.")

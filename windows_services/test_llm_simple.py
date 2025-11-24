"""Test llm_helper import with basic prints"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("1. Starting test...")
print("2. About to import llm_helper...")

try:
    from services.llm_helper import get_llm_helper, LLMServiceMixin
    print("3. ✓ llm_helper imported successfully!")
    print("4. Type of get_llm_helper:", type(get_llm_helper))
    print("5. Test PASSED")
except Exception as e:
    print(f"3. ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("6. Script completed")

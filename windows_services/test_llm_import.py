"""Test just the llm_helper import"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

logger.info("Testing llm_helper import...")

try:
    from services.llm_helper import get_llm_helper, LLMServiceMixin
    logger.info("✓ llm_helper imported successfully!")
except Exception as e:
    logger.error(f"❌ Failed: {e}", exc_info=True)
    sys.exit(1)

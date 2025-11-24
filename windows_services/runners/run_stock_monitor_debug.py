"""
Stock Monitor Service Runner - Debug Version
Identifies which specific import is hanging
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Ensure venv packages are accessible
venv_site_packages = project_root / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

# Configure logging
from loguru import logger

log_file = project_root / "logs" / "stock_monitor_debug.log"
logger.remove()  # Remove default handler
logger.add(
    str(log_file),
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)
logger.add(sys.stderr, level="INFO")

logger.info("=" * 70)
logger.info("🔍 STOCK MONITOR DEBUG - IDENTIFYING IMPORT HANG")
logger.info("=" * 70)

# Test each import individually
try:
    logger.info("1. Importing time...")
    import time
    logger.info("✓ time")
    
    logger.info("2. Importing logging...")
    import logging
    logger.info("✓ logging")
    
    logger.info("3. Importing typing...")
    from typing import List, Dict, Optional, Any
    logger.info("✓ typing")
    
    logger.info("4. Importing datetime...")
    from datetime import datetime
    logger.info("✓ datetime")
    
    logger.info("5. Importing dataclasses...")
    from dataclasses import dataclass, asdict
    logger.info("✓ dataclasses")
    
    logger.info("6. Importing json...")
    import json
    logger.info("✓ json")
    
    logger.info("7. Importing os...")
    import os
    logger.info("✓ os")
    
    logger.info("8. Importing services.llm_helper...")
    from services.llm_helper import get_llm_helper, LLMServiceMixin
    logger.info("✓ services.llm_helper - SHOULD NOW WORK (lazy imports added)")
    
    logger.info("8b. Testing get_llm_helper function (no actual instantiation)...")
    logger.info(f"✓ get_llm_helper function available: {callable(get_llm_helper)}")
    logger.info("   NOTE: Not calling it yet - would trigger llm_request_manager import")
    
    logger.info("9. Importing services.alert_system...")
    from services.alert_system import get_alert_system
    logger.info("✓ services.alert_system")
    
    logger.info("10. Importing services.ai_confidence_scanner...")
    from services.ai_confidence_scanner import AIConfidenceScanner
    logger.info("✓ services.ai_confidence_scanner")
    
    logger.info("11. Importing config_stock_informational...")
    try:
        import config_stock_informational as cfg
        logger.info("✓ config_stock_informational")
    except ImportError as e:
        logger.warning(f"config_stock_informational not found: {e}")
    
    logger.info("12. Importing services.stock_informational_monitor (THE BIG ONE)...")
    from services.stock_informational_monitor import get_stock_informational_monitor
    logger.info("✓✓✓ services.stock_informational_monitor IMPORTED SUCCESSFULLY!")
    
    logger.info("=" * 70)
    logger.info("🎉 ALL IMPORTS SUCCESSFUL - NO HANG DETECTED")
    logger.info("=" * 70)
    
    # Try to actually initialize
    logger.info("Initializing monitor...")
    monitor = get_stock_informational_monitor()
    logger.info(f"✓ Monitor initialized with {len(monitor.watchlist)} symbols")
    
    logger.info("=" * 70)
    logger.info("🎉 MONITOR INITIALIZED - READY TO RUN")
    logger.info("=" * 70)
    
except Exception as e:
    logger.error(f"❌ FAILED AT IMPORT: {e}", exc_info=True)
    sys.exit(1)

logger.info("Debug complete - exiting")

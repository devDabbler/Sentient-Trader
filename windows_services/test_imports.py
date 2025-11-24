"""
Test imports one at a time to identify which is hanging
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

venv_site_packages = project_root / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

from loguru import logger

logger.info("=" * 70)
logger.info("TESTING IMPORTS")
logger.info("=" * 70)

try:
    logger.info("1. Testing yfinance...")
    import yfinance as yf
    logger.info("✓ yfinance imported")
    
    logger.info("2. Testing llm_helper...")
    from services.llm_helper import get_llm_helper
    logger.info("✓ llm_helper imported")
    
    logger.info("3. Testing alert_system...")
    from services.alert_system import get_alert_system
    logger.info("✓ alert_system imported")
    
    logger.info("4. Testing penny_stock_analyzer...")
    from services.penny_stock_analyzer import PennyStockAnalyzer
    logger.info("✓ penny_stock_analyzer imported")
    
    logger.info("5. Testing top_trades_scanner...")
    from services.top_trades_scanner import TopTradesScanner
    logger.info("✓ top_trades_scanner imported")
    
    logger.info("6. Testing ai_confidence_scanner...")
    from services.ai_confidence_scanner import AIConfidenceScanner
    logger.info("✓ ai_confidence_scanner imported")
    
    logger.info("7. Testing stock_informational_monitor...")
    from services.stock_informational_monitor import get_stock_informational_monitor
    logger.info("✓ stock_informational_monitor imported")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL IMPORTS SUCCESSFUL!")
    logger.info("=" * 70)
    
except Exception as e:
    logger.error(f"❌ Import failed: {e}", exc_info=True)
    sys.exit(1)

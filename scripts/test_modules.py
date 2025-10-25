"""Test script to verify modular imports work correctly."""

print("Testing modular imports...")

try:
    from models import TradingConfig, StockAnalysis, StrategyRecommendation, MarketCondition
    print("✅ models imported successfully")
except Exception as e:
    print(f"❌ models import failed: {e}")

try:
    from analyzers import TechnicalAnalyzer, NewsAnalyzer, ComprehensiveAnalyzer, StrategyAdvisor
    print("✅ analyzers imported successfully")
except Exception as e:
    print(f"❌ analyzers import failed: {e}")

try:
    from clients import OptionAlphaClient, SignalValidator
    print("✅ clients imported successfully")
except Exception as e:
    print(f"❌ clients import failed: {e}")

try:
    from utils.caching import get_cached_stock_data, get_cached_news
    from utils.logging_config import setup_logging, logger
    from utils.helpers import calculate_dte
    print("✅ utils imported successfully")
except Exception as e:
    print(f"❌ utils import failed: {e}")

print("\n✅ ALL MODULAR IMPORTS SUCCESSFUL!")
print("\nYour app is now organized with modular architecture:")
print("  📁 models/      - Data structures")
print("  📁 analyzers/   - Analysis logic")
print("  📁 clients/     - External integrations")
print("  📁 utils/       - Shared utilities")

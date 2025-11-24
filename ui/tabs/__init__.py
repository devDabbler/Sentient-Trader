"""
UI Tabs Package
Contains all refactored tab modules from app.py
"""

# Import all tab modules for easy access
try:
    from . import dashboard_tab
except ImportError:
    dashboard_tab = None

try:
    from . import scanner_tab
except ImportError:
    scanner_tab = None

try:
    from . import watchlist_tab
except ImportError:
    watchlist_tab = None

try:
    from . import strategy_advisor_tab
except ImportError:
    strategy_advisor_tab = None

try:
    from . import generate_signal_tab
except ImportError:
    generate_signal_tab = None

try:
    from . import signal_history_tab
except ImportError:
    signal_history_tab = None

try:
    from . import strategy_guide_tab
except ImportError:
    strategy_guide_tab = None

try:
    from . import tradier_tab
except ImportError:
    tradier_tab = None

try:
    from . import ibkr_tab
except ImportError:
    ibkr_tab = None

try:
    from . import scalping_tab
except ImportError:
    scalping_tab = None

try:
    from . import strategy_analyzer_tab
except ImportError:
    strategy_analyzer_tab = None

try:
    from . import autotrader_tab
except ImportError:
    autotrader_tab = None

try:
    from . import crypto_tab
except ImportError:
    crypto_tab = None

try:
    from . import dca_tab
except ImportError:
    dca_tab = None

try:
    from . import llm_usage_tab
except ImportError:
    llm_usage_tab = None

__all__ = [
    'dashboard_tab',
    'scanner_tab',
    'watchlist_tab',
    'strategy_advisor_tab',
    'generate_signal_tab',
    'signal_history_tab',
    'strategy_guide_tab',
    'tradier_tab',
    'ibkr_tab',
    'scalping_tab',
    'strategy_analyzer_tab',
    'autotrader_tab',
    'crypto_tab',
    'dca_tab',
    'llm_usage_tab'
]

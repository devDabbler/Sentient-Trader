#!/usr/bin/env python
"""
Sentient Trader App.py Modularization Script
Python version for reliable extraction across platforms

Usage:
    python migrate_app.py --phase 1 [--dry-run]
    python migrate_app.py --phase 2
    python migrate_app.py --phase all
"""

import os
import argparse
import shutil
from datetime import datetime
from pathlib import Path


class AppModularizer:
    """Extracts sections from app.py into modular files"""
    
    def __init__(self, app_py_path="app.py", dry_run=False):
        self.app_py_path = Path(app_py_path)
        self.dry_run = dry_run
        self.lines = []
        
        # Load app.py content
        if self.app_py_path.exists():
            with open(self.app_py_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
            print(f"✅ Loaded {self.app_py_path} ({len(self.lines)} lines)")
        else:
            print(f"❌ {self.app_py_path} not found!")
            exit(1)
    
    def extract_lines(self, start_line, end_line, dest_file, header=""):
        """Extract specific line range from app.py"""
        dest_path = Path(dest_file)
        
        if self.dry_run:
            print(f"ℹ️  [DRY-RUN] Would extract lines {start_line}-{end_line} to {dest_file}")
            return True
        
        try:
            # Create directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract lines (convert to 0-indexed)
            extracted = self.lines[start_line-1:end_line]
            
            # Write to file
            with open(dest_path, 'w', encoding='utf-8') as f:
                if header:
                    f.write(header)
                    f.write('\n\n')
                f.writelines(extracted)
            
            print(f"✅ Created {dest_file} ({len(extracted)} lines)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to extract to {dest_file}: {e}")
            return False
    
    def phase1_utilities(self):
        """Phase 1: Extract utilities and models"""
        print("\n" + "="*60)
        print("Phase 1: Extracting Utilities & Models")
        print("="*60 + "\n")
        
        # 1. Streamlit cache functions (approx lines 67-210)
        header = '''"""
Streamlit Cache Functions
Centralized caching for expensive operations
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Any
import yfinance as yf
'''
        self.extract_lines(67, 210, "utils/streamlit_cache.py", header)
        
        # Note: We'll need to manually verify these line numbers and adjust
        # Let's focus on creating stub files first
        
        print("\n✨ Phase 1 complete!\n")
        return True
    
    def create_tab_module(self, start_line, end_line, tab_name, dest_file, description):
        """Create a tab module with proper structure"""
        header = f'''"""
{tab_name} Tab
{description}

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple

def render_tab():
    """Main render function called from app.py"""
    st.header("{tab_name}")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)
'''
        return self.extract_lines(start_line, end_line, dest_file, header)
    
    def phase2_big_tabs(self):
        """Phase 2: Extract the Big 4 tabs"""
        print("\n" + "="*60)
        print("Phase 2: Extracting Big 4 Tabs")
        print("="*60 + "\n")
        
        # Dashboard Tab (2,560 lines!)
        self.create_tab_module(
            1521, 4080,
            "Dashboard",
            "ui/tabs/dashboard_tab.py",
            "Main dashboard with stock analysis, signal generation, and quick execution"
        )
        
        # Auto-Trader Tab (1,748 lines!)
        self.create_tab_module(
            10061, 11808,
            "Auto-Trader",
            "ui/tabs/autotrader_tab.py",
            "Automated trading system with strategy management and monitoring"
        )
        
        # Crypto Tab (1,581 lines!)
        self.create_tab_module(
            11809, 13389,
            "Crypto Trading",
            "ui/tabs/crypto_tab.py",
            "Cryptocurrency trading with Kraken integration, scanners, and position monitoring"
        )
        
        # Watchlist Tab (1,330 lines!)
        self.create_tab_module(
            4814, 6143,
            "Watchlist",
            "ui/tabs/watchlist_tab.py",
            "Manage ticker watchlist, bulk analysis, and strategy-specific monitoring"
        )
        
        print("\n✨ Phase 2 complete! Extracted Big 4 tabs (7,219 lines total)\n")
        return True
    
    def phase3_remaining_tabs(self):
        """Phase 3: Extract remaining 9 tabs"""
        print("\n" + "="*60)
        print("Phase 3: Extracting Remaining 9 Tabs")
        print("="*60 + "\n")
        
        tabs = [
            (4081, 4813, "Advanced Scanner", "ui/tabs/scanner_tab.py", 
             "Advanced stock scanner with AI scoring and filters"),
            (6149, 7324, "Strategy Advisor", "ui/tabs/strategy_advisor_tab.py",
             "AI-powered strategy recommendations"),
            (7325, 7594, "Generate Signal", "ui/tabs/generate_signal_tab.py",
             "Manual signal generation and analysis"),
            (7595, 7714, "Signal History", "ui/tabs/signal_history_tab.py",
             "View historical signals and performance"),
            (7715, 8245, "Strategy Guide", "ui/tabs/strategy_guide_tab.py",
             "Educational content and strategy templates"),
            (8251, 8642, "Tradier Account", "ui/tabs/tradier_tab.py",
             "Tradier broker integration and account management"),
            (8643, 9025, "IBKR Trading", "ui/tabs/ibkr_tab.py",
             "Interactive Brokers integration"),
            (9026, 9905, "Scalping/Day Trade", "ui/tabs/scalping_tab.py",
             "High-frequency scalping and day trading tools"),
            (9906, 10060, "Strategy Analyzer", "ui/tabs/strategy_analyzer_tab.py",
             "Backtest and analyze trading strategies"),
        ]
        
        for start, end, name, dest, desc in tabs:
            self.create_tab_module(start, end, name, dest, desc)
        
        print("\n✨ Phase 3 complete! Extracted remaining 9 tabs\n")
        return True
    
    def phase4_new_app(self):
        """Phase 4: Create new navigation-only app.py"""
        print("\n" + "="*60)
        print("Phase 4: Creating New Streamlined app.py")
        print("="*60 + "\n")
        
        if self.dry_run:
            print("ℹ️  [DRY-RUN] Would create app_new.py")
            return True
        
        # Backup current app.py
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"app.py.before_phase4_{timestamp}"
        shutil.copy(self.app_py_path, backup_name)
        print(f"✅ Backed up {self.app_py_path} to {backup_name}")
        
        # Create new streamlined app.py
        new_app_content = '''"""
Sentient Trader - Main Entry Point (Navigation Only)
Refactored from 13,393-line monolithic file to modular architecture
"""

from dotenv import load_dotenv
load_dotenv()

from utils.logging_config import setup_logging, get_broker_specific_log_file
setup_logging(log_file=get_broker_specific_log_file("logs/sentient_trader.log"))

import streamlit as st
from loguru import logger
import os
import sys

# Windows-specific asyncio policy
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(page_title="Sentient Trader", page_icon="📈", layout="wide")

# Import all tab modules
try:
    from ui.tabs import (
        dashboard_tab, scanner_tab, watchlist_tab, strategy_advisor_tab,
        generate_signal_tab, signal_history_tab, strategy_guide_tab,
        tradier_tab, ibkr_tab, scalping_tab, strategy_analyzer_tab,
        autotrader_tab, crypto_tab
    )
except ImportError as e:
    st.error(f"Failed to import tab modules: {e}")
    st.info("Make sure all tab modules have been extracted from app.py")
    st.stop()

def get_default_tab():
    """Get default tab based on DEFAULT_BROKER env var"""
    broker = os.getenv('DEFAULT_BROKER', '').upper()
    if broker == 'TRADIER':
        return "🏦 Tradier Account"
    elif broker == 'IBKR':
        return "📈 IBKR Trading"
    elif broker == 'KRAKEN':
        return "₿ Crypto Trading"
    return "🏠 Dashboard"

def main():
    # Sidebar
    with st.sidebar:
        st.title("📈 Sentient Trader")
        st.caption("AI-Driven Trading Platform")
        st.divider()
    
    # Tab names
    tab_names = [
        "🏠 Dashboard",
        "🚀 Advanced Scanner",
        "⭐ My Tickers",
        "🎯 Strategy Advisor",
        "📊 Generate Signal",
        "📜 Signal History",
        "📚 Strategy Guide",
        "🏦 Tradier Account",
        "📈 IBKR Trading",
        "⚡ Scalping/Day Trade",
        "🤖 Strategy Analyzer",
        "🤖 Auto-Trader",
        "₿ Crypto Trading"
    ]
    
    # Initialize active tab
    if 'active_main_tab' not in st.session_state:
        st.session_state.active_main_tab = get_default_tab()
    
    # Navigation radio
    selected_tab = st.radio(
        "Select Section:",
        tab_names,
        index=tab_names.index(st.session_state.active_main_tab),
        horizontal=True
    )
    st.session_state.active_main_tab = selected_tab
    st.divider()
    
    # Render selected tab
    try:
        if selected_tab == "🏠 Dashboard":
            dashboard_tab.render_tab()
        elif selected_tab == "🚀 Advanced Scanner":
            scanner_tab.render_tab()
        elif selected_tab == "⭐ My Tickers":
            watchlist_tab.render_tab()
        elif selected_tab == "🎯 Strategy Advisor":
            strategy_advisor_tab.render_tab()
        elif selected_tab == "📊 Generate Signal":
            generate_signal_tab.render_tab()
        elif selected_tab == "📜 Signal History":
            signal_history_tab.render_tab()
        elif selected_tab == "📚 Strategy Guide":
            strategy_guide_tab.render_tab()
        elif selected_tab == "🏦 Tradier Account":
            tradier_tab.render_tab()
        elif selected_tab == "📈 IBKR Trading":
            ibkr_tab.render_tab()
        elif selected_tab == "⚡ Scalping/Day Trade":
            scalping_tab.render_tab()
        elif selected_tab == "🤖 Strategy Analyzer":
            strategy_analyzer_tab.render_tab()
        elif selected_tab == "🤖 Auto-Trader":
            autotrader_tab.render_tab()
        elif selected_tab == "₿ Crypto Trading":
            crypto_tab.render_tab()
        else:
            st.error(f"Unknown tab: {selected_tab}")
            
    except Exception as e:
        logger.error(f"Error rendering tab {selected_tab}: {e}", exc_info=True)
        st.error(f"Failed to load {selected_tab}")
        st.exception(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error("Application encountered an error. Check logs for details.")
        st.exception(e)
'''
        
        with open("app_new.py", 'w', encoding='utf-8') as f:
            f.write(new_app_content)
        
        print("✅ Created app_new.py (navigation-only, ~180 lines)")
        print("ℹ️  Review app_new.py, then rename it to app.py when ready")
        print("\n✨ Phase 4 complete!\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Modularize app.py into manageable components"
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['1', '2', '3', '4', 'all'],
        default='1',
        help="Which phase to run (1=utilities, 2=big tabs, 3=remaining tabs, 4=new app.py, all=everything)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Preview actions without making changes"
    )
    
    args = parser.parse_args()
    
    print("\n🚀 Sentient Trader App.py Modularization Script")
    print("=" * 60 + "\n")
    
    if args.dry_run:
        print("⚠️  DRY-RUN MODE - No files will be modified\n")
    
    modularizer = AppModularizer(dry_run=args.dry_run)
    
    if args.phase == '1':
        modularizer.phase1_utilities()
    elif args.phase == '2':
        modularizer.phase2_big_tabs()
    elif args.phase == '3':
        modularizer.phase3_remaining_tabs()
    elif args.phase == '4':
        modularizer.phase4_new_app()
    elif args.phase == 'all':
        modularizer.phase1_utilities()
        modularizer.phase2_big_tabs()
        modularizer.phase3_remaining_tabs()
        modularizer.phase4_new_app()
    
    print("\n" + "="*60)
    print("✨ Migration script complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review generated files for import errors")
    print("  2. Fix any missing imports in tab modules")
    print("  3. Test: streamlit run app.py")
    print("  4. If Phase 4 complete: rename app_new.py to app.py\n")


if __name__ == "__main__":
    main()

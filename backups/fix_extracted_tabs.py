#!/usr/bin/env python
"""
Fix Extracted Tab Modules
Removes conditional checks and adds proper imports
"""

import re
from pathlib import Path

# Common imports needed by all tabs
COMMON_IMPORTS = '''
# Standard library imports
import sys
import os
import asyncio
import io
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from loguru import logger

# Windows-specific asyncio policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Local imports - Add as needed based on tab requirements
from dotenv import load_dotenv
load_dotenv()
'''

def fix_tab_file(tab_path: Path):
    """Fix a single tab file"""
    print(f"\n🔧 Fixing {tab_path.name}...")
    
    with open(tab_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into header and body
    lines = content.split('\n')
    
    # Find where the actual code starts (after the render_tab() function definition)
    render_tab_index = None
    for i, line in enumerate(lines):
        if 'def render_tab():' in line:
            render_tab_index = i
            break
    
    if render_tab_index is None:
        print(f"  ⚠️  Could not find render_tab() in {tab_path.name}")
        return False
    
    # Find the first if statement checking selected_main_tab or similar
    code_start_index = None
    for i in range(render_tab_index, min(render_tab_index + 20, len(lines))):
        if re.match(r'\s*if selected_main_tab ==', lines[i]):
            code_start_index = i
            break
        elif re.match(r'\s*st\.header\(', lines[i]) and i > render_tab_index + 5:
            # Found actual content start
            code_start_index = i
            break
    
    if code_start_index is None:
        # Try to find first meaningful line after render_tab()
        for i in range(render_tab_index + 1, min(render_tab_index + 30, len(lines))):
            if lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('"""'):
                code_start_index = i
                break
    
    if code_start_index is None:
        print(f"  ⚠️  Could not find code start in {tab_path.name}")
        return False
    
    # Get the body content (skip the if statement line if present)
    if re.match(r'\s*if selected_main_tab ==', lines[code_start_index]):
        body_lines = lines[code_start_index + 1:]  # Skip the if statement
    else:
        body_lines = lines[code_start_index:]
    
    # Remove one level of indentation from body
    dedented_body = []
    for line in body_lines:
        if line.startswith('    '):  # Has indentation
            dedented_body.append(line[4:])  # Remove 4 spaces
        else:
            dedented_body.append(line)
    
    # Build new file content
    # Extract the docstring from original
    docstring_end = None
    for i, line in enumerate(lines):
        if i > 0 and '"""' in line and '"""' in lines[0]:
            docstring_end = i
            break
    
    if docstring_end:
        header_lines = lines[:docstring_end + 1]
    else:
        header_lines = lines[:10]  # Fallback
    
    # Create new content
    new_content = '\n'.join(header_lines) + '\n\n' + COMMON_IMPORTS + '\n\n'
    new_content += 'def render_tab():\n'
    new_content += '    """Main render function called from app.py"""\n'
    new_content += '    \n'.join(['    ' + line for line in dedented_body])
    
    # Write back
    with open(tab_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  ✅ Fixed {tab_path.name}")
    return True

def main():
    print("🚀 Fixing Extracted Tab Modules")
    print("=" * 60)
    
    tabs_dir = Path("ui/tabs")
    if not tabs_dir.exists():
        print(f"❌ Directory {tabs_dir} not found!")
        return
    
    # Get all Python files except __init__.py
    tab_files = [f for f in tabs_dir.glob("*.py") if f.name != "__init__.py"]
    
    print(f"\nFound {len(tab_files)} tab files to fix\n")
    
    success_count = 0
    for tab_file in sorted(tab_files):
        if fix_tab_file(tab_file):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✨ Fixed {success_count}/{len(tab_files)} tab files")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review fixed tab files for any remaining issues")
    print("  2. Add tab-specific imports as needed")
    print("  3. Test: streamlit run app_new.py")

if __name__ == "__main__":
    main()

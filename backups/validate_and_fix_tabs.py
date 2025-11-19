#!/usr/bin/env python
"""
Validate and Fix Tab Files - Remove ALL bad indentation
"""

from pathlib import Path
import re

def fix_tab_file(tab_path: Path):
    """Fix a tab file by validating Python syntax"""
    print(f"\n{'='*60}")
    print(f"Processing: {tab_path.name}")
    print('='*60)
    
    with open(tab_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find render_tab() definition
    render_tab_idx = None
    for i, line in enumerate(lines):
        if 'def render_tab():' in line and not line.strip().startswith('#'):
            render_tab_idx = i
            print(f"Found render_tab() at line {i+1}")
            break
    
    if render_tab_idx is None:
        print(f"⚠️  No render_tab() found, skipping")
        return False
    
    # Build new file
    new_lines = []
    
    # Keep everything before render_tab() as-is
    for i in range(render_tab_idx + 1):
        new_lines.append(lines[i])
    
    # Find docstring
    docstring_end = render_tab_idx + 1
    if render_tab_idx + 1 < len(lines) and '"""' in lines[render_tab_idx + 1]:
        # Has docstring, find end
        for i in range(render_tab_idx + 2, min(render_tab_idx + 10, len(lines))):
            if '"""' in lines[i]:
                docstring_end = i
                break
        # Add docstring lines
        for i in range(render_tab_idx + 1, docstring_end + 1):
            new_lines.append(lines[i])
    
    # Process rest of the function
    print(f"Processing function body from line {docstring_end + 2}...")
    
    for i in range(docstring_end + 1, len(lines)):
        line = lines[i]
        
        # Completely strip and re-indent
        stripped = line.lstrip()
        
        if not stripped:
            # Empty line - keep it empty
            new_lines.append('')
        elif stripped.startswith('#'):
            # Comment - add 4 space indent
            new_lines.append('    ' + stripped)
        else:
            # Code line - add 4 space indent
            new_lines.append('    ' + stripped)
    
    # Write back
    new_content = '\n'.join(new_lines)
    
    with open(tab_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)
    
    print(f"✅ Fixed {tab_path.name}")
    
    # Validate Python syntax
    try:
        compile(new_content, str(tab_path), 'exec')
        print(f"✅ Python syntax valid!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error on line {e.lineno}: {e.msg}")
        print(f"   Text: {e.text}")
        return False

def main():
    tabs_dir = Path("ui/tabs")
    tab_files = [f for f in tabs_dir.glob("*.py") 
                 if f.name != "__init__.py" and f.name != "common_imports.py"]
    
    print("\n" + "="*60)
    print(f"VALIDATING AND FIXING {len(tab_files)} TAB FILES")
    print("="*60)
    
    success_count = 0
    failed = []
    
    for tab_file in sorted(tab_files):
        if fix_tab_file(tab_file):
            success_count += 1
        else:
            failed.append(tab_file.name)
    
    print("\n" + "="*60)
    print(f"RESULTS: {success_count}/{len(tab_files)} successful")
    print("="*60)
    
    if failed:
        print(f"\n⚠️  Failed files: {', '.join(failed)}")
    else:
        print("\n✅ All files validated and fixed!")

if __name__ == "__main__":
    main()

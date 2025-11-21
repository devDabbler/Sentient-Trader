#!/usr/bin/env python
"""
Simple Tab Fix - Remove conditional check and dedent by 4 spaces
"""

from pathlib import Path

def fix_tab(tab_path: Path):
    """Fix a single tab file"""
    print(f"Fixing {tab_path.name}...")
    
    with open(tab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find render_tab()
    render_tab_idx = None
    for i, line in enumerate(lines):
        if 'def render_tab():' in line:
            render_tab_idx = i
            break
    
    if render_tab_idx is None:
        print(f"  ⚠️  No render_tab() found")
        return False
    
    # Find the if statement line
    if_idx = None
    for i in range(render_tab_idx, min(render_tab_idx + 20, len(lines))):
        if 'if selected_main_tab ==' in lines[i]:
            if_idx = i
            break
    
    if if_idx is None:
        print(f"  ⚠️  No 'if selected_main_tab ==' found")
        return False
    
    # Build new file:
    # 1. Keep everything up to and including render_tab()
    # 2. Skip the if statement line
    # 3. Dedent everything after by 4 spaces
    
    new_lines = lines[:if_idx]  # Everything before the if statement
    
    # Process rest - dedent by 4 spaces
    for i in range(if_idx + 1, len(lines)):
        line = lines[i]
        if line.startswith('    '):
            # Remove 4 spaces
            new_lines.append(line[4:])
        else:
            # Keep line as-is (probably blank or comment)
            new_lines.append(line)
    
    # Write back
    with open(tab_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"  ✅ Fixed {tab_path.name}")
    return True

def main():
    tabs_dir = Path("ui/tabs")
    tab_files = [f for f in tabs_dir.glob("*.py") 
                 if f.name != "__init__.py" and f.name != "common_imports.py"]
    
    print(f"\n🔧 Fixing {len(tab_files))} tab files\n")
    
    for tab_file in sorted(tab_files):
        fix_tab(tab_file)
    
    print("\n✅ All tabs fixed!\n")

if __name__ == "__main__":
    main()

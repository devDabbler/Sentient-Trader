#!/usr/bin/env python
"""
Fix Tab Indentation - Properly indent all code inside render_tab()
"""

from pathlib import Path

def fix_tab_indentation(tab_path: Path):
    """Fix indentation for a tab file"""
    print(f"Fixing {tab_path.name}...")
    
    with open(tab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find render_tab() definition
    render_tab_idx = None
    for i, line in enumerate(lines):
        if 'def render_tab():' in line:
            render_tab_idx = i
            break
    
    if render_tab_idx is None:
        print(f"  ⚠️  No render_tab() found in {tab_path.name}")
        return False
    
    # Find the docstring end
    docstring_start = None
    docstring_end = None
    for i in range(render_tab_idx + 1, min(render_tab_idx + 5, len(lines))):
        if '"""' in lines[i]:
            if docstring_start is None:
                docstring_start = i
            else:
                docstring_end = i
                break
    
    if docstring_end is None:
        docstring_end = render_tab_idx + 1
    
    # Everything after docstring_end needs to be indented by 4 spaces
    fixed_lines = lines[:docstring_end + 1]
    
    for i in range(docstring_end + 1, len(lines)):
        line = lines[i]
        # Add 4-space indent if line has content
        if line.strip():
            fixed_lines.append('    ' + line)
        else:
            fixed_lines.append(line)
    
    # Write back
    with open(tab_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"  ✅ Fixed {tab_path.name}")
    return True

def main():
    tabs_dir = Path("ui/tabs")
    tab_files = [f for f in tabs_dir.glob("*.py") 
                 if f.name != "__init__.py" and f.name != "common_imports.py"]
    
    print(f"\n🔧 Fixing indentation in {len(tab_files)} tab files\n")
    
    for tab_file in sorted(tab_files):
        fix_tab_indentation(tab_file)
    
    print("\n✅ All tabs fixed!\n")

if __name__ == "__main__":
    main()

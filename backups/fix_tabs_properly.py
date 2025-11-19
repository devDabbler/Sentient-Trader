#!/usr/bin/env python
"""
Fix Tab Indentation PROPERLY
Remove excessive indentation after render_tab() definition
"""

from pathlib import Path

def fix_tab_file(tab_path: Path):
    """Fix indentation in tab file"""
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
        print(f"  ⚠️  No render_tab() found")
        return False
    
    # Process lines after render_tab()
    fixed_lines = []
    in_function = False
    
    for i, line in enumerate(lines):
        if i <= render_tab_idx:
            # Keep everything before render_tab() as-is
            fixed_lines.append(line)
        elif i == render_tab_idx + 1:
            # This should be the docstring - keep as-is
            fixed_lines.append(line)
        else:
            # After render_tab() and docstring
            if in_function or (i > render_tab_idx + 1 and lines[render_tab_idx + 1].strip().startswith('"""')):
                in_function = True
                
                # Remove excessive indentation
                # Lines should have exactly 4 spaces of indentation
                stripped = line.lstrip()
                if stripped:
                    # Add back 4 spaces
                    fixed_lines.append('    ' + stripped)
                else:
                    # Keep blank lines
                    fixed_lines.append(line)
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
        fix_tab_file(tab_file)
    
    print("\n✅ All tabs fixed!")
    print("\nClearing cache and retrying...")

if __name__ == "__main__":
    main()

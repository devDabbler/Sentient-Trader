#!/usr/bin/env python3
"""Script to remove orphaned penny stock code from app.py"""

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep lines before the orphaned code (lines 0-3496) and after it (lines 3851+)
# Python uses 0-indexed, so line 3496 is index 3495, line 3851 is index 3850
cleaned_lines = lines[:3496] + lines[3851:]

# Write the cleaned file
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print(f"✅ Removed {3851 - 3496} orphaned lines (lines 3497-3851)")
print(f"✅ File now has {len(cleaned_lines)} lines (was {len(lines)} lines)")
print(f"✅ app.py has been cleaned!")

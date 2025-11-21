"""
Fix malformed f-strings in logging statements across the codebase
"""
import re
from pathlib import Path
from typing import Tuple, List
from loguru import logger

def fix_file(file_path: Path) -> Tuple[int, List[str]]:
    """Fix logging syntax issues in a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixes = []
        
        # Pattern 1: Fix extra ) before } in f-strings
        # Example: f"{something)}" -> f"{something}"
        pattern1 = r'(\{[^{}]*)\)\}'
        matches = re.findall(pattern1, content)
        if matches:
            content = re.sub(pattern1, r'\1}', content)
            fixes.append(f"Fixed {len(matches)} extra ) before }} in f-strings")
        
        # Pattern 2: Fix missing closing quote in .encode()
        # Example: {something.encode('utf-8'} -> {something.encode('utf-8')}
        pattern2 = r"(\{[^{}]*\.encode\(['\"][^'\"]*['\"])(\})"
        matches = re.findall(pattern2, content)
        if matches:
            content = re.sub(pattern2, r'\1)\2', content)
            fixes.append(f"Fixed {len(matches)} missing ) in .encode()")
        
        # Pattern 3: Fix empty placeholder ${}
        # Example: f"${} {value}" -> f"${value}"
        pattern3 = r'\$\{\}\s+'
        matches = re.findall(pattern3, content)
        if matches:
            content = re.sub(pattern3, r'$', content)
            fixes.append(f"Fixed {len(matches)} empty ${{}} placeholders")
        
        # Pattern 4: Fix double ))}} patterns
        # Example: {str(e))}" -> {str(e)}"
        pattern4 = r'(\{[^{}]*\([^)]*)\)\)\}'
        matches = re.findall(pattern4, content)
        if matches:
            content = re.sub(pattern4, r'\1)}', content)
            fixes.append(f"Fixed {len(matches)} double )) before }}")
        
        # Pattern 5: Fix unclosed ( in f-strings
        # Example: logger.info(f"text {something" -> needs closing )
        # This is trickier, let's handle specific cases
        
        # Write back if changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return len(fixes), fixes
        
        return 0, []
        
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return 0, []

def main():
    """Fix all Python files in the project"""
    root = Path(r"c:\Users\seaso\Sentient Trader")
    
    # Files to fix based on error report
    priority_files = [
        "analyzers/news.py",
        "clients/kraken_client.py",
        "services/discord_trade_approval.py",
        "ui/daily_scanner_ui.py",
        "ui/tabs/dashboard_tab.py"
    ]
    
    total_fixes = 0
    files_fixed = 0
    
    logger.info("=" * 80)
    logger.info("FIXING LOGGING SYNTAX ISSUES")
    logger.info("=" * 80)
    
    for rel_path in priority_files:
        file_path = root / rel_path
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        logger.info(f"\nProcessing: {rel_path}")
        fix_count, fixes = fix_file(file_path)
        
        if fix_count > 0:
            files_fixed += 1
            total_fixes += fix_count
            logger.success(f"  ✅ Fixed {fix_count} issues:")
            for fix in fixes:
                logger.info(f"     • {fix}")
        else:
            logger.info("  ✓ No issues found")
    
    logger.info("\n" + "=" * 80)
    logger.success(f"COMPLETE: Fixed {total_fixes} issues across {files_fixed} files")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

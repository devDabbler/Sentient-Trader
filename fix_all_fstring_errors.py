"""
Comprehensive f-string syntax fixer for logging statements
Handles all patterns of unclosed parentheses and malformed f-strings
"""
import re
from pathlib import Path
from typing import List
from loguru import logger

def fix_unclosed_parens_in_fstrings(content: str) -> tuple:
    """Fix unclosed parentheses in f-strings"""
    fixes = []
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        original_line = line
        
        # Pattern: logger.X(f"...{something.get('x', y}") -> missing closing )
        # Match f-strings with unbalanced parentheses before the closing }
        if 'logger.' in line and 'f"' in line:
            # Find all f-string sections
            fstring_pattern = r'f"([^"]*)"'
            matches = list(re.finditer(fstring_pattern, line))
            
            for match in reversed(matches):  # Process from end to avoid index issues
                fstring_content = match.group(1)
                start_pos = match.start()
                end_pos = match.end()
                
                # Check for {...} placeholders
                placeholder_pattern = r'\{([^{}]+)\}'
                placeholders = list(re.finditer(placeholder_pattern, fstring_content))
                
                for ph in placeholders:
                    placeholder_text = ph.group(1)
                    
                    # Count parentheses
                    open_parens = placeholder_text.count('(')
                    close_parens = placeholder_text.count(')')
                    
                    if open_parens > close_parens:
                        # Add missing closing parens
                        missing = open_parens - close_parens
                        fixed_placeholder = placeholder_text + ')' * missing
                        
                        # Replace in the line
                        old_pattern = '{' + re.escape(placeholder_text) + '}'
                        new_pattern = '{' + fixed_placeholder + '}'
                        line = re.sub(old_pattern, new_pattern, line, count=1)
                        
                        fixes.append(f"Line {i+1}: Added {missing} closing paren(s)")
                        modified = True
        
        lines[i] = line
    
    if modified:
        return '\n'.join(lines), fixes
    return content, []

def fix_file(file_path: Path) -> tuple:
    """Fix all f-string syntax issues in a file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        all_fixes = []
        
        # Apply all fix patterns
        content, fixes = fix_unclosed_parens_in_fstrings(content)
        all_fixes.extend(fixes)
        
        # Write back if changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return len(all_fixes), all_fixes
        
        return 0, []
        
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {e}")
        return 0, []

def main():
    """Fix all Python files"""
    root = Path(r"c:\Users\seaso\Sentient Trader")
    
    # Focus on problem files first
    priority_files = [
        "ui/tabs/dashboard_tab.py",
        "analyzers/news.py",
        "clients/kraken_client.py",
        "ui/daily_scanner_ui.py",
        "services/discord_trade_approval.py"
    ]
    
    logger.info("=" * 80)
    logger.info("FIXING UNCLOSED PARENTHESES IN F-STRINGS")
    logger.info("=" * 80)
    
    total_fixes = 0
    files_fixed = 0
    
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
            logger.success(f"  ✅ Fixed {fix_count} issues")
            # Show first 5 fixes
            for fix in fixes[:5]:
                logger.info(f"     • {fix}")
            if len(fixes) > 5:
                logger.info(f"     • ... and {len(fixes)-5} more")
        else:
            logger.info("  ✓ No issues found")
    
    logger.info("\n" + "=" * 80)
    logger.success(f"COMPLETE: Fixed {total_fixes} issues across {files_fixed} files")
    logger.info("=" * 80)
    logger.info("\n🎯 Clear cache and restart: streamlit run app.py")

if __name__ == "__main__":
    main()

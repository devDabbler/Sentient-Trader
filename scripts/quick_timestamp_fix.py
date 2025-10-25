"""
Quick Timestamp Fix - Run this to fix your ticker timestamps immediately
"""

import sqlite3
from datetime import datetime, timezone, timedelta

DB_PATH = "data/tickers.db"

print("🔧 Quick Timestamp Fix")
print("="*60)

# First, let's see what we have
with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, date_added FROM saved_tickers ORDER BY ticker")
    tickers = cursor.fetchall()
    
    if not tickers:
        print("❌ No tickers found in database.")
        exit()
    
    print(f"\n📊 Found {len(tickers)} tickers")
    print("\nCurrent timestamps:")
    for ticker, date_added in tickers[:5]:  # Show first 5
        try:
            dt = datetime.fromisoformat(date_added).replace(tzinfo=timezone.utc)
            local = dt.astimezone().strftime('%B %d, %Y at %I:%M %p')
            print(f"  {ticker}: {local}")
        except:
            print(f"  {ticker}: {date_added}")
    
    if len(tickers) > 5:
        print(f"  ... and {len(tickers) - 5} more")

print("\n" + "="*60)
print("These timestamps will be reset to October 22, 2025 at 8:45 AM")
print("(matching the date shown in your screenshot)")
print("="*60)

response = input("\nProceed with fix? (yes/no): ").strip().lower()

if response != "yes":
    print("❌ Cancelled.")
    exit()

# Set to the date from your screenshot
target_date = datetime(2025, 10, 22, 8, 45, 0, tzinfo=timezone.utc)
now = datetime.now(timezone.utc)

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    
    # Update all tickers
    cursor.execute("""
        UPDATE saved_tickers 
        SET date_added = ?, last_accessed = ?
    """, (target_date.isoformat(), now.isoformat()))
    
    updated = cursor.rowcount
    conn.commit()

print(f"\n✅ Updated {updated} tickers!")
print(f"   date_added: October 22, 2025 at 8:45 AM")
print(f"   last_accessed: {now.astimezone().strftime('%B %d, %Y at %I:%M %p')}")
print("\n💡 Now restart your Streamlit app to see the changes!")

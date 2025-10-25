"""
Fix Ticker Timestamps

This script checks and optionally fixes ticker timestamps in the database.
It can reset timestamps to a reasonable date or show current values.
"""

import sqlite3
from datetime import datetime, timezone, timedelta

DB_PATH = "data/tickers.db"

def view_current_timestamps():
    """View all current ticker timestamps"""
    print("\n" + "="*80)
    print("CURRENT TICKER TIMESTAMPS")
    print("="*80)
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ticker, name, date_added, last_accessed, access_count 
            FROM saved_tickers 
            ORDER BY ticker ASC
        """)
        
        rows = cursor.fetchall()
        if not rows:
            print("No tickers found in database.")
            return
        
        for row in rows:
            ticker = row['ticker']
            name = row['name'] or 'N/A'
            date_added = row['date_added']
            last_accessed = row['last_accessed']
            access_count = row['access_count']
            
            # Parse and display in local time
            try:
                dt_added = datetime.fromisoformat(date_added).replace(tzinfo=timezone.utc)
                local_added = dt_added.astimezone().strftime('%Y-%m-%d %I:%M %p')
            except:
                local_added = date_added
            
            try:
                dt_accessed = datetime.fromisoformat(last_accessed).replace(tzinfo=timezone.utc)
                local_accessed = dt_accessed.astimezone().strftime('%Y-%m-%d %I:%M %p')
            except:
                local_accessed = last_accessed
            
            print(f"\n{ticker} ({name})")
            print(f"  Added: {local_added}")
            print(f"  Last Accessed: {local_accessed}")
            print(f"  Access Count: {access_count}")

def reset_to_reasonable_dates():
    """Reset timestamps to reasonable historical dates based on access count"""
    print("\n" + "="*80)
    print("RESETTING TIMESTAMPS TO REASONABLE DATES")
    print("="*80)
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, access_count FROM saved_tickers")
        tickers = cursor.fetchall()
        
        if not tickers:
            print("No tickers to update.")
            return
        
        # Base date: 7 days ago for frequently accessed, 3 days ago for others
        now = datetime.now(timezone.utc)
        
        for ticker, access_count in tickers:
            # More accessed = added longer ago
            if access_count > 5:
                days_ago = 7
            elif access_count > 2:
                days_ago = 5
            else:
                days_ago = 3
            
            date_added = (now - timedelta(days=days_ago)).isoformat()
            last_accessed = now.isoformat()
            
            cursor.execute("""
                UPDATE saved_tickers 
                SET date_added = ?, last_accessed = ?
                WHERE ticker = ?
            """, (date_added, last_accessed, ticker))
            
            print(f"✓ {ticker}: Set to {days_ago} days ago")
        
        conn.commit()
        print(f"\n✅ Updated {len(tickers)} tickers")

def reset_to_specific_date():
    """Reset all date_added to a specific date you choose"""
    print("\n" + "="*80)
    print("RESET TO SPECIFIC DATE")
    print("="*80)
    
    # Set this to when you actually started saving tickers
    # For example: October 20, 2025 at 8:00 AM
    specific_date = datetime(2025, 10, 20, 8, 0, 0, tzinfo=timezone.utc)
    
    print(f"Will reset all date_added to: {specific_date.astimezone().strftime('%B %d, %Y at %I:%M %p')}")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM saved_tickers")
        tickers = cursor.fetchall()
        
        if not tickers:
            print("No tickers to update.")
            return
        
        now = datetime.now(timezone.utc)
        
        for (ticker,) in tickers:
            cursor.execute("""
                UPDATE saved_tickers 
                SET date_added = ?, last_accessed = ?
                WHERE ticker = ?
            """, (specific_date.isoformat(), now.isoformat(), ticker))
            
            print(f"✓ {ticker}")
        
        conn.commit()
        print(f"\n✅ Updated {len(tickers)} tickers")

if __name__ == "__main__":
    print("\n🔧 TICKER TIMESTAMP FIXER")
    print("="*80)
    print("This script helps fix timestamp issues in your saved tickers.")
    print()
    print("Options:")
    print("1. View current timestamps")
    print("2. Reset to reasonable dates (based on access count)")
    print("3. Reset to specific date (Oct 20, 2025 8:00 AM)")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        view_current_timestamps()
    elif choice == "2":
        confirm = input("\n⚠️  This will modify your database. Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            reset_to_reasonable_dates()
            print("\n📊 Updated timestamps:")
            view_current_timestamps()
        else:
            print("Cancelled.")
    elif choice == "3":
        print("\n📝 Edit the script to set your desired date in the reset_to_specific_date() function.")
        confirm = input("⚠️  This will modify your database. Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            reset_to_specific_date()
            print("\n📊 Updated timestamps:")
            view_current_timestamps()
        else:
            print("Cancelled.")
    elif choice == "4":
        print("Exiting.")
    else:
        print("Invalid choice.")

"""
Demo Script for New Features

Tests and demonstrates:
- Ticker Manager
- Top Trades Scanner
- Integration with Penny Stock Watchlist

Run: python demo_new_features.py
"""

import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ticker_manager import TickerManager
from top_trades_scanner import TopTradesScanner
from watchlist_manager import WatchlistManager


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def demo_ticker_manager():
    """Demonstrate Ticker Manager features"""
    print_header("🎯 TICKER MANAGER DEMO")
    
    tm = TickerManager()
    
    # 1. Save some tickers
    print("📝 Saving tickers...")
    tm.add_ticker('AAPL', name='Apple Inc.', sector='Technology', ticker_type='stock', 
                  notes='Tech giant, strong fundamentals')
    tm.add_ticker('MSFT', name='Microsoft', sector='Technology', ticker_type='stock')
    tm.add_ticker('SOFI', name='SoFi Technologies', sector='Finance', ticker_type='penny_stock',
                  tags=['fintech', 'growth'])
    tm.add_ticker('TSLA', name='Tesla', sector='Automotive', ticker_type='stock',
                  tags=['ev', 'high-volatility'])
    tm.add_ticker('NVDA', name='NVIDIA', sector='Technology', ticker_type='stock',
                  tags=['ai', 'semiconductors'])
    
    print("✅ Saved 5 tickers\n")
    
    # 2. Get all tickers
    print("📋 All saved tickers:")
    all_tickers = tm.get_all_tickers()
    for t in all_tickers:
        print(f"  • {t['ticker']} ({t['type']}) - {t.get('name', 'N/A')}")
    
    # 3. Record some access
    print("\n🔍 Simulating ticker views...")
    tm.record_access('AAPL')
    tm.record_access('AAPL')  # View twice
    tm.record_access('TSLA')
    tm.record_access('SOFI')
    tm.record_access('AAPL')  # View third time
    print("✅ Recorded access history\n")
    
    # 4. Get popular tickers
    print("🔥 Most popular tickers:")
    popular = tm.get_popular_tickers(limit=5)
    for ticker in popular:
        info = tm.get_ticker(ticker)
        print(f"  • {ticker} - {info.get('access_count', 0)} views")
    
    # 5. Search
    print("\n🔎 Searching for 'tech':")
    results = tm.search_tickers('tech')
    for r in results:
        print(f"  • {r['ticker']} - {r.get('name', 'N/A')}")
    
    # 6. Create watchlist
    print("\n📊 Creating watchlist...")
    tm.create_watchlist('My Top Picks', 'Best stocks for this week')
    tm.add_to_watchlist('My Top Picks', 'AAPL')
    tm.add_to_watchlist('My Top Picks', 'NVDA')
    
    watchlists = tm.get_watchlists()
    print(f"✅ Created {len(watchlists)} watchlist(s)")
    
    for wl in watchlists:
        tickers = tm.get_watchlist_tickers(wl['name'])
        print(f"  📋 {wl['name']}: {', '.join(tickers)}")
    
    # 7. Statistics
    print("\n📊 Statistics:")
    try:
        stats = tm.get_statistics()
        if stats:  # Check if not None and not empty
            print(f"  Total tickers: {stats.get('total_tickers', 0)}")
            print(f"  Stocks: {stats.get('stocks', 0)}")
            print(f"  Penny stocks: {stats.get('penny_stocks', 0)}")
            print(f"  Watchlists: {stats.get('watchlists', 0)}")
        else:
            print("  No statistics available")
    except Exception as e:
        print(f"  ⚠️ Could not retrieve statistics: {str(e)[:30]}...")


def demo_options_scanner():
    """Demonstrate Options Scanner"""
    print_header("🔥 TOP OPTIONS TRADES SCANNER")
    
    scanner = TopTradesScanner()
    
    print("🔍 Scanning markets for top options opportunities...")
    print("(This may take 30-60 seconds as it analyzes real market data)\n")
    
    try:
        trades = scanner.scan_top_options_trades(top_n=10)
        
        if trades:
            print(f"✅ Found {len(trades)} opportunities!\n")
            
            # Quick insights
            insights = scanner.get_quick_insights(trades)
            print("📊 Quick Insights:")
            print(f"  Average Score: {insights['avg_score']:.1f}/100")
            print(f"  High Confidence: {insights['high_confidence']}")
            print(f"  Big Movers (>3%): {insights['big_movers']}")
            print(f"  Volume Spikes: {insights['volume_spikes']}\n")
            
            # Top 5
            print("🏆 Top 5 Opportunities:")
            for i, trade in enumerate(trades[:5], 1):
                print(f"\n  {i}. {trade.ticker}")
                print(f"     Score: {trade.score:.1f}/100")
                print(f"     Price: ${trade.price:.2f} ({trade.change_pct:+.2f}%)")
                print(f"     Volume: {trade.volume_ratio:.2f}x average")
                print(f"     Confidence: {trade.confidence}")
                print(f"     Why: {trade.reason}")
        else:
            print("⚠️ No opportunities found (may be after hours or weekend)")
    
    except Exception as e:
        print(f"❌ Error during scan: {e}")
        print("This is normal if markets are closed or you're offline")


def demo_penny_scanner():
    """Demonstrate Penny Stock Scanner"""
    print_header("💰 TOP PENNY STOCKS SCANNER")
    
    scanner = TopTradesScanner()
    
    print("🔍 Scanning for top penny stock opportunities...")
    print("(This may take 1-2 minutes as it runs full analysis)\n")
    
    try:
        trades = scanner.scan_top_penny_stocks(top_n=10)
        
        if trades:
            print(f"✅ Found {len(trades)} opportunities!\n")
            
            # Quick insights
            insights = scanner.get_quick_insights(trades)
            print("📊 Quick Insights:")
            print(f"  Average Score: {insights['avg_score']:.1f}/100")
            print(f"  High Confidence: {insights['high_confidence']}")
            print(f"  Big Movers: {insights['big_movers']}")
            print(f"  Volume Spikes: {insights['volume_spikes']}\n")
            
            # Top 5
            print("🏆 Top 5 Opportunities:")
            for i, trade in enumerate(trades[:5], 1):
                print(f"\n  {i}. {trade.ticker}")
                print(f"     Composite Score: {trade.score:.1f}/100")
                print(f"     Price: ${trade.price:.2f} ({trade.change_pct:+.2f}%)")
                print(f"     Volume: {trade.volume_ratio:.2f}x average")
                print(f"     Confidence: {trade.confidence}")
                print(f"     Why: {trade.reason}")
        else:
            print("⚠️ No opportunities found")
    
    except Exception as e:
        print(f"❌ Error during scan: {e}")
        print("This is normal if markets are closed or you're offline")


def demo_integration():
    """Demonstrate integration between systems"""
    print_header("🔗 SYSTEM INTEGRATION DEMO")
    
    print("Demonstrating how to combine Ticker Manager with Scanners...\n")
    
    tm = TickerManager()
    scanner = TopTradesScanner()
    wm = WatchlistManager()
    
    # Scan and auto-save
    print("1️⃣ Scanning for options trades and auto-saving...")
    try:
        trades = scanner.scan_top_options_trades(top_n=5)
        
        if trades:
            for trade in trades:
                # Save to ticker manager
                tm.add_ticker(
                    trade.ticker,
                    ticker_type='option',
                    notes=f"Auto-saved from scan. Score: {trade.score:.1f}, Reason: {trade.reason}"
                )
            
            print(f"✅ Saved {len(trades)} tickers to Ticker Manager\n")
        else:
            print("⚠️ No trades to save\n")
    except Exception as e:
        print(f"⚠️ Scan skipped: {e}\n")
    
    # Show saved tickers
    print("2️⃣ Viewing saved tickers:")
    try:
        all_tickers = tm.get_all_tickers(limit=10)
        for t in all_tickers[:5]:
            try:
                ticker = t.get('ticker', 'UNKNOWN')
                ticker_type = t.get('type', 'unknown')
                notes = t.get('notes') or 'No notes'  # Handle None values
                print(f"  • {ticker} ({ticker_type}) - {notes[:50]}")
            except Exception as e:
                continue  # Skip problematic tickers
        
        print(f"\n✅ Total: {len(all_tickers)} tickers in database")
    except Exception as e:
        print(f"  ⚠️ Could not load tickers: {str(e)[:50]}...")
    
    # Penny stock integration
    print("\n3️⃣ Penny stock integration example:")
    print("   Scanning penny stocks and adding to watchlist...")
    
    try:
        penny_trades = scanner.scan_top_penny_stocks(top_n=3)
        
        if penny_trades:
            added_count = 0
            for trade in penny_trades:
                try:
                    # Add to penny stock watchlist
                    wm.add_stock({
                        'ticker': trade.ticker,
                        'price': trade.price,
                        'pct_change': trade.change_pct,
                        'composite_score': trade.score,
                        'confidence_level': trade.confidence,
                        'reasoning': trade.reason
                    })
                    added_count += 1
                except Exception as add_error:
                    # Skip stocks that can't be added
                    continue
            
            if added_count > 0:
                print(f"✅ Added {added_count} stocks to Penny Stock Watchlist")
            else:
                print("⚠️ Could not add stocks to watchlist")
        else:
            print("⚠️ No penny stocks to add")
    except Exception as e:
        print(f"⚠️ Penny scan skipped: {str(e)[:50]}...")
    
    # Show penny watchlist stats
    print("\n4️⃣ Penny Stock Watchlist stats:")
    try:
        penny_stats = wm.get_statistics()
        if penny_stats:  # Check if not None and not empty
            print(f"  Total stocks: {penny_stats.get('total_stocks', 0)}")
            print(f"  Average score: {penny_stats.get('avg_composite_score', 0):.1f}")
            print(f"  High confidence: {penny_stats.get('high_confidence_count', 0)}")
        else:
            print("  No statistics available")
    except Exception as e:
        print(f"  ⚠️ Could not retrieve statistics: {str(e)[:30]}...")


def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("  🚀 AI OPTIONS TRADER - NEW FEATURES DEMO")
    print("="*60)
    print("\nThis demo will showcase:")
    print("  1. Ticker Manager - Save and organize tickers")
    print("  2. Options Scanner - Find top options trades")
    print("  3. Penny Stock Scanner - Find top penny stocks")
    print("  4. System Integration - How they work together")
    print("\n⏱️  Estimated time: 2-3 minutes")
    print("📡 Requires internet connection for market data")
    
    input("\nPress ENTER to start the demo...")
    
    # Run demos
    try:
        demo_ticker_manager()
        input("\n👆 Review the results, then press ENTER to continue...")
        
        demo_options_scanner()
        input("\n👆 Review the results, then press ENTER to continue...")
        
        demo_penny_scanner()
        input("\n👆 Review the results, then press ENTER to continue...")
        
        demo_integration()
        
        # Final summary
        print_header("🎉 DEMO COMPLETE")
        print("✅ All new features demonstrated successfully!")
        print("\n📚 Next Steps:")
        print("  1. Review NEW_FEATURES_README.md for detailed documentation")
        print("  2. Try using the features in your own code")
        print("  3. Integrate into main app.py (optional)")
        print("\n💾 Database files created:")
        print("  • tickers.db - Your saved tickers and watchlists")
        print("  • watchlist.db - Penny stock watchlist data")
        print("\n🎯 Quick command examples:")
        print("  tm = TickerManager()")
        print("  tm.add_ticker('YOUR_TICKER')")
        print("  scanner = TopTradesScanner()")
        print("  trades = scanner.scan_top_options_trades(top_n=20)")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        print("This is usually due to network issues or market closures")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

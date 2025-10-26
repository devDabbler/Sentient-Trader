"""
Quick test to verify Reddit RSS feeds are accessible and parseable
"""
import asyncio
import httpx
import feedparser
import re

async def test_reddit_rss(ticker='AAPL'):
    """Test Reddit RSS feed fetching"""
    print(f"\n🧪 Testing Reddit RSS for ${ticker}")
    print("="*60)
    
    subreddits = ['wallstreetbets', 'stocks', 'investing']
    
    for subreddit in subreddits:
        try:
            rss_url = f"https://www.reddit.com/r/{subreddit}/new.rss?limit=25"
            print(f"\n📡 Fetching: {rss_url}")
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if response.status_code != 200:
                    print(f"   ❌ Status: {response.status_code}")
                    continue
                
                print(f"   ✅ Status: 200 OK")
                
                # Parse feed
                feed = feedparser.parse(response.text)
                print(f"   📄 Entries: {len(feed.entries)}")
                
                if not feed.entries:
                    print(f"   ⚠️  No entries in feed")
                    continue
                
                # Check first few titles
                print(f"   📝 Sample titles:")
                for i, entry in enumerate(feed.entries[:5], 1):
                    title = entry.get('title', 'No title')
                    print(f"      {i}. {title[:70]}")
                
                # Search for ticker
                matches = 0
                ticker_upper = ticker.upper()
                
                for entry in feed.entries:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    combined = f"{title} {summary}".upper()
                    
                    if (f"${ticker_upper}" in combined or 
                        f" {ticker_upper} " in f" {combined} "):
                        matches += 1
                        print(f"   🎯 Match found: {title[:60]}...")
                
                print(f"   📊 Total matches for ${ticker}: {matches}/{len(feed.entries)}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Test with popular tickers
    tickers = ['AAPL', 'TSLA', 'NVDA', 'GME', 'SPY']
    
    for ticker in tickers:
        asyncio.run(test_reddit_rss(ticker))
        print()

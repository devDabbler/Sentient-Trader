"""
Test RSS Feed Speed
Shows how fast Reddit RSS polling is vs scraping
"""

import asyncio
import time
import feedparser
import httpx
import re

async def test_rss_feeds():
    """Test RSS feed speed and data quality"""
    
    print("=" * 70)
    print("📡 Reddit RSS Feed Speed Test")
    print("=" * 70)
    
    subreddits = ['wallstreetbets', 'stocks', 'investing']
    ticker = 'AAPL'  # Test ticker
    
    print(f"\n🎯 Searching for: ${ticker}")
    print(f"📊 Subreddits: {', '.join(subreddits)}")
    print(f"⚡ Method: RSS feeds (no auth, no rate limits)")
    print("\n" + "=" * 70)
    
    total_start = time.time()
    total_posts = 0
    total_matches = 0
    
    # Compile ticker regex for fast matching
    ticker_pattern = re.compile(
        rf'\b(?:\${ticker.upper()}|{ticker.upper()})\b',
        re.IGNORECASE
    )
    
    for subreddit in subreddits:
        sub_start = time.time()
        
        try:
            # Poll RSS feed
            rss_url = f"https://www.reddit.com/r/{subreddit}/new.rss?limit=25"
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
                
                if response.status_code != 200:
                    print(f"❌ r/{subreddit}: HTTP {response.status_code}")
                    continue
                
                # Parse RSS
                feed = feedparser.parse(response.text)
                
                if not feed.entries:
                    print(f"⚠️  r/{subreddit}: No entries")
                    continue
                
                # Filter for ticker mentions
                matches = []
                for entry in feed.entries:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    combined = f"{title} {summary}"
                    
                    if ticker_pattern.search(combined):
                        matches.append({
                            'title': title[:80],
                            'published': entry.get('published', 'N/A'),
                            'link': entry.get('link', '')
                        })
                
                sub_elapsed = time.time() - sub_start
                total_posts += len(feed.entries)
                total_matches += len(matches)
                
                print(f"\nr/{subreddit}:")
                print(f"  ⏱️  Time: {sub_elapsed:.2f}s")
                print(f"  📄 Posts scanned: {len(feed.entries)}")
                print(f"  ✅ Ticker matches: {len(matches)}")
                
                if matches:
                    print(f"  📝 Sample matches:")
                    for i, match in enumerate(matches[:3], 1):
                        print(f"     {i}. {match['title']}...")
                
        except Exception as e:
            print(f"❌ r/{subreddit}: {e}")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("📊 Summary:")
    print("=" * 70)
    print(f"  ⏱️  Total time: {total_elapsed:.2f}s")
    print(f"  📄 Total posts scanned: {total_posts}")
    print(f"  ✅ Ticker mentions found: {total_matches}")
    print(f"  ⚡ Speed: {total_posts/total_elapsed:.1f} posts/second")
    
    if total_elapsed < 2:
        print(f"\n  🚀 EXCELLENT! Under 2 seconds")
    elif total_elapsed < 3:
        print(f"\n  ✅ GOOD! Under 3 seconds")
    else:
        print(f"\n  ⚠️  Slower than expected")
    
    print("\n" + "=" * 70)
    print("💡 Key Advantages of RSS:")
    print("=" * 70)
    print("  ✅ No authentication required")
    print("  ✅ No rate limits")
    print("  ✅ Lightweight XML format (~10-50KB per feed)")
    print("  ✅ Fast regex filtering")
    print("  ✅ Instant results (no browser, no JavaScript)")
    print("  ✅ Can poll every few seconds without penalty")

if __name__ == "__main__":
    asyncio.run(test_rss_feeds())

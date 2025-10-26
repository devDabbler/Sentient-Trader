"""
Test Speed Optimizations
Verifies Crawl4ai CSS selector and caching optimizations
"""

import asyncio
import time
from services.social_sentiment_analyzer import SocialSentimentAnalyzer

print("=" * 70)
print("⚡ Speed Optimization Test - Crawl4ai CSS Selectors + Caching")
print("=" * 70)

async def test_speed():
    """Test optimized speed with CSS selectors and caching"""
    analyzer = SocialSentimentAnalyzer()
    
    print("\n🔧 Optimizations Applied:")
    print("   ✅ CSS selectors instead of link preview crawling")
    print("   ✅ CacheMode.ENABLED for repeat requests")
    print("   ✅ Reduced timeouts (10s Reddit/Twitter, 8s StockTwits)")
    print("   ✅ No iframe processing, no image waiting")
    print("   ✅ Targeted content extraction only")
    print("\n" + "=" * 70)
    
    # Test with 3 popular tickers
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    total_start = time.time()
    
    for i, ticker in enumerate(test_tickers, 1):
        print(f"\n[{i}/{len(test_tickers)}] ⚡ Analyzing {ticker}...")
        ticker_start = time.time()
        
        result = await analyzer.analyze_social_buzz(ticker)
        
        ticker_elapsed = time.time() - ticker_start
        
        # Color code based on speed
        if ticker_elapsed < 3:
            speed_emoji = "🚀"
            speed_label = "EXCELLENT"
        elif ticker_elapsed < 5:
            speed_emoji = "✅"
            speed_label = "GOOD"
        elif ticker_elapsed < 7:
            speed_emoji = "⚠️"
            speed_label = "OK"
        else:
            speed_emoji = "🐌"
            speed_label = "SLOW"
        
        print(f"   {speed_emoji} Speed: {ticker_elapsed:.1f}s ({speed_label})")
        print(f"   📱 Reddit: {result['reddit_mentions']} mentions")
        print(f"   🐦 Twitter: {result['twitter_mentions']} tweets")
        print(f"   💬 StockTwits: {result['stocktwits_mentions']} messages")
        print(f"   📰 News: {result['news_mentions']} articles")
        print(f"   📊 Total: {result['total_mentions']} mentions")
        print(f"   💭 Sentiment: {result['sentiment']} ({result['sentiment_score']:.2f})")
    
    total_elapsed = time.time() - total_start
    avg_time = total_elapsed / len(test_tickers)
    
    print("\n" + "=" * 70)
    print(f"📊 Performance Results:")
    print("=" * 70)
    print(f"   Total time: {total_elapsed:.1f}s for {len(test_tickers)} tickers")
    print(f"   Average: {avg_time:.1f}s per ticker")
    
    if avg_time < 3:
        print(f"\n   🚀 EXCELLENT! Target achieved (<3s per ticker)")
        print(f"   📈 {((6-avg_time)/6*100):.0f}% faster than before (was ~6s)")
    elif avg_time < 5:
        print(f"\n   ✅ GOOD! Close to target")
        print(f"   📈 {((6-avg_time)/6*100):.0f}% faster than before (was ~6s)")
    elif avg_time < 7:
        print(f"\n   ⚠️  OK, but can be improved")
        print(f"   📈 {((6-avg_time)/6*100):.0f}% improvement")
    else:
        print(f"\n   🐌 SLOW - optimizations may not be working")
    
    print("\n" + "=" * 70)
    print("💡 Expected Speed with Optimizations:")
    print("=" * 70)
    print("   • Reddit scraping: ~2-3s (CSS selectors, no nested crawl)")
    print("   • Twitter scraping: ~1-2s (CSS selectors, fast Nitter)")
    print("   • StockTwits: ~1-2s (CSS selectors, reduced scroll wait)")
    print("   • News: ~1-2s (cached API calls)")
    print("   • Total target: 2-3s per ticker")
    
    await analyzer.close()

if __name__ == "__main__":
    asyncio.run(test_speed())

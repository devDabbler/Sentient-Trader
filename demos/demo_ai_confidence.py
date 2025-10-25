"""
Demo: AI Confidence Scanner

Tests the new AI-enhanced scanner that adds intelligent
confidence analysis to top trades.

Run: python demo_ai_confidence.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment FIRST before any other imports
load_dotenv()

from ai_confidence_scanner import AIConfidenceScanner


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def demo_options_with_ai():
    """Demo AI-enhanced options scanner"""
    print_header("🔥 TOP OPTIONS TRADES WITH AI CONFIDENCE")
    
    # Initialize scanner
    print("🤖 Initializing AI Confidence Scanner...")
    ai_scanner = AIConfidenceScanner()
    
    print(f"   LLM Available: {'✅ Yes' if ai_scanner.use_llm else '❌ No (using rule-based)'}")
    print()
    
    # Scan with AI
    print("🔍 Scanning markets for top options with AI analysis...")
    print("   (This may take 1-2 minutes)\n")
    
    try:
        trades = ai_scanner.scan_top_options_with_ai(top_n=10, min_ai_rating=5.0)
        
        if trades:
            # Show insights
            insights = ai_scanner.get_ai_insights(trades)
            
            print("📊 AI Insights:")
            print(f"   Total Opportunities: {insights['total']}")
            print(f"   Average AI Rating: {insights['avg_ai_rating']}/10 ⭐")
            print(f"   Average Quant Score: {insights['avg_quant_score']}/100")
            print(f"   Very High Confidence: {insights['very_high_confidence']}")
            print(f"   Top Pick: {insights['top_pick']}")
            print()
            
            # Show top 5
            print("🏆 TOP 5 AI-ANALYZED OPTIONS TRADES:\n")
            
            for i, trade in enumerate(trades[:5], 1):
                print(f"{i}. {trade.ticker}")
                print(f"   ├─ Quantitative Score: {trade.score}/100")
                print(f"   ├─ AI Rating: {trade.ai_rating}/10 ⭐")
                print(f"   ├─ AI Confidence: {trade.ai_confidence}")
                print(f"   ├─ Price: ${trade.price:.2f} ({trade.change_pct:+.2f}%)")
                print(f"   ├─ Volume: {trade.volume_ratio:.2f}x average")
                print(f"   │")
                print(f"   ├─ 💡 AI Reasoning:")
                print(f"   │  {trade.ai_reasoning}")
                print(f"   │")
                print(f"   └─ ⚠️ AI Risks:")
                print(f"      {trade.ai_risks}")
                print()
            
            # Show comparison
            print("📈 QUANTITATIVE vs AI RATINGS:\n")
            for trade in trades[:5]:
                quant_bar = "█" * int(trade.score / 10)
                ai_bar = "█" * int(trade.ai_rating)
                print(f"{trade.ticker:6} Quant: {quant_bar:10} {trade.score}/100")
                print(f"       AI:    {ai_bar:10} {trade.ai_rating}/10")
                print()
        
        else:
            print("⚠️ No opportunities found (markets may be closed)")
    
    except Exception as e:
        print(f"❌ Error during scan: {e}")
        print("   This is normal if markets are closed or you're offline")


def demo_penny_stocks_with_ai():
    """Demo AI-enhanced penny stock scanner"""
    print_header("💰 TOP PENNY STOCKS WITH AI CONFIDENCE")
    
    # Initialize scanner
    print("🤖 Initializing AI Confidence Scanner...")
    ai_scanner = AIConfidenceScanner()
    
    print(f"   LLM Available: {'✅ Yes' if ai_scanner.use_llm else '❌ No (using rule-based)'}")
    print()
    
    # Scan with AI
    print("🔍 Scanning for top penny stocks with AI analysis...")
    print("   (This may take 2-3 minutes for full analysis)\n")
    
    try:
        trades = ai_scanner.scan_top_penny_stocks_with_ai(top_n=10, min_ai_rating=5.0)
        
        if trades:
            # Show insights
            insights = ai_scanner.get_ai_insights(trades)
            
            print("📊 AI Insights:")
            print(f"   Total Opportunities: {insights['total']}")
            print(f"   Average AI Rating: {insights['avg_ai_rating']}/10 ⭐")
            print(f"   Average Composite Score: {insights['avg_quant_score']}/100")
            print(f"   Very High Confidence: {insights['very_high_confidence']}")
            print(f"   Top Pick: {insights['top_pick']}")
            print()
            
            # Show top 5
            print("🏆 TOP 5 AI-ANALYZED PENNY STOCKS:\n")
            
            for i, trade in enumerate(trades[:5], 1):
                print(f"{i}. {trade.ticker}")
                print(f"   ├─ Composite Score: {trade.score}/100")
                print(f"   ├─ AI Rating: {trade.ai_rating}/10 ⭐")
                print(f"   ├─ AI Confidence: {trade.ai_confidence}")
                print(f"   ├─ Price: ${trade.price:.2f} ({trade.change_pct:+.2f}%)")
                print(f"   ├─ Volume: {trade.volume_ratio:.2f}x average")
                print(f"   │")
                print(f"   ├─ 💡 AI Reasoning:")
                print(f"   │  {trade.ai_reasoning}")
                print(f"   │")
                print(f"   └─ ⚠️ AI Risks:")
                print(f"      {trade.ai_risks}")
                print()
            
            # Filter by high AI confidence
            high_confidence = [t for t in trades if t.ai_rating >= 7.0]
            if high_confidence:
                print(f"⭐ HIGH AI CONFIDENCE PICKS (7.0+): {len(high_confidence)}")
                for trade in high_confidence:
                    print(f"   • {trade.ticker}: {trade.ai_rating}/10 - {trade.ai_confidence}")
                print()
        
        else:
            print("⚠️ No opportunities found (markets may be closed)")
    
    except Exception as e:
        print(f"❌ Error during scan: {e}")
        print("   This is normal if markets are closed or you're offline")


def demo_comparison():
    """Compare regular vs AI-enhanced scans"""
    print_header("📊 COMPARISON: REGULAR vs AI-ENHANCED SCANS")
    
    from top_trades_scanner import TopTradesScanner
    
    print("Running both scanners for comparison...\n")
    
    # Regular scanner
    print("1️⃣ Regular Scanner (Quantitative only):")
    regular_scanner = TopTradesScanner()
    
    try:
        regular_trades = regular_scanner.scan_top_options_trades(top_n=5)
        
        if regular_trades:
            for trade in regular_trades[:3]:
                print(f"   {trade.ticker}: {trade.score}/100 - {trade.confidence}")
                print(f"      Reason: {trade.reason[:60]}...")
            print()
        else:
            print("   No trades found\n")
    except:
        print("   Scan skipped\n")
    
    # AI-enhanced scanner
    print("2️⃣ AI-Enhanced Scanner (Quantitative + AI):")
    ai_scanner = AIConfidenceScanner()
    
    try:
        ai_trades = ai_scanner.scan_top_options_with_ai(top_n=5, min_ai_rating=5.0)
        
        if ai_trades:
            for trade in ai_trades[:3]:
                print(f"   {trade.ticker}: {trade.score}/100 (Quant) + {trade.ai_rating}/10 (AI) = {trade.ai_confidence}")
                print(f"      Quant: {trade.reason[:50]}...")
                print(f"      AI: {trade.ai_reasoning[:50]}...")
                print(f"      Risks: {trade.ai_risks[:50]}...")
            print()
        else:
            print("   No trades found\n")
    except:
        print("   Scan skipped\n")
    
    print("✨ AI Enhancement adds:")
    print("   • Intelligent confidence assessment")
    print("   • Context-aware reasoning")
    print("   • Risk identification")
    print("   • 0-10 rating scale for easy comparison")


def main():
    print("\n" + "="*70)
    print("  🤖 AI CONFIDENCE SCANNER DEMO")
    print("="*70)
    print("\nThis demo showcases the NEW AI-enhanced scanning features:")
    print("  1. Options Scanner with AI Confidence")
    print("  2. Penny Stock Scanner with AI Confidence")
    print("  3. Comparison: Regular vs AI-Enhanced")
    print("\n⏱️  Estimated time: 3-5 minutes")
    print("📡 Requires internet connection")
    print("🔑 Works with or without LLM API keys (rule-based fallback)")
    
    input("\nPress ENTER to start...")
    
    try:
        # Demo 1: Options with AI
        demo_options_with_ai()
        input("\n👆 Review results, then press ENTER to continue...")
        
        # Demo 2: Penny stocks with AI
        demo_penny_stocks_with_ai()
        input("\n👆 Review results, then press ENTER to continue...")
        
        # Demo 3: Comparison
        demo_comparison()
        
        # Summary
        print_header("🎉 DEMO COMPLETE")
        print("✅ AI Confidence feature demonstrated!\n")
        print("📚 Key Takeaways:")
        print("  • AI adds intelligent analysis to quantitative scores")
        print("  • Get reasoning and risks for every trade")
        print("  • 0-10 rating scale for easy comparison")
        print("  • Works with or without LLM API keys")
        print("\n🚀 Next Steps:")
        print("  1. Review FIXED_TABS_GUIDE.md for integration")
        print("  2. Use in your trading workflow")
        print("  3. Compare AI ratings vs actual performance")
        print("\n💡 Quick Usage:")
        print("  from ai_confidence_scanner import AIConfidenceScanner")
        print("  ai_scanner = AIConfidenceScanner()")
        print("  trades = ai_scanner.scan_top_options_with_ai(top_n=20)")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Demo interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

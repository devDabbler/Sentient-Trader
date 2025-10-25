"""
YOUR DAILY TRADING WORKFLOW
ML-Enhanced Confidence Scanning

This is how YOU should use the ML-enhanced scanner for the most confident
trading decisions every day.

Run this script daily before market open or during trading hours to find
the BEST opportunities with maximum confidence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from datetime import datetime
from services.ml_enhanced_scanner import MLEnhancedScanner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def daily_scan_workflow():
    """
    YOUR DAILY WORKFLOW FOR MAXIMUM CONFIDENCE TRADES
    
    This combines:
    - Qlib ML (158 alpha factors, trained models)
    - LLM reasoning (GPT/Claude/Llama analysis)
    - Quantitative signals (RSI, MACD, volume, IV)
    
    Only trades scoring high on ALL THREE make it through.
    """
    
    print("\n" + "="*70)
    print("🎯 ML-ENHANCED DAILY TRADING SCAN")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Initialize scanner
    scanner = MLEnhancedScanner()
    
    print("🔍 Configuration:")
    print(f"   ML Enabled: {scanner.use_ml}")
    print(f"   LLM Enabled: {scanner.ai_scanner.use_llm}")
    print(f"   Ensemble Weights: ML={scanner.weights['ml']}, LLM={scanner.weights['llm']}, Quant={scanner.weights['quant']}")
    print()
    
    # ============================================
    # STEP 1: SCAN OPTIONS (Your main trades)
    # ============================================
    print("\n" + "="*70)
    print("📊 STEP 1: SCANNING OPTIONS OPPORTUNITIES")
    print("="*70)
    
    options_trades = scanner.scan_top_options_with_ml(
        top_n=10,  # Get top 10 trades
        min_ensemble_score=70.0  # Only show very confident trades (70%+)
    )
    
    if options_trades:
        print(f"\n✅ Found {len(options_trades)} HIGH-CONFIDENCE OPTIONS trades:\n")
        
        for i, trade in enumerate(options_trades, 1):
            print(f"\n{'─'*70}")
            print(f"#{i} - {trade.ticker} | Ensemble Score: {trade.combined_score:.1f}/100 | {trade.ensemble_confidence}")
            print(f"{'─'*70}")
            print(f"💰 Price: ${trade.price:.2f} ({trade.change_pct:+.2f}%)")
            print(f"📊 Volume: {trade.volume:,} ({trade.volume_ratio:.2f}x average)")
            print(f"⚠️  Risk: {trade.risk_level}")
            print()
            print(f"🧠 ML Prediction: {trade.ml_prediction_score:.1f}/100 ({trade.ml_confidence})")
            print(f"   - Based on {trade.ml_features_count} alpha factors")
            print()
            print(f"🤖 LLM Analysis: {trade.ai_rating:.1f}/10 ({trade.ai_confidence})")
            print(f"   - Reasoning: {trade.ai_reasoning[:150]}...")
            print(f"   - Risks: {trade.ai_risks[:150]}...")
            print()
            print(f"📈 Quant Score: {trade.score:.1f}/100 ({trade.confidence})")
            print(f"   - {trade.reason}")
            
            # Decision recommendation
            if trade.combined_score >= 85:
                print(f"\n✅ STRONG BUY - All three systems highly confident")
            elif trade.combined_score >= 75:
                print(f"\n✅ BUY - High confidence across systems")
            else:
                print(f"\n⚠️ CONSIDER - Good opportunity but watch risk")
        
        # Show best trade details
        print("\n" + "="*70)
        print("🏆 TOP PICK DETAILED ANALYSIS")
        print("="*70)
        print(scanner.get_trade_explanation(options_trades[0]))
        
    else:
        print("\n⚠️ No high-confidence options trades found at this time.")
        print("   Try lowering min_ensemble_score or expanding the universe.")
    
    # ============================================
    # STEP 2: SCAN PENNY STOCKS (High risk/reward)
    # ============================================
    print("\n" + "="*70)
    print("💎 STEP 2: SCANNING PENNY STOCK OPPORTUNITIES")
    print("="*70)
    
    penny_trades = scanner.scan_top_penny_stocks_with_ml(
        top_n=5,  # Get top 5 penny stocks
        min_ensemble_score=65.0  # Slightly lower threshold for penny stocks
    )
    
    if penny_trades:
        print(f"\n✅ Found {len(penny_trades)} HIGH-CONFIDENCE PENNY STOCKS:\n")
        
        for i, trade in enumerate(penny_trades, 1):
            print(f"{i}. {trade.ticker}: ${trade.price:.2f} | Score: {trade.combined_score:.1f} ({trade.ensemble_confidence})")
            print(f"   ML: {trade.ml_prediction_score:.1f} | LLM: {trade.ai_rating*10:.1f} | Quant: {trade.score:.1f}")
            print(f"   Volume: {trade.volume_ratio:.2f}x | Risk: {trade.risk_level}")
            print()
    else:
        print("\n⚠️ No high-confidence penny stocks found.")
    
    # ============================================
    # STEP 3: SUMMARY & RECOMMENDATIONS
    # ============================================
    print("\n" + "="*70)
    print("📋 DAILY TRADING SUMMARY")
    print("="*70)
    
    total_trades = len(options_trades) + len(penny_trades)
    
    if total_trades > 0:
        print(f"\n✅ Total High-Confidence Opportunities: {total_trades}")
        print(f"   • Options: {len(options_trades)}")
        print(f"   • Penny Stocks: {len(penny_trades)}")
        
        # Risk distribution
        high_risk = len([t for t in options_trades + penny_trades if t.risk_level == 'H'])
        med_risk = len([t for t in options_trades + penny_trades if t.risk_level == 'M'])
        low_risk = len([t for t in options_trades + penny_trades if t.risk_level == 'L'])
        
        print(f"\n📊 Risk Distribution:")
        print(f"   • High Risk: {high_risk}")
        print(f"   • Medium Risk: {med_risk}")
        print(f"   • Low Risk: {low_risk}")
        
        # Confidence distribution
        very_high = len([t for t in options_trades + penny_trades if t.ensemble_confidence == 'VERY HIGH'])
        high = len([t for t in options_trades + penny_trades if t.ensemble_confidence == 'HIGH'])
        
        print(f"\n🎯 Confidence Distribution:")
        print(f"   • VERY HIGH: {very_high}")
        print(f"   • HIGH: {high}")
        
        print(f"\n💡 RECOMMENDATION:")
        if very_high > 0:
            print(f"   Focus on the {very_high} VERY HIGH confidence trades first.")
        if high > 0:
            print(f"   Consider the {high} HIGH confidence trades as secondary opportunities.")
        
        print(f"\n⚠️  RISK MANAGEMENT:")
        print(f"   • Never risk more than 2% of capital per trade")
        print(f"   • Size positions based on risk level (L=larger, H=smaller)")
        print(f"   • Set stop losses based on AI risk analysis")
        print(f"   • Diversify across multiple opportunities")
        
    else:
        print("\n⚠️ No high-confidence trades found today.")
        print("\nPossible reasons:")
        print("   • Market conditions not favorable")
        print("   • Thresholds too high (try lowering min_ensemble_score)")
        print("   • Check if ML/LLM are properly configured")
    
    print("\n" + "="*70)
    print("✅ Scan Complete!")
    print("="*70 + "\n")
    
    return options_trades, penny_trades


def save_trades_to_watchlist(trades, filename='daily_watchlist.txt'):
    """
    Save high-confidence trades to a watchlist file.
    """
    if not trades:
        return
    
    with open(filename, 'w') as f:
        f.write(f"ML-Enhanced Trading Watchlist\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for trade in trades:
            f.write(f"{trade.ticker} | Score: {trade.combined_score:.1f} | {trade.ensemble_confidence}\n")
            f.write(f"  Price: ${trade.price:.2f} ({trade.change_pct:+.2f}%)\n")
            f.write(f"  ML: {trade.ml_prediction_score:.1f} | LLM: {trade.ai_rating*10:.1f} | Quant: {trade.score:.1f}\n")
            f.write(f"  Reason: {trade.reason}\n")
            f.write(f"  AI: {trade.ai_reasoning[:100]}...\n")
            f.write(f"  Risk: {trade.risk_level}\n")
            f.write("\n")
    
    print(f"✅ Watchlist saved to {filename}")


def quick_check_ticker(ticker: str):
    """
    Quick check a specific ticker with ML-enhanced analysis.
    """
    print(f"\n🔍 Quick Check: {ticker}")
    print("="*70)
    
    scanner = MLEnhancedScanner()
    
    # This is simplified - in production you'd scan and filter for this ticker
    print(f"ℹ️  For full analysis, run the daily workflow and look for {ticker}")
    print(f"   Or integrate this with your existing Stock Intelligence tab")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML-Enhanced Trading Workflow")
    parser.add_argument('--ticker', type=str, help='Quick check specific ticker')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--save', action='store_true', help='Save to watchlist')
    
    args = parser.parse_args()
    
    if args.ticker:
        # Quick ticker check
        quick_check_ticker(args.ticker)
    
    elif args.backtest:
        # Run backtest
        print("\n🔬 BACKTESTING ML-ENHANCED STRATEGY")
        print("="*70)
        
        scanner = MLEnhancedScanner()
        results = scanner.backtest_strategy(
            start_date='2024-01-01',
            end_date='2024-10-01',
            min_ensemble_score=70.0
        )
        
        print("\nBacktest Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    
    else:
        # Run daily workflow
        options_trades, penny_trades = daily_scan_workflow()
        
        # Save to watchlist if requested
        if args.save and (options_trades or penny_trades):
            all_trades = options_trades + penny_trades
            save_trades_to_watchlist(all_trades)

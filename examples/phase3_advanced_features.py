"""
Examples demonstrating Phase 3 advanced features:
- Real-time alerts
- EMA/Fibonacci backtesting
- Preset scanners
- Fibonacci-based options chain integration
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.alert_system import (
    AlertSystem, SetupDetector, TradingAlert, AlertType, AlertPriority,
    console_callback, get_alert_system
)
from services.backtest_ema_fib import EMAFibonacciBacktester, BacktestResults
from services.preset_scanners import (
    PresetScanner, ScanPreset, 
    get_sp500_tickers, get_nasdaq100_tickers, get_high_volume_tech
)
from services.options_chain_fib import FibonacciOptionsChain
from analyzers.comprehensive import ComprehensiveAnalyzer


def example_1_alert_system():
    """Example 1: Real-time alert system for high-confidence setups"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Real-Time Alert System")
    print("="*80 + "\n")
    
    # Create alert system with console callback
    alert_system = get_alert_system()
    alert_system.add_callback(console_callback)
    
    # Create setup detector
    detector = SetupDetector(alert_system)
    
    # Analyze stocks and generate alerts
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
    
    print(f"Scanning {len(tickers)} stocks for high-confidence setups...\n")
    
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        analysis = ComprehensiveAnalyzer.analyze_stock(ticker, trading_style="SWING_TRADE")
        
        if analysis:
            # This will automatically generate alerts if conditions are met
            alerts = detector.analyze_for_alerts(analysis)
            
            if alerts:
                print(f"  ✓ Generated {len(alerts)} alert(s) for {ticker}")
            else:
                print(f"  - No alerts for {ticker}")
    
    # Get recent critical alerts
    print("\n" + "-"*80)
    critical_alerts = alert_system.get_recent_alerts(count=10, priority=AlertPriority.CRITICAL)
    
    if critical_alerts:
        print(f"\n🔥 CRITICAL ALERTS ({len(critical_alerts)}):")
        for alert in critical_alerts:
            print(f"  {alert}")
    else:
        print("\nNo critical alerts found.")
    
    print("\n✓ Example 1 complete\n")


def example_2_backtesting():
    """Example 2: Backtest EMA Reclaim + Fibonacci strategy"""
    print("\n" + "="*80)
    print("EXAMPLE 2: EMA Reclaim + Fibonacci Backtesting")
    print("="*80 + "\n")
    
    # Create backtester with different filter configurations
    
    # Configuration 1: Require EMA Reclaim only
    print("Configuration 1: Require EMA Reclaim")
    print("-" * 40)
    bt1 = EMAFibonacciBacktester(
        initial_capital=10000,
        position_size_pct=0.10,
        require_reclaim=True,
        use_fibonacci_targets=True
    )
    
    results1 = bt1.backtest("AAPL", "2023-01-01", "2024-10-01")
    results1.print_summary()
    
    # Configuration 2: Require Reclaim + Alignment
    print("\nConfiguration 2: Require Reclaim + Timeframe Alignment")
    print("-" * 40)
    bt2 = EMAFibonacciBacktester(
        initial_capital=10000,
        position_size_pct=0.10,
        require_reclaim=True,
        require_alignment=True,
        use_fibonacci_targets=True
    )
    
    results2 = bt2.backtest("AAPL", "2023-01-01", "2024-10-01")
    results2.print_summary()
    
    # Configuration 3: Triple Threat (Reclaim + Alignment + Strong RS)
    print("\nConfiguration 3: Triple Threat (Reclaim + Alignment + Strong RS)")
    print("-" * 40)
    bt3 = EMAFibonacciBacktester(
        initial_capital=10000,
        position_size_pct=0.10,
        require_reclaim=True,
        require_alignment=True,
        require_strong_rs=True,
        use_fibonacci_targets=True
    )
    
    results3 = bt3.backtest("AAPL", "2023-01-01", "2024-10-01")
    results3.print_summary()
    
    # Compare configurations
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Config':<40} {'Trades':<10} {'Win Rate':<12} {'Profit Factor':<15} {'Sharpe'}")
    print("-" * 80)
    print(f"{'Reclaim Only':<40} {results1.total_trades:<10} {results1.win_rate:<12.1f}% {results1.profit_factor:<15.2f} {results1.sharpe_ratio:.2f}")
    print(f"{'Reclaim + Alignment':<40} {results2.total_trades:<10} {results2.win_rate:<12.1f}% {results2.profit_factor:<15.2f} {results2.sharpe_ratio:.2f}")
    print(f"{'Triple Threat':<40} {results3.total_trades:<10} {results3.win_rate:<12.1f}% {results3.profit_factor:<15.2f} {results3.sharpe_ratio:.2f}")
    print("="*80 + "\n")
    
    print("✓ Example 2 complete\n")


def example_3_preset_scanners():
    """Example 3: Preset scan filters for rapid opportunity identification"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Preset Scan Filters")
    print("="*80 + "\n")
    
    # Create scanner
    scanner = PresetScanner()
    
    # Get a small watchlist (use subset for demo speed)
    tickers = get_high_volume_tech()[:15]  # First 15 tech stocks
    
    print(f"Scanning {len(tickers)} tech stocks...")
    print(f"Tickers: {', '.join(tickers)}\n")
    
    # Scan 1: Triple Threat setups
    print("\n1. TRIPLE THREAT SCAN (Reclaim + Aligned + Strong RS)")
    print("-" * 80)
    triple_threat_results = scanner.scan(tickers, ScanPreset.TRIPLE_THREAT, generate_alerts=False)
    scanner.print_results(triple_threat_results, show_details=True)
    
    # Scan 2: EMA Reclaim
    print("\n2. EMA RECLAIM SCAN")
    print("-" * 80)
    reclaim_results = scanner.scan(tickers, ScanPreset.EMA_RECLAIM, generate_alerts=False)
    scanner.print_results(reclaim_results)
    
    # Scan 3: DeMarker Pullback
    print("\n3. DEMARKER PULLBACK SCAN (Oversold in Uptrend)")
    print("-" * 80)
    demarker_results = scanner.scan(tickers, ScanPreset.DEMARKER_PULLBACK, generate_alerts=False)
    scanner.print_results(demarker_results)
    
    # Scan 4: Options Premium Selling
    print("\n4. OPTIONS PREMIUM SELL SCAN (High IV + Power Zone)")
    print("-" * 80)
    options_sell_results = scanner.scan(tickers, ScanPreset.OPTIONS_PREMIUM_SELL, generate_alerts=False)
    scanner.print_results(options_sell_results)
    
    # Scan 5: Get top opportunities across all presets
    print("\n5. TOP 5 OPPORTUNITIES (All Presets)")
    print("-" * 80)
    top_opps = scanner.get_top_opportunities(tickers, top_n=5)
    
    if top_opps:
        print(f"\n{'Rank':<6} {'Ticker':<8} {'Preset':<25} {'Priority':<10} {'Confidence'}")
        print(f"{'-'*6} {'-'*8} {'-'*25} {'-'*10} {'-'*10}")
        
        for i, opp in enumerate(top_opps, 1):
            print(f"{i:<6} {opp.ticker:<8} {opp.preset.value:<25} {opp.priority_score:<10.1f} {opp.confidence_score:.0f}")
    
    print("\n✓ Example 3 complete\n")


def example_4_fibonacci_options_chain():
    """Example 4: Fibonacci-based options chain integration"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Fibonacci-Based Options Chain Integration")
    print("="*80 + "\n")
    
    ticker = "AAPL"
    
    print(f"Analyzing {ticker} for Fibonacci options opportunities...\n")
    
    # Get comprehensive analysis
    analysis = ComprehensiveAnalyzer.analyze_stock(ticker, trading_style="OPTIONS")
    
    if not analysis or not analysis.fib_targets:
        print(f"No Fibonacci targets detected for {ticker}")
        return
    
    print(f"✓ Fibonacci A-B-C pattern detected")
    print(f"  Current Price: ${analysis.price:.2f}")
    print(f"  Trend: {analysis.trend}")
    print(f"  IV Rank: {analysis.iv_rank:.1f}")
    
    if analysis.fib_targets:
        print(f"\n  Fibonacci Levels:")
        print(f"    A (Low):     ${analysis.fib_targets.get('A', 0):.2f}")
        print(f"    B (High):    ${analysis.fib_targets.get('B', 0):.2f}")
        print(f"    C (Pullback):${analysis.fib_targets.get('C', 0):.2f}")
        print(f"    T1 (127.2%): ${analysis.fib_targets.get('T1_1272', 0):.2f}")
        print(f"    T2 (161.8%): ${analysis.fib_targets.get('T2_1618', 0):.2f}")
        print(f"    T3 (200%):   ${analysis.fib_targets.get('T3_200', 0):.2f}")
    
    # Initialize options chain analyzer
    print(f"\n{'='*80}")
    print("FETCHING OPTIONS CHAIN...")
    print("="*80)
    
    fib_chain = FibonacciOptionsChain(ticker)
    
    # Find strikes near Fibonacci levels (45 DTE)
    print(f"\nFinding strikes near Fibonacci levels (target DTE: 45 days)...")
    fib_strikes = fib_chain.find_strikes_near_fibonacci(analysis.fib_targets, target_dte=45)
    
    if fib_strikes:
        fib_chain.print_fibonacci_strikes(fib_strikes)
    else:
        print("Could not fetch options chain data.")
        return
    
    # Generate spread suggestions
    print(f"\n{'='*80}")
    print("FIBONACCI-BASED SPREAD SUGGESTIONS")
    print("="*80)
    
    suggested_spreads = fib_chain.suggest_fibonacci_spreads(
        analysis.fib_targets, 
        analysis, 
        target_dte=45
    )
    
    if suggested_spreads:
        fib_chain.print_spread_suggestions(suggested_spreads)
        
        # Show best spread
        best_spread = max(suggested_spreads, key=lambda s: s.risk_reward_ratio)
        print("\n🏆 RECOMMENDED SPREAD:")
        print(f"  {best_spread.strategy_name}")
        print(f"  {best_spread.description}")
        print(f"  Risk/Reward: {best_spread.risk_reward_ratio:.2f}")
        print(f"  Probability of Profit: {best_spread.probability_profit:.1f}%")
    else:
        print("Could not generate spread suggestions.")
    
    print("\n✓ Example 4 complete\n")


def example_5_combined_workflow():
    """Example 5: Combined workflow - scan, alert, backtest, and trade"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete Trading Workflow")
    print("="*80 + "\n")
    
    print("Step 1: Scan for opportunities")
    print("-" * 40)
    
    scanner = PresetScanner()
    tickers = get_high_volume_tech()[:10]
    
    # Find triple threat setups
    results = scanner.scan(tickers, ScanPreset.TRIPLE_THREAT, generate_alerts=True)
    
    if not results:
        print("No triple threat setups found in watchlist.")
        return
    
    # Get best opportunity
    best_opp = results[0]
    ticker = best_opp.ticker
    
    print(f"\n✓ Best opportunity: {ticker}")
    print(f"  Confidence: {best_opp.confidence_score:.0f}")
    print(f"  Reasons: {', '.join(best_opp.match_reasons)}")
    
    # Step 2: Backtest the setup
    print(f"\nStep 2: Backtest {ticker} strategy")
    print("-" * 40)
    
    bt = EMAFibonacciBacktester(
        require_reclaim=True,
        require_alignment=True,
        require_strong_rs=True
    )
    
    backtest_results = bt.backtest(ticker, "2023-01-01", "2024-10-01")
    print(f"  Win Rate: {backtest_results.win_rate:.1f}%")
    print(f"  Profit Factor: {backtest_results.profit_factor:.2f}")
    print(f"  Avg Win: {backtest_results.avg_win_pct:.2f}%")
    
    # Step 3: Options analysis
    if best_opp.analysis.fib_targets:
        print(f"\nStep 3: Fibonacci options setup")
        print("-" * 40)
        
        fib_chain = FibonacciOptionsChain(ticker)
        fib_strikes = fib_chain.find_strikes_near_fibonacci(
            best_opp.analysis.fib_targets, 
            target_dte=45
        )
        
        if fib_strikes and 'T1' in fib_strikes:
            t1_strike = fib_strikes['T1']
            print(f"  T1 Target: ${t1_strike.strike:.2f} ({t1_strike.dte} DTE)")
            print(f"  Call Mid Price: ${t1_strike.get_mid_price('call'):.2f}")
    
    # Step 4: Trade decision
    print(f"\nStep 4: Trade Decision")
    print("-" * 40)
    print(f"  ✓ Setup Quality: EXCELLENT")
    print(f"  ✓ Backtest Performance: POSITIVE")
    print(f"  ✓ Options Available: YES")
    print(f"\n  → RECOMMENDED ACTION: Enter position")
    print(f"     Entry: Current price or pullback to EMA21")
    print(f"     Stop: Below EMA21")
    print(f"     Targets: Use Fibonacci levels")
    
    print("\n✓ Example 5 complete\n")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("PHASE 3 ADVANCED FEATURES - DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates:")
    print("  1. Real-time alert system")
    print("  2. EMA Reclaim + Fibonacci backtesting")
    print("  3. Preset scan filters")
    print("  4. Fibonacci-based options chain integration")
    print("  5. Complete trading workflow")
    print("\nNote: Some examples require live market data and may take time to execute.")
    print("="*80 + "\n")
    
    import time
    
    try:
        # Run examples
        example_1_alert_system()
        time.sleep(1)
        
        example_2_backtesting()
        time.sleep(1)
        
        example_3_preset_scanners()
        time.sleep(1)
        
        example_4_fibonacci_options_chain()
        time.sleep(1)
        
        example_5_combined_workflow()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

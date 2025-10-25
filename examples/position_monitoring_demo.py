"""
Position Monitoring Demo

Demonstrates how to use the PositionMonitor to track active trades
and receive alerts for position changes, profit targets, and stop losses.
"""

import sys
import os
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.position_monitor import PositionMonitor, get_position_monitor
from services.alert_system import AlertSystem, get_alert_system, console_callback
from src.integrations.tradier_client import create_tradier_client_from_env
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def demo_position_monitoring():
    """Demonstrate position monitoring with live updates"""
    
    print("=" * 80)
    print("POSITION MONITORING DEMO")
    print("=" * 80)
    print()
    
    # 1. Setup alert system with console callback
    print("1. Setting up alert system...")
    alert_system = get_alert_system()
    alert_system.add_callback(console_callback)
    print("   ✓ Alert system configured with console output")
    print()
    
    # 2. Initialize Tradier client
    print("2. Connecting to Tradier...")
    try:
        tradier_client = create_tradier_client_from_env()
        if not tradier_client:
            print("   ✗ Failed to create Tradier client. Check your .env file.")
            return
        print("   ✓ Connected to Tradier")
    except Exception as e:
        print(f"   ✗ Error connecting to Tradier: {e}")
        return
    print()
    
    # 3. Create position monitor
    print("3. Initializing position monitor...")
    position_monitor = get_position_monitor(alert_system, tradier_client)
    
    # Configure alert thresholds
    position_monitor.pnl_alert_threshold = 5.0  # Alert every 5% change
    position_monitor.significant_loss_threshold = -10.0  # Alert on 10% loss
    position_monitor.significant_gain_threshold = 15.0  # Alert on 15% gain
    
    print("   ✓ Position monitor configured")
    print(f"   - P&L alert threshold: {position_monitor.pnl_alert_threshold}%")
    print(f"   - Significant loss threshold: {position_monitor.significant_loss_threshold}%")
    print(f"   - Significant gain threshold: {position_monitor.significant_gain_threshold}%")
    print()
    
    # 4. Fetch and monitor positions
    print("4. Monitoring positions (Press Ctrl+C to stop)...")
    print()
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Update #{iteration} at {datetime.now().strftime('%H:%M:%S')} ---")
            
            # Update positions and get alerts
            success, alerts = position_monitor.update_positions()
            
            if success:
                # Get summary
                summary = position_monitor.get_position_summary()
                
                print(f"Open Positions: {summary['total_positions']}")
                print(f"Profitable: {summary['profitable_positions']} | Losing: {summary['losing_positions']}")
                print(f"Total P&L: ${summary['total_pnl']:,.2f}")
                
                if summary['positions']:
                    print("\nPosition Details:")
                    for symbol, pos in summary['positions'].items():
                        pnl_emoji = "🟢" if pos['pnl_dollars'] >= 0 else "🔴"
                        print(f"  {pnl_emoji} {symbol}: {pos['quantity']:.0f} shares @ ${pos['current_price']:.2f} | "
                              f"P&L: ${pos['pnl_dollars']:+,.2f} ({pos['pnl_percent']:+.2f}%)")
                
                if alerts:
                    print(f"\n{len(alerts)} new alert(s) generated this update")
                else:
                    print("\nNo new alerts")
            else:
                print("⚠️ Failed to update positions")
            
            # Wait 30 seconds before next update
            print("\nWaiting 30 seconds for next update...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped by user")
        print()
    
    # 5. Show final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    summary = position_monitor.get_position_summary()
    print(f"Total Positions Tracked: {summary['total_positions']}")
    print(f"Total P&L: ${summary['total_pnl']:,.2f}")
    
    # Show recent alerts
    recent_alerts = alert_system.get_recent_alerts(count=10)
    if recent_alerts:
        print(f"\nLast {len(recent_alerts)} Alerts:")
        for alert in recent_alerts:
            print(f"  [{alert.timestamp.strftime('%H:%M:%S')}] {alert.ticker}: {alert.message}")
    
    print()


def demo_manual_position_tracking():
    """Demonstrate setting stop losses and profit targets"""
    
    print("=" * 80)
    print("MANUAL POSITION TRACKING DEMO")
    print("=" * 80)
    print()
    
    # Setup
    alert_system = get_alert_system()
    alert_system.add_callback(console_callback)
    
    tradier_client = create_tradier_client_from_env()
    if not tradier_client:
        print("Failed to create Tradier client")
        return
    
    position_monitor = get_position_monitor(alert_system, tradier_client)
    
    # Initial update
    print("Fetching current positions...")
    success, alerts = position_monitor.update_positions()
    
    if not success:
        print("Failed to fetch positions")
        return
    
    summary = position_monitor.get_position_summary()
    
    if summary['total_positions'] == 0:
        print("No open positions to track")
        return
    
    print(f"\nFound {summary['total_positions']} open position(s):")
    for symbol in summary['positions'].keys():
        print(f"  - {symbol}")
    
    print("\nSetting stop losses and profit targets...")
    
    # Example: Set stop loss and profit target for first position
    for symbol, pos_data in summary['positions'].items():
        current_price = pos_data['current_price']
        
        # Set 10% stop loss
        stop_loss = current_price * 0.90
        position_monitor.set_stop_loss(symbol, stop_loss)
        print(f"  ✓ {symbol}: Stop Loss set at ${stop_loss:.2f} (-10%)")
        
        # Set 20% profit target
        profit_target = current_price * 1.20
        position_monitor.set_profit_target(symbol, profit_target)
        print(f"  ✓ {symbol}: Profit Target set at ${profit_target:.2f} (+20%)")
    
    print("\n✓ Alerts will be sent when stop losses or profit targets are hit")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Position Monitoring Demo")
    parser.add_argument(
        '--mode',
        choices=['monitor', 'manual'],
        default='monitor',
        help='Demo mode: monitor (continuous) or manual (set stops/targets)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        demo_position_monitoring()
    else:
        demo_manual_position_tracking()

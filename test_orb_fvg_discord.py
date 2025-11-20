"""
Test script for ORB+FVG Discord alerts and journaling

Verifies:
1. Discord webhook is configured correctly
2. Alerts are sent successfully
3. Trade journal integration works
4. All components are properly connected
"""

from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
from loguru import logger

# Configure logger
logger.add("logs/orb_fvg_test.log", rotation="1 day")

from services.orb_fvg_strategy import ORBFVGSignal, ORBLevel, FairValueGap
from services.orb_fvg_alerts import create_orb_fvg_alert_manager


def test_discord_webhook():
    """Test if Discord webhook is configured"""
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    print("\n" + "="*60)
    print("TESTING DISCORD WEBHOOK CONFIGURATION")
    print("="*60)
    
    if not webhook_url:
        print("❌ DISCORD_WEBHOOK_URL not found in .env")
        print("   Please add: DISCORD_WEBHOOK_URL=your_webhook_url")
        return False
    
    if not webhook_url.startswith('https://discord.com/api/webhooks/'):
        print(f"⚠️  Webhook URL format looks incorrect: {webhook_url[:50]}...")
        print("   Should start with: https://discord.com/api/webhooks/")
        return False
    
    print(f"✅ Discord webhook configured: {webhook_url[:50]}...")
    return True


def create_mock_signal():
    """Create a mock ORB+FVG signal for testing"""
    
    # Mock FVG
    fvg = FairValueGap(
        gap_type='bullish',
        top=450.20,
        bottom=449.80,
        timestamp=datetime.now(),
        strength=75.0
    )
    
    # Mock signal
    signal = ORBFVGSignal(
        symbol='SPY',
        timestamp=datetime.now(),
        signal_type='LONG',
        confidence=82.5,
        orb_high=450.00,
        orb_low=449.00,
        orb_range_pct=0.22,
        fvg=fvg,
        fvg_alignment=True,
        entry_price=450.15,
        stop_loss=449.00,
        target_price=452.45,
        risk_amount=1.15,
        reward_amount=2.30,
        risk_reward_ratio=2.0,
        current_price=450.15,
        volume_ratio=2.1
    )
    
    return signal


def test_alert_manager():
    """Test ORB+FVG alert manager creation"""
    print("\n" + "="*60)
    print("TESTING ALERT MANAGER CREATION")
    print("="*60)
    
    try:
        alert_manager = create_orb_fvg_alert_manager()
        print("✅ Alert manager created successfully")
        print(f"   - Journal: {type(alert_manager.journal).__name__}")
        print(f"   - Alert System: {type(alert_manager.alert_system).__name__}")
        return alert_manager
    except Exception as e:
        print(f"❌ Error creating alert manager: {e}")
        logger.error(f"Alert manager creation error: {e}", exc_info=True)
        return None


def test_send_signal_alert(alert_manager):
    """Test sending a signal alert to Discord"""
    print("\n" + "="*60)
    print("TESTING SIGNAL ALERT (Discord)")
    print("="*60)
    
    try:
        # Create mock signal
        signal = create_mock_signal()
        
        print(f"Sending test alert for {signal.symbol}...")
        print(f"  Signal: {signal.signal_type}")
        print(f"  Confidence: {signal.confidence:.1f}%")
        print(f"  Entry: ${signal.entry_price:.2f}")
        print(f"  Target: ${signal.target_price:.2f}")
        print(f"  Stop: ${signal.stop_loss:.2f}")
        
        # Send alert
        alert_manager.send_signal_alert(signal)
        
        print("✅ Signal alert sent successfully!")
        print("   Check your Discord channel for the alert")
        return True
        
    except Exception as e:
        print(f"❌ Error sending signal alert: {e}")
        logger.error(f"Signal alert error: {e}", exc_info=True)
        return False


def test_journal_entry(alert_manager):
    """Test logging trade to journal"""
    print("\n" + "="*60)
    print("TESTING TRADE JOURNAL ENTRY")
    print("="*60)
    
    try:
        # Create mock signal
        signal = create_mock_signal()
        
        print(f"Logging test trade for {signal.symbol}...")
        
        # Log trade entry
        trade_id = alert_manager.log_trade_entry(
            signal=signal,
            actual_entry_price=450.15,
            quantity=100,
            broker="TRADIER_PAPER"
        )
        
        if trade_id:
            print(f"✅ Trade logged successfully!")
            print(f"   Trade ID: {trade_id}")
            print(f"   Symbol: {signal.symbol}")
            print(f"   Quantity: 100 shares")
            print(f"   Entry: ${signal.entry_price:.2f}")
            print("   Check Discord for execution alert")
            return trade_id
        else:
            print("❌ Trade logging failed - no trade_id returned")
            return None
        
    except Exception as e:
        print(f"❌ Error logging trade: {e}")
        logger.error(f"Journal entry error: {e}", exc_info=True)
        return None


def test_journal_stats(alert_manager):
    """Test retrieving ORB+FVG stats from journal"""
    print("\n" + "="*60)
    print("TESTING JOURNAL STATISTICS")
    print("="*60)
    
    try:
        stats = alert_manager.get_orb_fvg_stats()
        
        print("✅ Statistics retrieved:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        logger.error(f"Stats error: {e}", exc_info=True)
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ORB+FVG DISCORD & JOURNAL INTEGRATION TEST")
    print("="*60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Discord webhook configuration
    discord_ok = test_discord_webhook()
    
    if not discord_ok:
        print("\n⚠️  Discord webhook not configured correctly")
        print("   Continuing with other tests...\n")
    
    # Test 2: Alert manager creation
    alert_manager = test_alert_manager()
    
    if not alert_manager:
        print("\n❌ Cannot continue - alert manager failed to initialize")
        return
    
    # Test 3: Send signal alert
    alert_ok = test_send_signal_alert(alert_manager)
    
    # Test 4: Journal entry
    trade_id = test_journal_entry(alert_manager)
    
    # Test 5: Journal stats
    stats_ok = test_journal_stats(alert_manager)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Discord Webhook: {'✅' if discord_ok else '❌'}")
    print(f"Alert Manager: {'✅' if alert_manager else '❌'}")
    print(f"Signal Alert: {'✅' if alert_ok else '❌'}")
    print(f"Journal Entry: {'✅' if trade_id else '❌'}")
    print(f"Journal Stats: {'✅' if stats_ok else '❌'}")
    
    if all([discord_ok, alert_manager, alert_ok, trade_id, stats_ok]):
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour ORB+FVG strategy is fully integrated with:")
        print("  ✅ Discord alerts")
        print("  ✅ Trade journaling")
        print("  ✅ Automated logging")
        print("\nNext steps:")
        print("  1. Check your Discord channel for test alerts")
        print("  2. Verify trade appears in journal")
        print("  3. Ready to use in production!")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Check logs/orb_fvg_test.log for details")
        print("   Review .env file for missing configuration")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

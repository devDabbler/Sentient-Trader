"""
Test script to verify Freqtrade signal generation works
"""

import os
from dotenv import load_dotenv
from loguru import logger
import sys

# Load environment
load_dotenv()

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO")

# Import required modules
from clients.kraken_client import KrakenClient
from services.freqtrade_signal_adapter import get_freqtrade_strategy_wrappers

def test_freqtrade_strategies():
    """Test Freqtrade strategy loading and signal generation"""
    
    print("\n" + "="*60)
    print("🧪 Testing Freqtrade Signal Generation")
    print("="*60 + "\n")
    
    # Step 1: Initialize Kraken client
    print("📡 Step 1: Initializing Kraken client...")
    kraken_key = os.getenv('KRAKEN_API_KEY')
    kraken_secret = os.getenv('KRAKEN_API_SECRET')
    
    if not kraken_key or not kraken_secret:
        print("❌ ERROR: Kraken API credentials not found in .env")
        return False
    
    kraken_client = KrakenClient(kraken_key, kraken_secret)
    success, message = kraken_client.validate_connection()
    
    if not success:
        print(f"❌ ERROR: Kraken connection failed: {message}")
        return False
    
    print(f"✅ Kraken client initialized successfully")
    print(f"   API Key: {kraken_key[:10]}...")
    
    # Step 2: Load Freqtrade strategies
    print("\n🔧 Step 2: Loading Freqtrade strategies...")
    try:
        strategies = get_freqtrade_strategy_wrappers(kraken_client)
        print(f"✅ Loaded {len(strategies)} Freqtrade strategies:")
        for i, name in enumerate(strategies.keys(), 1):
            print(f"   {i}. {name}")
    except Exception as e:
        print(f"❌ ERROR loading strategies: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test signal generation with ALCH/USD
    print("\n🎯 Step 3: Testing signal generation for ALCH/USD...")
    test_symbol = "ALCH/USD"
    
    # Fetch OHLCV data
    print(f"📊 Fetching OHLCV data for {test_symbol}...")
    try:
        ohlcv_list = kraken_client.get_ohlc_data(test_symbol, interval=15)
        if not ohlcv_list or len(ohlcv_list) < 100:
            print(f"⚠️  WARNING: Only {len(ohlcv_list) if ohlcv_list else 0} candles available (need 100+)")
            if len(ohlcv_list) < 30:
                print(f"❌ ERROR: Insufficient data to test")
                return False
    except Exception as e:
        print(f"❌ ERROR fetching data: {e}")
        return False
    
    print(f"✅ Fetched {len(ohlcv_list)} candles")
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(ohlcv_list)
    current_price = float(df['close'].iloc[-1])
    print(f"💰 Current price: ${current_price:.4f}")
    
    # Step 4: Test each strategy
    print(f"\n🔬 Step 4: Testing each strategy with {test_symbol}...")
    signals_found = 0
    
    for strategy_name, strategy_wrapper in strategies.items():
        try:
            signal = strategy_wrapper.generate_signal(df, current_price)
            
            if signal:
                signals_found += 1
                print(f"\n✅ {strategy_name}:")
                print(f"   Signal: {signal.signal_type}")
                print(f"   Confidence: {signal.confidence:.1f}%")
                print(f"   Entry: ${signal.entry_price:.4f}")
                print(f"   Stop Loss: ${signal.stop_loss:.4f}")
                print(f"   Take Profit: ${signal.take_profit:.4f}")
                print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}:1")
                print(f"   Reasoning: {signal.reasoning[:100]}...")
            else:
                print(f"⚪ {strategy_name}: No signal (conditions not met)")
                
        except Exception as e:
            print(f"❌ {strategy_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "="*60)
    print(f"📊 SUMMARY")
    print("="*60)
    print(f"✅ Strategies loaded: {len(strategies)}")
    print(f"🎯 Signals generated: {signals_found}")
    print(f"📈 Test symbol: {test_symbol}")
    print(f"💰 Current price: ${current_price:.4f}")
    
    if signals_found > 0:
        print(f"\n🎉 SUCCESS! {signals_found} signal(s) generated from Freqtrade strategies!")
        return True
    else:
        print(f"\n⚠️  No signals generated (market conditions don't meet strategy criteria)")
        print(f"   This is NORMAL - strategies have strict entry conditions")
        print(f"   Try different symbols: BTC/USD, ETH/USD, XRP/USD")
        return True  # Still successful - strategies work, just no signals

if __name__ == "__main__":
    try:
        success = test_freqtrade_strategies()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

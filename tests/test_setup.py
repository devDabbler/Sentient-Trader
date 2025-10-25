"""
Test script to verify Option Alpha and Tradier integration setup
Run this script to test your configuration before using the main application
"""

import os
import requests
import json
from dotenv import load_dotenv
from tradier_client import TradierClient, validate_tradier_connection

# Load environment variables
load_dotenv()

def test_environment_variables():
    """Test if all required environment variables are set"""
    print("🔍 Testing Environment Variables...")
    
    required_vars = {
        'OPTION_ALPHA_WEBHOOK_URL': 'Option Alpha webhook URL',
        'TRADIER_ACCOUNT_ID': 'Tradier account ID',
        'TRADIER_ACCESS_TOKEN': 'Tradier access token',
        'TRADIER_API_URL': 'Tradier API URL'
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            if 'TOKEN' in var or 'URL' in var:
                # Mask sensitive values
                masked_value = value[:10] + "..." + value[-4:] if len(value) > 14 else "***"
                print(f"✅ {description}: {masked_value}")
            else:
                print(f"✅ {description}: {value}")
        else:
            print(f"❌ {description}: NOT SET")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all variables are set.")
        return False
    
    print("✅ All environment variables are set!")
    return True

def test_option_alpha_webhook():
    """Test Option Alpha webhook connection"""
    print("\n🔗 Testing Option Alpha Webhook...")
    
    webhook_url = os.getenv('OPTION_ALPHA_WEBHOOK_URL')
    if not webhook_url:
        print("❌ Option Alpha webhook URL not configured")
        return False
    
    # Test webhook with a simple signal
    test_signal = {
        "ticker": "SPY",
        "action": "SELL_PUT",
        "strike": 450,
        "expiry": "2024-02-16",
        "qty": 1,
        "note": "Test signal from setup verification"
    }
    
    try:
        response = requests.post(
            webhook_url,
            json=test_signal,
            timeout=10,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            print("✅ Option Alpha webhook test successful!")
            print(f"   Response: {response.status_code}")
            return True
        else:
            print(f"❌ Option Alpha webhook test failed!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Option Alpha webhook test failed!")
        print(f"   Error: {str(e)}")
        return False

def test_tradier_connection():
    """Test Tradier API connection"""
    print("\n🏦 Testing Tradier API Connection...")
    
    success, message = validate_tradier_connection()
    
    if success:
        print(f"✅ {message}")
        
        # Get additional account info
        client = TradierClient(
            account_id=os.getenv('TRADIER_ACCOUNT_ID'),
            access_token=os.getenv('TRADIER_ACCESS_TOKEN'),
            api_url=os.getenv('TRADIER_API_URL', 'https://sandbox.tradier.com')
        )
        
        # Test getting account balance
        success, balance = client.get_account_balance()
        if success:
            print("✅ Account balance retrieved successfully")
            bal_data = balance.get('balances', {})
            print(f"   Total Cash: ${float(bal_data.get('total_cash', 0)):,.2f}")
            print(f"   Buying Power: ${float(bal_data.get('buying_power', 0)):,.2f}")
        else:
            print(f"⚠️ Could not retrieve account balance: {balance.get('error', 'Unknown error')}")
        
        return True
    else:
        print(f"❌ {message}")
        return False

def test_signal_conversion():
    """Test converting Option Alpha signal to Tradier order"""
    print("\n🔄 Testing Signal Conversion...")
    
    client = TradierClient(
        account_id=os.getenv('TRADIER_ACCOUNT_ID'),
        access_token=os.getenv('TRADIER_ACCESS_TOKEN'),
        api_url=os.getenv('TRADIER_API_URL', 'https://sandbox.tradier.com')
    )
    
    test_signal = {
        "ticker": "AAPL",
        "action": "SELL_PUT",
        "strike": 150,
        "expiry": "2024-02-16",
        "qty": 1,
        "iv_rank": 65,
        "estimated_risk": 15000,
        "llm_score": 0.75,
        "note": "Test conversion"
    }
    
    try:
        order_data = client.convert_signal_to_order(test_signal)
        
        if "error" in order_data:
            print(f"❌ Signal conversion failed: {order_data['error']}")
            return False
        else:
            print("✅ Signal conversion successful!")
            print(f"   Converted order: {json.dumps(order_data, indent=2)}")
            return True
            
    except Exception as e:
        print(f"❌ Signal conversion failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Option Alpha + Tradier Integration Test")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Option Alpha Webhook", test_option_alpha_webhook),
        ("Tradier Connection", test_tradier_connection),
        ("Signal Conversion", test_signal_conversion)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready to use.")
        print("You can now run: streamlit run app.py")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please check the setup guide.")
        print("Refer to SETUP_GUIDE.md for detailed instructions.")

if __name__ == "__main__":
    main()

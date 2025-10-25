"""
Example usage of OTOCO bracket orders with Tradier API

This demonstrates how to place bracket orders with automatic
take-profit and stop-loss orders.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integrations.tradier_client import TradierClient

# Initialize client
client = TradierClient(
    account_id='YOUR_ACCOUNT_ID',
    access_token='YOUR_ACCESS_TOKEN',
    api_url='https://sandbox.tradier.com'  # Use sandbox for testing
)

# Example 1: Basic bracket order with specific prices
print("Example 1: Bracket order with specific prices")
success, result = client.place_bracket_order(
    symbol='AAPL',
    side='buy',
    quantity=100,
    entry_price=150.00,
    take_profit_price=160.00,  # Sell at $160 for profit
    stop_loss_price=145.00,     # Sell at $145 to limit loss
    duration='gtc',
    tag='BRACKET_AAPL_001'
)

if success:
    print(f"✅ Bracket order placed successfully!")
    print(f"Order ID: {result.get('order', {}).get('id')}")
else:
    print(f"❌ Error: {result.get('error')}")

print("\n" + "="*60 + "\n")

# Example 2: Bracket order with percentage-based exits
print("Example 2: Bracket order with percentage-based exits")
success, result = client.place_bracket_order_percentage(
    symbol='TSLA',
    side='buy',
    quantity=50,
    entry_price=200.00,
    take_profit_pct=5.0,    # Take profit at +5% ($210)
    stop_loss_pct=3.0,      # Stop loss at -3% ($194)
    duration='gtc',
    tag='BRACKET_TSLA_PCT'
)

if success:
    print(f"✅ Percentage-based bracket order placed!")
    print(f"Order ID: {result.get('order', {}).get('id')}")
else:
    print(f"❌ Error: {result.get('error')}")

print("\n" + "="*60 + "\n")

# Example 3: Options bracket order
print("Example 3: Options bracket order")
success, result = client.place_bracket_order(
    symbol='SPY',
    side='buy',
    quantity=10,  # 10 contracts
    entry_price=2.50,
    take_profit_price=3.00,
    stop_loss_price=2.20,
    option_symbol='SPY250117C00450000',  # OCC format
    duration='gtc',
    tag='BRACKET_SPY_OPTION'
)

if success:
    print(f"✅ Options bracket order placed!")
    print(f"Order ID: {result.get('order', {}).get('id')}")
else:
    print(f"❌ Error: {result.get('error')}")

print("\n" + "="*60 + "\n")

# Example 4: Short position with bracket
print("Example 4: Short position with bracket")
success, result = client.place_bracket_order_percentage(
    symbol='NVDA',
    side='sell',  # Short entry
    quantity=100,
    entry_price=500.00,
    take_profit_pct=4.0,    # Take profit at -4% ($480) - buying back lower
    stop_loss_pct=2.5,      # Stop loss at +2.5% ($512.50) - buying back higher
    duration='gtc',
    tag='BRACKET_NVDA_SHORT'
)

if success:
    print(f"✅ Short bracket order placed!")
    print(f"Order ID: {result.get('order', {}).get('id')}")
else:
    print(f"❌ Error: {result.get('error')}")

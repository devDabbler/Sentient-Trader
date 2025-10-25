"""
Test script for sending a Discord alert.
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.alerts import TradingAlert, AlertType, AlertPriority
from src.integrations.discord_webhook import send_discord_alert

def test_discord_notification():
    """Creates and sends a sample Discord alert."""
    print('Loading environment variables from .env file...')
    load_dotenv()

    if not os.getenv('DISCORD_WEBHOOK_URL'):
        print('ERROR: DISCORD_WEBHOOK_URL is not set in the .env file.')
        print('Please make sure your .env file is in the root directory and contains the webhook URL.')
        return

    print('Constructing a sample trading alert...')
    test_alert = TradingAlert(
        ticker='TSLA',
        alert_type=AlertType.EMA_RECLAIM,
        priority=AlertPriority.CRITICAL,
        message='🔥 TEST: EMA RECLAIM CONFIRMED - High probability bullish setup',
        confidence_score=95.5,
        details={
            'ema8': 305.50,
            'ema21': 301.00,
            'price': 306.00,
            'demarker': 0.25
        }
    )

    print(f'Sending test alert for {test_alert.ticker} to Discord...')
    try:
        send_discord_alert(test_alert)
        print('✅ Test alert sent successfully!')
        print('Please check your Discord channel to confirm you received the notification.')
    except Exception as e:
        print(f'❌ Failed to send test alert: {e}')

if __name__ == '__main__':
    test_discord_notification()

"""
Discord Webhook for Trading Alerts
"""

import os
import requests
import logging
from models.alerts import TradingAlert

logger = logging.getLogger(__name__)

def send_discord_alert(alert: TradingAlert):
    """Sends a trading alert to a Discord webhook."""
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        logger.warning('DISCORD_WEBHOOK_URL not set. Skipping Discord notification.')
        return

    embed = {
        'title': f'🚨 {alert.priority.value} Alert: {alert.ticker}',
        'description': alert.message,
        'color': {
            'CRITICAL': 15158332,
            'HIGH': 15844367,
            'MEDIUM': 16776960,
            'LOW': 10070709
        }.get(alert.priority.value, 10070709),
        'fields': [
            {'name': 'Confidence Score', 'value': f'{alert.confidence_score:.2f}', 'inline': True},
            {'name': 'Alert Type', 'value': alert.alert_type.value, 'inline': True}
        ],
        'timestamp': alert.timestamp.isoformat()
    }

    payload = {
        'embeds': [embed]
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logger.info(f'Successfully sent alert for {alert.ticker} to Discord.')
    except requests.exceptions.RequestException as e:
        logger.error(f'Failed to send Discord alert for {alert.ticker}: {e}')

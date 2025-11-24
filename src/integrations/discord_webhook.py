"""
Discord Webhook for Trading Alerts

IMPORTANT: Uses lazy import of requests to prevent Task Scheduler hangs
"""

import os
from loguru import logger
from models.alerts import TradingAlert, AlertType

# Lazy import - only load when actually sending webhook
def _get_requests():
    import requests
    return requests



def _build_earnings_fields(details: dict) -> list:
    """Build fields for earnings alerts"""
    fields = []
    
    if details.get('days_until') is not None:
        fields.append({
            'name': '📅 Days Until',
            'value': str(details['days_until']),
            'inline': True
        })
    
    if details.get('eps_estimate'):
        fields.append({
            'name': '💰 EPS Estimate',
            'value': f"${details['eps_estimate']:.2f}",
            'inline': True
        })
    
    if details.get('iv_rank'):
        fields.append({
            'name': '📊 IV Rank',
            'value': f"{details['iv_rank']:.1f}%",
            'inline': True
        })
    
    if details.get('has_position'):
        fields.append({
            'name': '⚠️ Position Status',
            'value': 'ACTIVE POSITION',
            'inline': False
        })
    
    return fields


def _build_news_fields(details: dict) -> list:
    """Build fields for news alerts"""
    fields = []
    
    if details.get('sentiment'):
        sentiment_emoji = {'positive': '📈', 'negative': '📉', 'neutral': '➡️'}
        fields.append({
            'name': f"{sentiment_emoji.get(details['sentiment'], '📰')} Sentiment",
            'value': details['sentiment'].upper(),
            'inline': True
        })
    
    if details.get('news_count'):
        fields.append({
            'name': '📰 News Items',
            'value': str(details['news_count']),
            'inline': True
        })
    
    if details.get('publisher'):
        fields.append({
            'name': '📡 Source',
            'value': details['publisher'],
            'inline': True
        })
    
    if details.get('link'):
        fields.append({
            'name': '🔗 Link',
            'value': f"[Read Article]({details['link']})",
            'inline': False
        })
    
    return fields


def _build_sec_fields(details: dict) -> list:
    """Build fields for SEC filing alerts"""
    fields = []
    
    if details.get('form_type'):
        fields.append({
            'name': '📋 Form Type',
            'value': details['form_type'],
            'inline': True
        })
    
    if details.get('description'):
        fields.append({
            'name': '📄 Description',
            'value': details['description'],
            'inline': True
        })
    
    if details.get('url'):
        fields.append({
            'name': '🔗 Filing Link',
            'value': f"[View on SEC EDGAR]({details['url']})",
            'inline': False
        })
    
    return fields


def _build_economic_fields(details: dict) -> list:
    """Build fields for economic event alerts"""
    fields = []
    
    if details.get('days_until') is not None:
        fields.append({
            'name': '📅 Days Until',
            'value': str(details['days_until']),
            'inline': True
        })
    
    if details.get('impact'):
        fields.append({
            'name': '💥 Impact',
            'value': details['impact'].upper(),
            'inline': True
        })
    
    if details.get('affected_sectors'):
        fields.append({
            'name': '🏢 Affected Sectors',
            'value': details['affected_sectors'],
            'inline': False
        })
    
    if details.get('estimate'):
        fields.append({
            'name': '📊 Estimate',
            'value': str(details['estimate']),
            'inline': True
        })
    
    if details.get('previous'):
        fields.append({
            'name': '📈 Previous',
            'value': str(details['previous']),
            'inline': True
        })
    
    return fields


def _build_trading_decision_fields(details: dict) -> list:
    """Build fields for buy/sell/speculation alerts"""
    fields = []
    
    if details.get('entry_price'):
        fields.append({
            'name': '💵 Entry Price',
            'value': f"${details['entry_price']:.2f}",
            'inline': True
        })
    
    if details.get('target_price'):
        fields.append({
            'name': '🎯 Target',
            'value': f"${details['target_price']:.2f}",
            'inline': True
        })
    
    if details.get('stop_loss'):
        fields.append({
            'name': '🛑 Stop Loss',
            'value': f"${details['stop_loss']:.2f}",
            'inline': True
        })
    
    if details.get('risk_reward'):
        fields.append({
            'name': '⚖️ Risk/Reward',
            'value': f"{details['risk_reward']:.2f}:1",
            'inline': True
        })
    
    if details.get('position_size'):
        fields.append({
            'name': '📦 Position Size',
            'value': details['position_size'],
            'inline': True
        })
    
    if details.get('reasoning'):
        fields.append({
            'name': '💡 Reasoning',
            'value': details['reasoning'][:1024],  # Discord limit
            'inline': False
        })
    
    return fields


def send_discord_alert(alert: TradingAlert):
    """Sends a trading alert to a Discord webhook with enhanced formatting."""
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        logger.warning('DISCORD_WEBHOOK_URL not set. Skipping Discord notification.')
        return
    
    # Validate webhook URL format
    if not webhook_url.startswith('https://discord.com/api/webhooks/') and not webhook_url.startswith('https://discordapp.com/api/webhooks/'):
        logger.error(f'Invalid Discord webhook URL format. URL must start with https://discord.com/api/webhooks/ or https://discordapp.com/api/webhooks/')
        return

    # Color mapping
    color_map = {
        'CRITICAL': 15158332,  # Red
        'HIGH': 15844367,      # Orange
        'MEDIUM': 16776960,    # Yellow
        'LOW': 10070709        # Blue
    }
    
    # Build base embed
    try:
        embed = {
            'title': f'🚨 {alert.priority.value} Alert: {alert.ticker}',
            'description': alert.message or 'No message provided',
            'color': color_map.get(alert.priority.value, 10070709),
            'timestamp': alert.timestamp.isoformat() if hasattr(alert.timestamp, 'isoformat') else str(alert.timestamp)
        }
    except Exception as e:
        logger.error(f'Error building Discord embed: {e}', exc_info=True)
        return
    
    # Add type-specific fields
    fields = []
    
    if alert.alert_type == AlertType.EARNINGS_UPCOMING:
        fields = _build_earnings_fields(alert.details)
    elif alert.alert_type == AlertType.MAJOR_NEWS:
        fields = _build_news_fields(alert.details)
    elif alert.alert_type == AlertType.SEC_FILING:
        fields = _build_sec_fields(alert.details)
    elif alert.alert_type == AlertType.ECONOMIC_EVENT:
        fields = _build_economic_fields(alert.details)
    elif alert.alert_type in [AlertType.BUY_SIGNAL, AlertType.SELL_SIGNAL, 
                              AlertType.SPECULATION_OPPORTUNITY, AlertType.REVIEW_REQUIRED]:
        fields = _build_trading_decision_fields(alert.details)
    elif alert.alert_type == AlertType.TRADE_EXECUTED:
        # Build fields for trade execution alerts
        fields = []
        try:
            if alert.details:
                if alert.details.get('order_id'):
                    fields.append({
                        'name': '🆔 Order ID',
                        'value': str(alert.details['order_id']),
                        'inline': True
                    })
                if alert.details.get('price'):
                    try:
                        price = float(alert.details['price'])
                        fields.append({
                            'name': '💰 Price',
                            'value': f"${price:.4f}",
                            'inline': True
                        })
                    except (ValueError, TypeError):
                        fields.append({
                            'name': '💰 Price',
                            'value': str(alert.details['price']),
                            'inline': True
                        })
                if alert.details.get('quantity'):
                    try:
                        quantity = float(alert.details['quantity'])
                        fields.append({
                            'name': '📊 Quantity',
                            'value': f"{quantity:.6f}",
                            'inline': True
                        })
                    except (ValueError, TypeError):
                        fields.append({
                            'name': '📊 Quantity',
                            'value': str(alert.details['quantity']),
                            'inline': True
                        })
                if alert.details.get('direction'):
                    fields.append({
                        'name': '📈 Direction',
                        'value': str(alert.details['direction']),
                        'inline': True
                    })
                if alert.details.get('position_size'):
                    try:
                        pos_size = float(alert.details['position_size'])
                        fields.append({
                            'name': '💵 Position Size',
                            'value': f"${pos_size:,.2f}",
                            'inline': True
                        })
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.error(f'Error building TRADE_EXECUTED fields: {e}', exc_info=True)
    else:
        # Default fields for technical alerts
        if alert.confidence_score > 0:
            fields.append({
                'name': '📊 Confidence Score',
                'value': f'{alert.confidence_score:.1f}%',
                'inline': True
            })
        
        fields.append({
            'name': '🏷️ Alert Type',
            'value': alert.alert_type.value.replace('_', ' ').title(),
            'inline': True
        })
    
    # Add fields to embed
    if fields:
        embed['fields'] = fields
    
    # Add footer with alert type
    embed['footer'] = {
        'text': f"Alert Type: {alert.alert_type.value.replace('_', ' ').title()}"
    }
    
    payload = {
        'embeds': [embed]
    }

    try:
        requests = _get_requests()  # Lazy import
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f'✅ Successfully sent alert for {alert.ticker} to Discord.')
    except Exception as e:
        # Simplified error handling - requests.exceptions may not be loaded yet
        if 'Timeout' in str(type(e).__name__):
            logger.error(f'❌ Discord webhook timeout for {alert.ticker}. Webhook may be slow or unreachable.')
        elif 'ConnectionError' in str(type(e).__name__):
            logger.error(f'❌ Discord webhook connection error for {alert.ticker}: {e}')
        elif 'HTTPError' in str(type(e).__name__):
            logger.error(f'❌ Discord webhook HTTP error for {alert.ticker}: {e}')
        else:
            logger.error(f'❌ Failed to send Discord alert for {alert.ticker}: {e}', exc_info=True)

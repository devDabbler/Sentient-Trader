"""Helper utility functions."""

from datetime import datetime
from loguru import logger



def calculate_dte(expiry_date: str) -> int:
    """Calculate days to expiration"""
    try:
        exp_date = datetime.strptime(expiry_date, '%Y-%m-%d')
        return (exp_date - datetime.now()).days
    except Exception as e:
        logger.error(f"Error calculating DTE: {e}")
        return 0

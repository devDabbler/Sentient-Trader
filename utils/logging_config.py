"""Logging configuration for the trading application using Loguru."""

import sys
import os
from datetime import datetime
from pathlib import Path
import pytz
from loguru import logger

# PST timezone for log timestamps
PST = pytz.timezone('America/Los_Angeles')

# Track if logging has been configured to prevent duplicate handlers
_logging_configured = False

# Remove default handler
logger.remove()

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)


def get_broker_specific_log_file(default_log_file: str = "logs/sentient_trader.log") -> str:
    """
    Get broker-specific log file name based on DEFAULT_BROKER environment variable
    
    Args:
        default_log_file: Default log file path (e.g., "logs/sentient_trader.log")
        
    Returns:
        Broker-specific log file path if DEFAULT_BROKER is set, otherwise returns default
        
    Examples:
        >>> get_broker_specific_log_file("logs/sentient_trader.log")
        'logs/sentient_trader_tradier.log'  # if DEFAULT_BROKER=TRADIER
        'logs/sentient_trader_ibkr.log'     # if DEFAULT_BROKER=IBKR
        'logs/sentient_trader_kraken.log'   # if DEFAULT_BROKER=KRAKEN
        'logs/sentient_trader.log'          # if DEFAULT_BROKER not set
    """
    default_broker = os.getenv('DEFAULT_BROKER', '').upper()
    
    # If DEFAULT_BROKER is not set, return original filename
    if not default_broker or default_broker not in ['TRADIER', 'IBKR', 'KRAKEN']:
        return default_log_file
    
    # Extract directory, base name, and extension
    log_path = Path(default_log_file)
    directory = log_path.parent
    base_name = log_path.stem  # filename without extension
    extension = log_path.suffix  # .log
    
    # Create broker-specific filename
    broker_suffix = default_broker.lower()
    broker_specific_filename = f"{base_name}_{broker_suffix}{extension}"
    
    # Return full path
    return str(directory / broker_specific_filename)


def setup_logging(
    log_file: str = "logs/sentient_trader.log",
    level: str = "DEBUG",
    rotation: str = "500 MB",
    retention: str = "7 days",
    format_string: str = None,
    force: bool = False,
    console_output: bool = True
):
    """
    Configure Loguru logging for the entire application.
    
    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: Log rotation size (e.g., "500 MB", "1 day")
        retention: Log retention period (e.g., "7 days", "30 days")
        format_string: Custom format string (uses Loguru format syntax)
        force: Force re-configuration even if already configured (default: False)
        console_output: Enable console logging (default: True). Set False for pythonw.exe
    """
    global _logging_configured
    
    # Skip if already configured (prevents duplicate handlers on Streamlit reruns)
    if _logging_configured and not force:
        return
    
    # Remove all existing handlers to prevent duplicates
    logger.remove()
    
    # Default format with colors for console, simple for file
    if format_string is None:
        console_format = (
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} PST | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
    else:
        console_format = format_string
        file_format = format_string
    
    # Console handler (stdout) with colors - only if console_output is True
    # For background services (pythonw.exe), set console_output=False
    if console_output and sys.stdout is not None:
        try:
            logger.add(
                sys.stdout,
                level=level,
                format=console_format,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        except Exception:
            # If stdout is not available (e.g., pythonw.exe), skip console logging
            pass
    
    # File handler with rotation and retention
    # IMPORTANT: enqueue=True makes logging thread-safe and works better with pythonw.exe
    logger.add(
        log_file,
        level=level,
        format=file_format,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Thread-safe logging (important for background services)
        buffering=1     # Line buffering - flush after each line
    )
    
    # Suppress noisy third-party loggers by disabling their handlers
    # These libraries will still work, but won't spam logs
    _suppress_third_party_logs()
    
    # Mark as configured
    _logging_configured = True


def _suppress_third_party_logs():
    """Suppress verbose logging from third-party libraries."""
    import logging
    
    # Map of logger names to their desired levels
    suppress_loggers = {
        'urllib3': 'INFO',
        'requests': 'INFO',
        'httpcore': 'WARNING',
        'httpx': 'WARNING',
        'openai': 'WARNING',
        'openai._base_client': 'WARNING',
        'yfinance': 'WARNING',
        'peewee': 'WARNING',
        'discord': 'INFO',
        'discord.client': 'WARNING',
        'discord.gateway': 'WARNING',
        'asyncio': 'WARNING',
        'aiohttp': 'WARNING',
    }
    
    for logger_name, level in suppress_loggers.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level))

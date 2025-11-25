#!/usr/bin/env python3
"""
ORB FVG Scanner - Simple Runner

Standalone runner for the 15-Minute ORB + FVG strategy scanner.
Can be run directly or via systemd/Task Scheduler.

Usage:
    python windows_services/runners/run_orb_fvg_simple.py

Environment Variables:
    ORB_FVG_SCAN_INTERVAL - Scan interval in seconds (default: 60)
    ORB_FVG_MIN_CONFIDENCE - Minimum confidence for alerts (default: 70)
    DISCORD_WEBHOOK_URL - Discord webhook for alerts
"""

import sys
import os
import time
import signal
from pathlib import Path
from datetime import datetime, time as dt_time
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

# Setup logging
from utils.logging_config import setup_logging
log_file = project_root / 'logs' / 'orb_fvg_service.log'
setup_logging(log_file=str(log_file))

from loguru import logger
import pytz

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False


def is_trading_window(current_time: datetime) -> tuple[bool, str]:
    """
    Check if we're in the ORB FVG trading window
    
    Returns:
        (is_valid, status_message)
    """
    eastern = pytz.timezone('US/Eastern')
    
    if current_time.tzinfo is None:
        current_time = eastern.localize(current_time)
    else:
        current_time = current_time.astimezone(eastern)
    
    current_time_only = current_time.time()
    
    # Time windows
    pre_market = dt_time(4, 0)
    market_open = dt_time(9, 30)
    orb_end = dt_time(9, 45)
    optimal_window_end = dt_time(12, 30)
    market_close = dt_time(16, 0)
    
    if current_time_only < pre_market:
        return False, "OVERNIGHT"
    elif current_time_only < market_open:
        return False, "PRE-MARKET"
    elif current_time_only < orb_end:
        return True, "ORB_ESTABLISHING"
    elif current_time_only < optimal_window_end:
        return True, "OPTIMAL_TRADING_WINDOW"
    elif current_time_only < market_close:
        return True, "EXTENDED_WINDOW"
    else:
        return False, "AFTER_HOURS"


def get_scan_tickers() -> List[str]:
    """Get tickers to scan from service watchlist or defaults"""
    # First try service-specific watchlist (from control panel)
    try:
        import json
        watchlist_file = project_root / 'data' / 'service_watchlists.json'
        if watchlist_file.exists():
            with open(watchlist_file, 'r') as f:
                watchlists = json.load(f)
            
            service_config = watchlists.get('sentient-orb-fvg', {})
            tickers = service_config.get('tickers', [])
            
            if tickers:
                logger.info(f"Using {len(tickers)} tickers from service watchlist")
                return tickers
    except Exception as e:
        logger.warning(f"Could not load service watchlist: {e}")
    
    # Fallback to My Tickers
    try:
        from services.ticker_manager import TickerManager
        ticker_manager = TickerManager()
        tickers = ticker_manager.get_my_tickers()
        
        if tickers and len(tickers) > 0:
            logger.info(f"Using {len(tickers)} tickers from My Tickers watchlist")
            return tickers
    except Exception as e:
        logger.warning(f"Could not load watchlist: {e}")
    
    # Default ORB FVG tickers - liquid stocks with good intraday moves
    default_tickers = [
        # Major indices
        'SPY', 'QQQ', 'IWM',
        # Mega caps with volume
        'NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'META', 'AMZN', 'GOOGL',
        # High beta / momentum
        'PLTR', 'SOFI', 'COIN', 'MARA', 'RIOT', 'HOOD',
        # Additional liquid names
        'NFLX', 'CRM', 'SHOP', 'SQ', 'ROKU', 'SNAP',
        # Semiconductor
        'AVGO', 'MU', 'INTC', 'QCOM'
    ]
    
    logger.info(f"Using {len(default_tickers)} default tickers")
    return default_tickers


def run_scanner():
    """Main scanner loop"""
    global running
    
    logger.info("=" * 70)
    logger.info("📊 ORB FVG SCANNER STARTING")
    logger.info("=" * 70)
    
    # Configuration
    scan_interval = int(os.getenv('ORB_FVG_SCAN_INTERVAL', '60'))
    min_confidence = float(os.getenv('ORB_FVG_MIN_CONFIDENCE', '70'))
    
    logger.info(f"Configuration:")
    logger.info(f"  Scan interval: {scan_interval} seconds")
    logger.info(f"  Min confidence: {min_confidence}%")
    logger.info(f"  Log file: {log_file}")
    
    # Initialize strategy and alerts
    try:
        from services.orb_fvg_strategy import ORBFVGStrategy
        from services.orb_fvg_alerts import ORBFVGAlertManager
        
        strategy = ORBFVGStrategy(
            orb_minutes=15,
            min_gap_pct=0.2,
            min_volume_ratio=1.5,
            target_rr=2.0
        )
        
        alert_manager = ORBFVGAlertManager()
        
        logger.info("✅ Strategy and alert manager initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        return
    
    eastern = pytz.timezone('US/Eastern')
    scan_count = 0
    signals_today = 0
    last_trading_day = None
    scan_results = []  # Store results for control panel
    
    # Load Discord settings
    discord_enabled = True
    discord_min_confidence = min_confidence
    try:
        from service_config_loader import load_discord_settings, save_analysis_results
        discord_settings = load_discord_settings('sentient-orb-fvg')
        discord_enabled = discord_settings.get('enabled', True)
        discord_min_confidence = discord_settings.get('min_confidence', min_confidence)
        logger.info(f"Discord alerts: {'enabled' if discord_enabled else 'DISABLED'} (min conf: {discord_min_confidence}%)")
    except ImportError:
        logger.debug("service_config_loader not available")
        save_analysis_results = None
    except Exception as e:
        logger.warning(f"Could not load Discord settings: {e}")
        save_analysis_results = None
    
    while running:
        try:
            current_time = datetime.now(eastern)
            current_date = current_time.date()
            
            # Reset daily counter
            if last_trading_day != current_date:
                signals_today = 0
                last_trading_day = current_date
                logger.info(f"📅 New trading day: {current_date}")
            
            # Check trading window
            is_valid, status = is_trading_window(current_time)
            
            if not is_valid:
                logger.info(f"⏰ {status} ({current_time.strftime('%H:%M:%S')} ET) - waiting...")
                
                # Sleep longer outside market hours
                if status in ['OVERNIGHT', 'AFTER_HOURS']:
                    time.sleep(300)  # 5 minutes
                else:
                    time.sleep(60)  # 1 minute
                continue
            
            # Get tickers to scan
            tickers = get_scan_tickers()
            scan_count += 1
            
            logger.info(f"🔍 Scan #{scan_count} | {status} | {current_time.strftime('%H:%M:%S')} ET | {len(tickers)} tickers")
            
            # Scan each ticker
            signals_found = 0
            for ticker in tickers:
                try:
                    signal = strategy.analyze_ticker(ticker, current_time)
                    
                    if signal and signal.confidence >= min_confidence:
                        signals_found += 1
                        signals_today += 1
                        
                        logger.info(
                            f"🎯 SIGNAL #{signals_today}: {signal.symbol} {signal.signal_type} "
                            f"@ ${signal.entry_price:.2f} | Stop: ${signal.stop_loss:.2f} | "
                            f"Target: ${signal.target_price:.2f} | R:R {signal.risk_reward_ratio:.1f} | "
                            f"Conf: {signal.confidence:.1f}%"
                        )
                        
                        # Store result for control panel
                        scan_results.append({
                            'ticker': signal.symbol,
                            'signal': signal.signal_type,
                            'confidence': signal.confidence,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'target_price': signal.target_price,
                            'risk_reward': signal.risk_reward_ratio,
                            'timestamp': current_time.isoformat(),
                        })
                        
                        # Send Discord alert (if enabled and meets threshold)
                        if discord_enabled and signal.confidence >= discord_min_confidence:
                            try:
                                alert_manager.send_signal_alert(signal)
                            except Exception as e:
                                logger.error(f"Failed to send alert: {e}")
                        elif not discord_enabled:
                            logger.debug(f"Discord alerts disabled - skipping alert")
                            
                except Exception as e:
                    logger.debug(f"Error analyzing {ticker}: {e}")
                    continue
            
            # Summary
            if signals_found > 0:
                logger.info(f"✅ Scan complete: {signals_found} signals found | {signals_today} total today")
                
                # Save results to control panel (keep last 50)
                if save_analysis_results:
                    try:
                        save_analysis_results('sentient-orb-fvg', scan_results[-50:])
                    except Exception as e:
                        logger.debug(f"Could not save results: {e}")
            else:
                logger.debug(f"Scan complete: No signals")
            
            # Wait for next scan
            time.sleep(scan_interval)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            running = False
            break
            
        except Exception as e:
            logger.error(f"Error in scan loop: {e}", exc_info=True)
            time.sleep(60)
    
    logger.info("=" * 70)
    logger.info("📊 ORB FVG SCANNER STOPPED")
    logger.info(f"Total scans: {scan_count} | Signals today: {signals_today}")
    logger.info("=" * 70)


if __name__ == '__main__':
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        run_scanner()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

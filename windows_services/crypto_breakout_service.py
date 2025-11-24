"""
Crypto Breakout Monitor - Windows Service Wrapper

Runs the Crypto Breakout Monitor as a Windows service with:
- Auto-start on system boot
- Windows Event Log integration
- Service management via Services panel
- Automatic restart on failure

Monitors Kraken crypto markets for:
- High-volume breakouts
- EMA crossovers and trend changes
- RSI/MACD/Bollinger Band signals
- AI-confirmed opportunities

Installation:
    python windows_services\crypto_breakout_service.py install

Start/Stop:
    python windows_services\crypto_breakout_service.py start
    python windows_services\crypto_breakout_service.py stop

Removal:
    python windows_services\crypto_breakout_service.py remove
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.windows_service_base import WindowsServiceBase, install_service, check_pywin32_installed
from loguru import logger


class CryptoBreakoutService(WindowsServiceBase):
    """Windows service wrapper for Crypto Breakout Monitor"""
    
    _svc_name_ = "SentientCryptoBreakout"
    _svc_display_name_ = "Sentient Crypto Breakout Monitor"
    _svc_description_ = (
        "Monitors cryptocurrency markets on Kraken for breakout patterns. "
        "Detects high-volume moves, EMA crossovers, and AI-confirmed opportunities. "
        "Sends Discord alerts for instant trade execution."
    )
    
    def run_service(self):
        """Main service loop - runs the crypto breakout monitor continuously"""
        logger.info("=" * 70)
        logger.info("📈 CRYPTO BREAKOUT MONITOR SERVICE STARTED")
        logger.info("=" * 70)
        
        try:
            from services.crypto_breakout_monitor import CryptoBreakoutMonitor
            
            # Initialize monitor with sensible defaults
            monitor = CryptoBreakoutMonitor(
                scan_interval_seconds=300,  # 5 minutes
                min_score=70.0,
                min_confidence='HIGH',
                use_ai=True,
                use_watchlist=True,
                alert_cooldown_minutes=60,
                auto_add_to_watchlist=True
            )
            
            logger.info("Crypto Breakout Monitor initialized")
            logger.info("Scan interval: 5 minutes")
            logger.info("Min score: 70.0")
            logger.info("AI enabled: True")
            
            # Check if monitor has run_continuous method
            if hasattr(monitor, 'run_continuous'):
                # Monitor has built-in continuous run
                import threading
                
                def run_monitor():
                    """Run monitor in thread"""
                    try:
                        monitor.run_continuous()
                    except Exception as e:
                        logger.error(f"Monitor error: {e}", exc_info=True)
                
                # Start monitor in background thread
                monitor_thread = threading.Thread(target=run_monitor, daemon=True)
                monitor_thread.start()
                
                # Main service loop - check for stop requests
                while self.running and not self.is_stop_requested():
                    time.sleep(10)  # Check every 10 seconds
                    
                    # Verify monitor thread is still alive
                    if not monitor_thread.is_alive():
                        logger.error("Monitor thread died unexpectedly, restarting...")
                        monitor_thread = threading.Thread(target=run_monitor, daemon=True)
                        monitor_thread.start()
                
            else:
                # No run_continuous, implement our own loop
                logger.info("Implementing manual scan loop")
                
                while self.running and not self.is_stop_requested():
                    try:
                        logger.info("Starting scan cycle...")
                        
                        # Scan for opportunities
                        if hasattr(monitor, 'scan_and_alert'):
                            monitor.scan_and_alert()
                        elif hasattr(monitor, 'scan'):
                            opportunities = monitor.scan()
                            logger.info(f"Found {len(opportunities)} opportunities")
                        
                        # Wait for scan interval
                        logger.info("Scan complete, waiting for next cycle...")
                        time.sleep(300)  # 5 minutes
                        
                    except Exception as e:
                        logger.error(f"Error in scan cycle: {e}", exc_info=True)
                        time.sleep(60)  # Wait 1 min on error
            
            logger.info("Service stop requested, shutting down...")
            
            # Stop the monitor gracefully
            if hasattr(monitor, 'stop'):
                monitor.stop()
                
        except Exception as e:
            logger.error(f"Fatal error in service: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    if not check_pywin32_installed():
        sys.exit(1)
    
    install_service(CryptoBreakoutService)

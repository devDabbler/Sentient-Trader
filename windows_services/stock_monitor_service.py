R"""
Stock Informational Monitor - Windows Service Wrapper

Runs the Stock Informational Monitor as a Windows service with:
- Auto-start on system boot
- Windows Event Log integration
- Service management via Services panel
- Automatic restart on failure

Installation:
    python windows_services\stock_monitor_service.py install

Start/Stop:
    python windows_services\stock_monitor_service.py start
    python windows_services\stock_monitor_service.py stop

Removal:
    python windows_services\stock_monitor_service.py remove
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


class StockMonitorService(WindowsServiceBase):
    """Windows service wrapper for Stock Informational Monitor"""
    
    _svc_name_ = "SentientStockMonitor"
    _svc_display_name_ = "Sentient Stock Informational Monitor"
    _svc_description_ = (
        "Monitors stocks for trading opportunities without executing trades. "
        "Sends Discord alerts for high-probability setups based on technical analysis, "
        "ML predictions, and LLM validation."
    )
    
    def run_service(self):
        """Main service loop - runs the stock monitor continuously"""
        logger.info("=" * 70)
        logger.info("🔍 STOCK INFORMATIONAL MONITOR SERVICE STARTED")
        logger.info("=" * 70)
        
        try:
            import threading
            import asyncio
            
            def run_monitor():
                """Run monitor in thread with initialization"""
                try:
                    from services.stock_informational_monitor import get_stock_informational_monitor
                    
                    logger.info("Initializing stock monitor in background...")
                    
                    # Get monitor instance from config (this may take time)
                    monitor = get_stock_informational_monitor()
                    
                    logger.info(f"Monitor initialized with {len(monitor.watchlist)} symbols")
                    # Get scan interval from monitor attributes
                    scan_interval = getattr(monitor, 'scan_interval_minutes', 
                                           getattr(monitor, 'scan_interval', 30))
                    logger.info(f"Scan interval: {scan_interval} minutes")
                    
                    # Run continuous monitoring
                    if hasattr(monitor, 'run_continuous_async'):
                        asyncio.run(monitor.run_continuous_async())
                    else:
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
            
            logger.info("Service stop requested, shutting down...")
            # Monitor cleanup handled in background thread
                
        except Exception as e:
            logger.error(f"Fatal error in service: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    if not check_pywin32_installed():
        sys.exit(1)
    
    install_service(StockMonitorService)

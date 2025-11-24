R"""
DEX Launch Monitor - Windows Service Wrapper

Runs the DEX Launch Announcement Monitor as a Windows service with:
- Auto-start on system boot
- Windows Event Log integration
- Service management via Services panel
- Automatic restart on failure

Monitors:
1. Pump.fun (Solana) launches
2. DexScreener boosted tokens
3. Twitter mentions (if configured)
4. DEX Launch Hunter analysis

Installation:
    python windows_services\dex_launch_service.py install

Start/Stop:
    python windows_services\dex_launch_service.py start
    python windows_services\dex_launch_service.py stop

Removal:
    python windows_services\dex_launch_service.py remove
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.windows_service_base import WindowsServiceBase, install_service, check_pywin32_installed
from loguru import logger


class DEXLaunchService(WindowsServiceBase):
    """Windows service wrapper for DEX Launch Monitor"""
    
    _svc_name_ = "SentientDEXLaunch"
    _svc_display_name_ = "Sentient DEX Launch Monitor"
    _svc_description_ = (
        "Monitors crypto DEX launches and announcements in real-time. "
        "Tracks Pump.fun, DexScreener, and Twitter for new token launches. "
        "Analyzes tokens with AI and sends alerts for high-score opportunities."
    )
    
    def run_service(self):
        """Main service loop - runs the DEX launch monitor continuously"""
        logger.info("=" * 70)
        logger.info("🚀 DEX LAUNCH MONITOR SERVICE STARTED")
        logger.info("=" * 70)
        
        try:
            import threading
            
            # Async monitor loop with initialization
            async def monitor_loop():
                """Main monitoring loop with initialization"""
                from services.launch_announcement_monitor import get_announcement_monitor
                from services.dex_launch_hunter import get_dex_launch_hunter
                from services.alert_system import get_alert_system
                
                logger.info("Initializing services in background...")
                
                # Initialize services (this may take time)
                monitor = get_announcement_monitor(scan_interval=300)  # 5 min
                dex_hunter = get_dex_launch_hunter()
                alert_system = get_alert_system()
                
                logger.info("DEX Launch Monitor initialized")
                logger.info("Scan interval: 5 minutes")
                
                # Start announcement monitoring in background
                monitor_task = asyncio.create_task(monitor.start_monitoring())
                
                try:
                    while self.running and not self.is_stop_requested():
                        # Get announcements from last 10 minutes
                        recent = monitor.get_recent_announcements(minutes=10)
                        
                        if recent:
                            logger.info(f"📢 Processing {len(recent)} recent announcements...")
                            
                            for announcement in recent:
                                # Skip if already analyzed
                                if announcement.token_address in dex_hunter.discovered_tokens:
                                    continue
                                
                                logger.info(
                                    f"🔍 Analyzing: {announcement.token_symbol} "
                                    f"from {announcement.source}"
                                )
                                
                                try:
                                    # Analyze the token
                                    success, token = await dex_hunter._analyze_token(
                                        announcement.token_address,
                                        announcement.chain
                                    )
                                    
                                    if success and token:
                                        logger.info(
                                            f"✅ Analyzed: {token.symbol} - "
                                            f"Score: {token.composite_score:.1f}, "
                                            f"Risk: {token.risk_level.value}"
                                        )
                                        
                                        # Send alert for high scores
                                        if token.composite_score >= 60:
                                            alert_msg = (
                                                f"🚨 HIGH SCORE LAUNCH DETECTED!\n\n"
                                                f"Source: {announcement.source}\n"
                                                f"Token: {token.symbol}\n"
                                                f"Chain: {token.chain.value}\n"
                                                f"Score: {token.composite_score:.1f}/100\n"
                                                f"Liquidity: ${token.liquidity_usd:,.0f}\n"
                                                f"Risk: {token.risk_level.value}\n\n"
                                                f"DexScreener: https://dexscreener.com/{token.chain.value}/{token.contract_address}"
                                            )
                                            
                                            alert_system.send_alert(
                                                "LAUNCH_DETECTED",
                                                alert_msg,
                                                priority="HIGH" if token.composite_score >= 70 else "MEDIUM"
                                            )
                                
                                except Exception as e:
                                    logger.error(f"Error analyzing {announcement.token_symbol}: {e}")
                        
                        # Show stats
                        stats = monitor.get_stats()
                        logger.info(
                            f"📊 Stats: {stats['total_announcements']} total, "
                            f"{stats['last_30_min']} in last 30min"
                        )
                        
                        # Wait 5 minutes before next check
                        await asyncio.sleep(300)
                        
                except Exception as e:
                    logger.error(f"Error in monitor loop: {e}", exc_info=True)
                finally:
                    monitor.stop_monitoring()
                    monitor_task.cancel()
            
            # Run async loop in background thread
            def run_async_loop():
                """Run the async loop in a thread"""
                try:
                    asyncio.run(monitor_loop())
                except Exception as e:
                    logger.error(f"Async loop error: {e}", exc_info=True)
            
            # Start monitor in background thread
            monitor_thread = threading.Thread(target=run_async_loop, daemon=True)
            monitor_thread.start()
            logger.info("Monitor thread started")
            
            # Main service loop - check for stop requests
            while self.running and not self.is_stop_requested():
                time.sleep(10)  # Check every 10 seconds
                
                # Verify monitor thread is still alive
                if not monitor_thread.is_alive():
                    logger.error("Monitor thread died unexpectedly, restarting...")
                    monitor_thread = threading.Thread(target=run_async_loop, daemon=True)
                    monitor_thread.start()
            
            logger.info("Service stop requested, shutting down...")
            # Monitor cleanup handled in async loop
            
        except Exception as e:
            logger.error(f"Fatal error in service: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    if not check_pywin32_installed():
        sys.exit(1)
    
    install_service(DEXLaunchService)

"""
Background Launch Announcement Monitor - Runs continuously

Monitors:
1. Pump.fun (Solana) - Every 5 min
2. DexScreener boosted - Every 5 min
3. Twitter mentions - Every 5 min (if configured)
4. Your DEX Launch Hunter - Every 30 min

Usage: python run_launch_monitor_background.py
"""

import asyncio
import sys
import os
from datetime import datetime
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.launch_announcement_monitor import get_announcement_monitor
from services.dex_launch_hunter import get_dex_launch_hunter
from services.alert_system import get_alert_system


logger.remove()
logger.add(
    "logs/launch_monitor.log",
    rotation="100 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
)
logger.add(sys.stdout, level="INFO", colorize=True)


async def monitor_loop():
    """Main monitoring loop"""
    logger.info("=" * 70)
    logger.info("🚀 LAUNCH ANNOUNCEMENT MONITOR STARTED")
    logger.info("=" * 70)
    
    # Get monitor instance (5 min scan interval)
    monitor = get_announcement_monitor(scan_interval=300)
    
    # Get alert system for notifications
    alert_system = get_alert_system()
    
    # Get DEX hunter for full analysis
    dex_hunter = get_dex_launch_hunter()
    
    # Start announcement monitoring in background
    monitor_task = asyncio.create_task(monitor.start_monitoring())
    
    # Main loop: check for new announcements and analyze them
    try:
        while True:
            # Get announcements from last 10 minutes
            recent = monitor.get_recent_announcements(minutes=10)
            
            if recent:
                logger.info(f"📢 Processing {len(recent)} recent announcements...")
                
                for announcement in recent:
                    # Skip if already analyzed
                    if announcement.token_address in dex_hunter.discovered_tokens:
                        continue
                    
                    logger.info(
                        f"🔍 Analyzing new announcement: {announcement.token_symbol} "
                        f"from {announcement.source}"
                    )
                    
                    try:
                        # Analyze the token with DEX hunter
                        success, token = await dex_hunter._analyze_token(
                            announcement.token_address,
                            announcement.chain
                        )
                        
                        if success and token:
                            logger.info(
                                f"✅ Token analyzed: {token.symbol} - "
                                f"Score: {token.composite_score:.1f}, "
                                f"Risk: {token.risk_level.value}"
                            )
                            
                            # If high score, send alert!
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
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down monitor...")
        monitor.stop_monitoring()
        monitor_task.cancel()
    except Exception as e:
        logger.error(f"Fatal error in monitor loop: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Starting Launch Announcement Monitor...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        logger.info("\nMonitor stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

"""
DEX Launch Monitor - SIMPLE VERSION
No threading, no timeouts - straightforward execution with async support
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file (CRITICAL for Discord alerts, API keys, etc.)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Add venv site-packages (handle both Windows and Linux)
if sys.platform == "win32":
    venv_site_packages = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
else:
    # Linux/Unix
    import glob
    venv_pattern = PROJECT_ROOT / "venv" / "lib" / "python3*" / "site-packages"
    venv_matches = glob.glob(str(venv_pattern))
    venv_site_packages = Path(venv_matches[0]) if venv_matches else None
    
if venv_site_packages and venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

# Setup logging
from loguru import logger

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

logger.remove()
# Add stderr handler (so we see logs in terminal)
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True
)
# Add file handler
logger.add(
    str(log_dir / "dex_launch_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False,
    buffering=1
)

logger.info("=" * 70)
logger.info("🚀 DEX LAUNCH - SIMPLE RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Python: {sys.executable}")
logger.info(f"✓ User: {os.getenv('USERNAME', 'UNKNOWN')}")

os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress verbose HTTP error output
import logging
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger.info("")
logger.info("Starting imports (be patient)...")

try:
    import_start = time.time()
    
    logger.info("DEBUG: Step 1 - Importing DEX launch services...")
    sys.stdout.write("DEBUG: Importing DEX launch services...\n")
    sys.stdout.flush()
    from services.launch_announcement_monitor import get_announcement_monitor
    logger.info("DEBUG: Step 2 - launch_announcement_monitor imported")
    from services.dex_launch_hunter import get_dex_launch_hunter
    logger.info("DEBUG: Step 3 - dex_launch_hunter imported")
    from services.alert_system import get_alert_system
    logger.info("DEBUG: Step 4 - alert_system imported")
    
    logger.info(f"✓ Imported in {time.time() - import_start:.1f}s")
    sys.stdout.write("DEBUG: Import complete\n")
    sys.stdout.flush()
    
    logger.info("DEBUG: Step 5 - Creating announcement monitor...")
    sys.stdout.write("DEBUG: Initializing services...\n")
    sys.stdout.flush()
    monitor = get_announcement_monitor(scan_interval=300)
    logger.info("DEBUG: Step 6 - Announcement monitor created")
    sys.stdout.write("DEBUG: Announcement monitor created\n")
    sys.stdout.flush()
    
    logger.info("DEBUG: Step 7 - Creating DEX hunter...")
    dex_hunter = get_dex_launch_hunter()
    logger.info("DEBUG: Step 8 - DEX hunter created")
    sys.stdout.write("DEBUG: DEX hunter created\n")
    sys.stdout.flush()
    
    logger.info("DEBUG: Step 9 - Creating alert system...")
    alert_system = get_alert_system()
    logger.info("DEBUG: Step 10 - Alert system created")
    logger.info("✓ Services initialized")
    sys.stdout.write("DEBUG: All services initialized\n")
    sys.stdout.flush()
    
    # Print SERVICE READY to both stdout AND logger
    logger.info("")
    logger.info("=" * 70)
    service_ready_msg = f"🚀 SERVICE READY (total startup: {time.time() - import_start:.1f}s)"
    logger.info(service_ready_msg)
    print(f"\n{'='*70}")
    print(service_ready_msg)
    print(f"{'='*70}\n")
    sys.stdout.flush()
    logger.info("=" * 70)
    logger.info("")
    
    # CRITICAL DEBUG: Verify we're still executing
    logger.info("DEBUG: After SERVICE READY print - execution continuing...")
    print("DEBUG: After SERVICE READY print - execution continuing...", file=sys.stdout, flush=True)
    sys.stdout.flush()
    
    # Write status file for batch script verification
    logger.info("DEBUG: About to write status file...")
    print("DEBUG: About to write status file...", file=sys.stdout, flush=True)
    sys.stdout.flush()
    try:
        status_file = PROJECT_ROOT / "logs" / ".dex_launch_ready"
        status_file.write_text(f"SERVICE READY at {time.time()}")
        logger.info("DEBUG: Status file written successfully")
        print("DEBUG: Status file written successfully", file=sys.stdout, flush=True)
    except Exception as e:
        logger.error(f"DEBUG: Error writing status file: {e}", exc_info=True)
        print(f"DEBUG: Error writing status file: {e}", file=sys.stdout, flush=True)
    sys.stdout.flush()
    
    logger.info("DEBUG: Status file written, about to define monitor_loop()...")
    print("DEBUG: Status file written, about to define monitor_loop()...", file=sys.stdout, flush=True)
    sys.stdout.flush()
    
    # Async monitoring loop
    print("DEBUG: About to define monitor_loop() function...", file=sys.stdout, flush=True)
    logger.info("DEBUG: About to define monitor_loop() function...")
    sys.stdout.flush()
    
    async def monitor_loop():
        print("DEBUG: Inside monitor_loop() - first line executed!", file=sys.stdout, flush=True)
        sys.stdout.flush()
        
        # Quick heartbeat to verify async is working
        print("DEBUG: Starting heartbeat task to verify event loop...", file=sys.stdout, flush=True)
        
        async def heartbeat():
            for i in range(3):
                print(f"DEBUG: Heartbeat {i+1}/3", file=sys.stdout, flush=True)
                await asyncio.sleep(0.5)
            print("DEBUG: Heartbeat complete!", file=sys.stdout, flush=True)
        
        # Test simple async operation first
        print("DEBUG: Calling heartbeat()...", file=sys.stdout, flush=True)
        await heartbeat()
        print("DEBUG: Heartbeat finished, continuing...", file=sys.stdout, flush=True)
        sys.stdout.flush()
        
        # Test if we can continue executing
        print("DEBUG: Testing basic print after first line...", file=sys.stdout, flush=True)
        
        # Test if logger works in async context
        print("DEBUG: About to call logger.info('=' * 70)...", file=sys.stdout, flush=True)
        sys.stdout.flush()
        try:
            # Use print first to see if we even get here
            print("=" * 70, file=sys.stdout, flush=True)
            logger.info("=" * 70)
            print("DEBUG: First logger.info completed!", file=sys.stdout, flush=True)
        except Exception as e:
            import traceback
            print(f"DEBUG: logger.info FAILED with: {e}", file=sys.stdout, flush=True)
            traceback.print_exc()
        sys.stdout.flush()
        
        print("DEBUG: About to log 'Starting DEX launch monitoring...'", file=sys.stdout, flush=True)
        print("Starting DEX launch monitoring (announcements + active scanning)...", flush=True)
        logger.info("Starting DEX launch monitoring (announcements + active scanning)...")
        print("DEBUG: Second logger.info completed!", file=sys.stdout, flush=True)
        logger.info("=" * 70)
        print("DEBUG: Third logger.info completed!", file=sys.stdout, flush=True)
        sys.stdout.flush()
        
        # Start announcement monitoring in background
        print("DEBUG: About to create monitor_task...", file=sys.stdout, flush=True)
        sys.stdout.flush()
        print("DEBUG: About to call logger.info for Creating monitor_task...", file=sys.stdout, flush=True)
        sys.stdout.flush()
        logger.info("DEBUG: Creating monitor_task...")
        print("DEBUG: logger.info completed for Creating monitor_task", file=sys.stdout, flush=True)
        sys.stdout.flush()
        
        try:
            print("DEBUG: Inside try block, about to define start_monitor_with_timeout()...", file=sys.stdout, flush=True)
            sys.stdout.flush()
            
            # Wrap in timeout to catch hangs
            async def start_monitor_with_timeout():
                print("DEBUG: start_monitor_with_timeout() called!", file=sys.stdout, flush=True)
                logger.info("DEBUG: Calling monitor.start_monitoring()...")
                await monitor.start_monitoring()
            
            print("DEBUG: start_monitor_with_timeout defined, about to create_task()...", file=sys.stdout, flush=True)
            sys.stdout.flush()
            
            monitor_task = asyncio.create_task(start_monitor_with_timeout())
            
            print("DEBUG: create_task() completed!", file=sys.stdout, flush=True)
            sys.stdout.flush()
            logger.info("DEBUG: monitor_task created successfully")
        except Exception as e:
            print(f"DEBUG: ERROR in task creation: {e}", file=sys.stdout, flush=True)
            import traceback
            traceback.print_exc()
            logger.error(f"ERROR creating monitor_task: {e}", exc_info=True)
            raise
        
        print("DEBUG: About to enter main loop...", file=sys.stdout, flush=True)
        sys.stdout.flush()
        logger.info("DEBUG: monitor_task created, entering main loop...")
        
        scan_counter = 0
        
        print("DEBUG: About to enter while True loop...", file=sys.stdout, flush=True)
        sys.stdout.flush()
        
        try:
            while True:
                scan_counter += 1
                print(f"DEBUG: Scan cycle #{scan_counter} starting...", file=sys.stdout, flush=True)
                sys.stdout.flush()
                logger.info(f"🔄 Scan cycle #{scan_counter} starting...")
                
                # ===== PART 1: Active DEX Scanning =====
                # Run the DEX hunter's own scanner to find new launches
                print("DEBUG: About to run active DEX scan...", file=sys.stdout, flush=True)
                sys.stdout.flush()
                logger.info("🔍 Running active DEX scan...")
                try:
                    print("DEBUG: Calling dex_hunter._scan_for_launches() with 120s timeout...", file=sys.stdout, flush=True)
                    sys.stdout.flush()
                    # Add timeout to prevent hanging
                    await asyncio.wait_for(
                        dex_hunter._scan_for_launches(),
                        timeout=120  # 2 minute timeout
                    )
                    print("DEBUG: _scan_for_launches() returned successfully!", file=sys.stdout, flush=True)
                    logger.info("DEBUG: Active DEX scan completed")
                    
                    # Check for high-score tokens from active scan
                    discovered_count = len(dex_hunter.discovered_tokens)
                    print(f"DEBUG: Checking {discovered_count} discovered tokens for high scores...", file=sys.stdout, flush=True)
                    
                    for addr, token in dex_hunter.discovered_tokens.items():
                        if hasattr(token, '_alerted'):
                            continue  # Already alerted
                        
                        if token.composite_score >= 60:
                            print(f"DEBUG: High score token found: {token.symbol} ({token.composite_score})", file=sys.stdout, flush=True)
                            alert_msg = (
                                f"🚀 NEW TOKEN DISCOVERED!\n\n"
                                f"Token: {token.symbol}\n"
                                f"Chain: {token.chain.value}\n"
                                f"Score: {token.composite_score:.1f}/100\n"
                                f"Liquidity: ${token.liquidity_usd:,.0f}\n"
                                f"Volume 24h: ${token.volume_24h:,.0f}\n"
                                f"Age: {token.age_hours:.1f}h\n"
                                f"Risk: {token.risk_level.value}\n\n"
                                f"🔗 https://dexscreener.com/{token.chain.value}/{token.contract_address}"
                            )
                            
                            alert_system.send_alert(
                                "DEX_DISCOVERY",
                                alert_msg,
                                priority="HIGH" if token.composite_score >= 70 else "MEDIUM",
                                metadata={'symbol': token.symbol, 'score': token.composite_score}
                            )
                            token._alerted = True
                            logger.info(f"✅ Alert sent for {token.symbol} (Score: {token.composite_score:.1f})")
                    
                    print("DEBUG: High score check complete", file=sys.stdout, flush=True)
                
                except asyncio.TimeoutError:
                    print("DEBUG: _scan_for_launches() TIMEOUT!", file=sys.stdout, flush=True)
                    logger.warning("⚠️ Active DEX scan timed out after 120s, continuing...")
                except Exception as e:
                    print(f"DEBUG: _scan_for_launches() ERROR: {e}", file=sys.stdout, flush=True)
                    logger.error(f"Active scan error: {e}")
                
                # ===== PART 2: Process Announcements =====
                print("DEBUG: Getting recent announcements...", file=sys.stdout, flush=True)
                # Get recent announcements from various sources
                recent = monitor.get_recent_announcements(minutes=10)
                print(f"DEBUG: Got {len(recent) if recent else 0} recent announcements", file=sys.stdout, flush=True)
                
                if recent:
                    logger.info(f"📢 Processing {len(recent)} announcements...")
                    
                    for announcement in recent:
                        if announcement.token_address.lower() in dex_hunter.discovered_tokens:
                            continue
                        
                        logger.info(f"🔍 Analyzing: {announcement.token_symbol} from {announcement.source}")
                        
                        try:
                            success, token = await dex_hunter.analyze_token(
                                announcement.token_address,
                                announcement.chain
                            )
                            
                            if success and token and token.composite_score >= 60:
                                alert_msg = (
                                    f"🚨 HIGH SCORE LAUNCH!\n\n"
                                    f"Source: {announcement.source}\n"
                                    f"Token: {token.symbol}\n"
                                    f"Chain: {token.chain.value}\n"
                                    f"Score: {token.composite_score:.1f}/100\n"
                                    f"Liquidity: ${token.liquidity_usd:,.0f}\n"
                                    f"Risk: {token.risk_level.value}\n\n"
                                    f"🔗 https://dexscreener.com/{token.chain.value}/{token.contract_address}"
                                )
                                
                                alert_system.send_alert(
                                    "LAUNCH_DETECTED",
                                    alert_msg,
                                    priority="HIGH" if token.composite_score >= 70 else "MEDIUM",
                                    metadata={'symbol': token.symbol, 'score': token.composite_score}
                                )
                                logger.info(f"✅ Alert sent for {token.symbol} (Score: {token.composite_score:.1f})")
                        
                        except Exception as e:
                            logger.error(f"Error analyzing {announcement.token_symbol}: {e}")
                
                # ===== Show Stats =====
                print("DEBUG: Gathering stats...", file=sys.stdout, flush=True)
                stats = monitor.get_stats()
                discovered_count = len(dex_hunter.discovered_tokens)
                high_score_count = sum(1 for t in dex_hunter.discovered_tokens.values() if t.composite_score >= 60)
                
                print(f"DEBUG: Stats - discovered={discovered_count}, high_score={high_score_count}", file=sys.stdout, flush=True)
                logger.info(
                    f"📊 Stats: {discovered_count} tokens discovered, "
                    f"{high_score_count} high-score, "
                    f"{stats.get('total_announcements', 0)} announcements"
                )
                
                print("DEBUG: 🎉 SCAN CYCLE COMPLETE! Sleeping 5 minutes...", file=sys.stdout, flush=True)
                logger.info(f"💤 Sleeping 5 minutes until next scan...")
                await asyncio.sleep(300)  # 5 minutes
                print("DEBUG: Woke up from sleep, starting next cycle...", file=sys.stdout, flush=True)
                
        except Exception as e:
            logger.error(f"Monitor loop error: {e}", exc_info=True)
        finally:
            monitor.stop_monitoring()
            monitor_task.cancel()
    
    # Run the async loop
    print("DEBUG: monitor_loop() function defined, about to call asyncio.run()...", file=sys.stdout, flush=True)
    logger.info("DEBUG: monitor_loop() function defined, about to call asyncio.run()...")
    sys.stdout.flush()
    
    # Test if asyncio works at all
    print("DEBUG: Testing asyncio with simple function...", file=sys.stdout, flush=True)
    async def test_async():
        print("DEBUG: Test async function executed!", file=sys.stdout, flush=True)
        logger.info("DEBUG: Test async function executed!")
        return True
    
    logger.info("DEBUG: Testing asyncio with simple function...")
    try:
        print("DEBUG: Calling asyncio.run(test_async())...", file=sys.stdout, flush=True)
        result = asyncio.run(test_async())
        print(f"DEBUG: Async test passed: {result}", file=sys.stdout, flush=True)
        logger.info(f"DEBUG: Async test passed: {result}")
    except Exception as e:
        print(f"DEBUG: Async test failed: {e}", file=sys.stdout, flush=True)
        logger.error(f"DEBUG: Async test failed: {e}", exc_info=True)
        raise
    
    print("DEBUG: About to start async monitor_loop()...", file=sys.stdout, flush=True)
    logger.info("DEBUG: About to start async monitor_loop()...")
    print("DEBUG: Calling asyncio.run(monitor_loop()) now...", file=sys.stdout, flush=True)
    logger.info("DEBUG: Calling asyncio.run(monitor_loop()) now...")
    sys.stdout.flush()  # Force flush before async call
    
    try:
        asyncio.run(monitor_loop())
    except Exception as e:
        logger.error(f"ERROR in asyncio.run(): {e}", exc_info=True)
        sys.stdout.flush()
        raise

except KeyboardInterrupt:
    logger.info("Service stopped by user")
except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    sys.exit(1)

"""
Stock Monitor Service Runner - SIMPLE VERSION
No threading, no timeouts - just let Python do its thing naturally
Sometimes timeout mechanisms cause more problems than they solve
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# CRITICAL: Set working directory FIRST
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)

# Add to Python path
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

# Setup logging BEFORE any service imports
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
# Add file handler with explicit buffering
logger.add(
    str(log_dir / "stock_monitor_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False,  # Synchronous writes to ensure logs appear immediately
    buffering=1  # Line buffering
)

logger.info("=" * 70)
logger.info("📊 STOCK MONITOR - ENHANCED RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Project root: {PROJECT_ROOT}")
logger.info(f"✓ Python: {sys.executable}")
logger.info(f"✓ User: {os.getenv('USER', os.getenv('USERNAME', 'UNKNOWN'))}")
logger.info(f"✓ Features: Stats tracking, Health monitoring, Circuit breaker, Auto-recovery")

# Force flush stdout/stderr for systemd
sys.stdout.flush()
sys.stderr.flush()

# Set environment
os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress verbose yfinance/urllib HTTP error output to stderr
# These errors are non-fatal but clutter the output
import logging
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger.info("")
logger.info("Starting imports (may take 30-60 seconds in Task Scheduler)...")
logger.info("Be patient - network libraries need time to initialize...")

# Just import directly - no threading, no timeouts
# This may take a while but should eventually work
import_start = time.time()

try:
    logger.info("  Step 1/3: Importing stock_informational_monitor...")
    import sys
    sys.stdout.write("DEBUG: About to import stock_informational_monitor\n")
    sys.stdout.flush()
    from services.stock_informational_monitor import get_stock_informational_monitor
    sys.stdout.write("DEBUG: Import statement completed\n")
    sys.stdout.flush()
    logger.info(f"  ✓ Import completed in {time.time() - import_start:.1f}s")
    sys.stdout.write("DEBUG: Logger message written\n")
    sys.stdout.flush()
    
    logger.info("")
    logger.info("  Step 2/3: Creating monitor instance...")
    sys.stdout.write("DEBUG: About to create monitor instance\n")
    sys.stdout.flush()
    instance_start = time.time()
    monitor = get_stock_informational_monitor()
    sys.stdout.write(f"DEBUG: get_stock_informational_monitor() returned\n")
    sys.stdout.flush()
    
    # Force stdout/stderr flush before logger calls
    sys.stdout.flush()
    sys.stderr.flush()
    
    sys.stdout.write(f"DEBUG: About to log instance created message\n")
    sys.stdout.flush()
    logger.info(f"  ✓ Instance created in {time.time() - instance_start:.1f}s")
    sys.stdout.write(f"DEBUG: Logged instance created\n")
    sys.stdout.flush()
    
    # ============================================================
    # Load watchlist from Control Panel config (if available)
    # ============================================================
    save_analysis_results = None
    try:
        from service_config_loader import load_service_watchlist, load_discord_settings, save_analysis_results
        
        custom_watchlist = load_service_watchlist('sentient-stock-monitor')
        if custom_watchlist:
            logger.info(f"  📋 Control Panel watchlist: {len(custom_watchlist)} tickers")
            # Override monitor's watchlist
            if hasattr(monitor, 'watchlist'):
                monitor.watchlist = custom_watchlist
            elif hasattr(monitor, 'tickers'):
                monitor.tickers = custom_watchlist
        
        # Load Discord settings
        discord_settings = load_discord_settings('sentient-stock-monitor')
        if discord_settings.get('enabled') is False:
            logger.info("  🔕 Discord alerts DISABLED via Control Panel")
            os.environ['DISCORD_ALERTS_DISABLED'] = 'true'
        else:
            min_conf = discord_settings.get('min_confidence', 70)
            logger.info(f"  🔔 Discord alerts enabled (min confidence: {min_conf}%)")
            os.environ['DISCORD_MIN_CONFIDENCE'] = str(min_conf)
        
        # Load discovery configuration
        try:
            from windows_services.runners.service_discovery_config import apply_config_to_monitor
            apply_config_to_monitor(monitor)
            logger.info("  🔍 Discovery config loaded from Control Panel")
        except ImportError:
            logger.debug("Discovery config not available - using defaults")
        except Exception as e:
            logger.warning(f"Could not load discovery config: {e}")
            
    except ImportError:
        logger.debug("service_config_loader not available - using defaults")
        save_analysis_results = None
    except Exception as e:
        logger.warning(f"Could not load Control Panel config: {e}")
        save_analysis_results = None
    
    sys.stdout.write(f"DEBUG: About to access monitor.watchlist\n")
    sys.stdout.flush()
    watchlist_len = len(monitor.watchlist) if hasattr(monitor, 'watchlist') else 0
    sys.stdout.write(f"DEBUG: Watchlist has {watchlist_len} symbols\n")
    sys.stdout.flush()
    logger.info(f"  ✓ Watchlist: {watchlist_len} symbols")
    
    # Get scan interval
    sys.stdout.write(f"DEBUG: About to get scan_interval\n")
    sys.stdout.flush()
    scan_interval = getattr(monitor, 'scan_interval_minutes', 30)
    sys.stdout.write(f"DEBUG: scan_interval = {scan_interval}\n")
    sys.stdout.flush()
    logger.info(f"  ✓ Scan interval: {scan_interval} minutes")
    
    sys.stdout.write(f"DEBUG: About to print SERVICE READY\n")
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
    
    # Write status file for batch script verification
    status_file = PROJECT_ROOT / "logs" / ".stock_monitor_ready"
    status_file.write_text(f"SERVICE READY at {time.time()}")
    
    # Step 3/3: Start monitoring loop with enhanced error handling
    logger.info("=" * 70)
    logger.info("Starting monitoring service with resilience features...")
    logger.info("=" * 70)
    
    # Use the main run_continuous() method which includes all resilience features
    if hasattr(monitor, 'run_continuous'):
        logger.info("✅ Using monitor.run_continuous() method with stats tracking")
        try:
            monitor.run_continuous()
        except KeyboardInterrupt:
            logger.info("✅ Service cleanly shut down")
            if hasattr(monitor, '_print_final_stats'):
                monitor._print_final_stats()
            sys.exit(0)
        
    elif hasattr(monitor, 'run_continuous_async'):
        logger.info("Using monitor.run_continuous_async() method")
        import asyncio
        asyncio.run(monitor.run_continuous_async())
        
    else:
        # Fallback: Manual scan loop (shouldn't reach this with enhanced monitor)
        logger.info("⚠️  Fallback: Using manual scan loop (enhanced monitor.run_continuous() should be available)")
        scan_count = 0
        all_results = []
        
        while True:
            try:
                scan_count += 1
                logger.info(f"Starting scan #{scan_count}...")
                scan_start = time.time()
                
                opportunities = None
                if hasattr(monitor, 'scan_all_tickers'):
                    opportunities = monitor.scan_all_tickers()
                elif hasattr(monitor, 'scan'):
                    opportunities = monitor.scan()
                elif hasattr(monitor, 'scan_and_alert'):
                    opportunities = monitor.scan_and_alert()
                else:
                    logger.warning("No scan method found - service may not be functional")
                
                # Save results to control panel
                if opportunities and save_analysis_results:
                    try:
                        # Convert StockOpportunity objects to dicts
                        results = []
                        for opp in opportunities:
                            if hasattr(opp, 'to_dict'):
                                result = opp.to_dict()
                            elif isinstance(opp, dict):
                                result = opp
                            else:
                                result = {
                                    'ticker': getattr(opp, 'symbol', 'N/A'),
                                    'signal': getattr(opp, 'opportunity_type', 'SIGNAL'),
                                    'confidence': int(getattr(opp, 'ensemble_score', 0)),
                                    'price': getattr(opp, 'price', 0),
                                    'timestamp': datetime.now().isoformat()
                                }
                            results.append(result)
                        
                        all_results.extend(results)
                        # Keep last 100 results (increased from 50)
                        all_results = all_results[-100:]
                        
                        save_analysis_results('sentient-stock-monitor', all_results)
                        logger.debug(f"Saved {len(results)} results to control panel")
                    except Exception as e:
                        logger.debug(f"Could not save results: {e}")
                
                scan_duration = time.time() - scan_start
                logger.info(f"Scan #{scan_count} complete ({scan_duration:.1f}s)")
                logger.info(f"Next scan in {scan_interval} minutes...")
                logger.info("")
                
                time.sleep(scan_interval * 60)
                
            except KeyboardInterrupt:
                logger.info("Scan interrupted by user")
                raise
            except Exception as e:
                logger.error(f"Scan error: {e}", exc_info=True)
                logger.info("Waiting 60s before retry...")
                time.sleep(60)

except KeyboardInterrupt:
    logger.info("Service stopped by user")
    sys.exit(0)

except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    logger.error("Service failed to start or crashed during operation")
    sys.exit(1)

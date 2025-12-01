"""
Monitor and maintain Analysis Queue Processor service.

This script:
1. Checks if the processor is running
2. Restarts it if it crashed
3. Reports on queue status
4. Ensures consistent updates to the UI

Usage:
    python monitor_analysis_processor.py
    
Or add to Windows Task Scheduler to run every 5 minutes:
    SchTasks /Create /TN "Monitor Analysis Processor" /TR "python monitor_analysis_processor.py" /SC MINUTE /MO 5
"""

import sys
import os
import json
import psutil
import time
from pathlib import Path
from datetime import datetime
from loguru import logger

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Configure logging
LOG_FILE = PROJECT_ROOT / "logs" / "monitor_analysis_processor.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(str(LOG_FILE), rotation="5 MB", retention="7 days", level="DEBUG")

# Files
ANALYSIS_REQUESTS_FILE = PROJECT_ROOT / "data" / "analysis_requests.json"
ANALYSIS_RESULTS_FILE = PROJECT_ROOT / "data" / "analysis_results.json"
PROCESSOR_SCRIPT = PROJECT_ROOT / "windows_services" / "runners" / "run_analysis_queue_processor.py"
STATUS_FILE = PROJECT_ROOT / "data" / "analysis_processor_status.json"


def find_processor_process():
    """Find the analysis processor process"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('run_analysis_queue_processor.py' in str(cmd) for cmd in cmdline):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Error finding processor process: {e}")
    return None


def is_processor_running():
    """Check if analysis processor is running"""
    proc = find_processor_process()
    if proc:
        try:
            if proc.is_running():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def start_processor():
    """Start the analysis queue processor"""
    try:
        logger.info(f"🚀 Starting Analysis Queue Processor...")
        logger.info(f"   Script: {PROCESSOR_SCRIPT}")
        
        # Use the same Python executable as this script
        python_exe = sys.executable
        
        # Start as detached process on Windows
        if sys.platform == 'win32':
            import subprocess
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            CREATE_NO_WINDOW = 0x08000000
            
            subprocess.Popen(
                [python_exe, str(PROCESSOR_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
            )
        else:
            import subprocess
            subprocess.Popen(
                [python_exe, str(PROCESSOR_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        logger.info("✅ Process started successfully")
        time.sleep(2)  # Give it time to start
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to start processor: {e}")
        return False


def get_queue_status():
    """Get current queue status"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "processor_running": is_processor_running(),
        "pending_requests": 0,
        "completed_requests": 0,
        "latest_results_count": 0,
        "latest_results_time": None
    }
    
    try:
        # Check requests
        if ANALYSIS_REQUESTS_FILE.exists():
            with open(ANALYSIS_REQUESTS_FILE, 'r') as f:
                requests = json.load(f)
            status["pending_requests"] = len([r for r in requests if r.get("status") == "pending"])
            status["completed_requests"] = len([r for r in requests if r.get("status") == "complete"])
        
        # Check results
        if ANALYSIS_RESULTS_FILE.exists():
            with open(ANALYSIS_RESULTS_FILE, 'r') as f:
                results = json.load(f)
            
            latest = results.get("queue_latest", {})
            status["latest_results_count"] = latest.get("count", 0)
            status["latest_results_time"] = latest.get("updated", None)
    
    except Exception as e:
        logger.warning(f"Error getting queue status: {e}")
    
    return status


def save_status(status):
    """Save status to file"""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving status: {e}")


def main():
    """Main monitor loop"""
    logger.info("=" * 60)
    logger.info("📊 Analysis Processor Monitor Started")
    logger.info(f"    Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Check if processor is running
    is_running = is_processor_running()
    status = get_queue_status()
    
    if not is_running:
        logger.warning("⚠️  Analysis Processor is NOT running!")
        logger.info("🔄 Attempting to start it...")
        
        if start_processor():
            logger.info("✅ Processor started")
            status["processor_running"] = True
            # Wait for startup
            time.sleep(3)
        else:
            logger.error("❌ Failed to start processor")
            status["processor_running"] = False
    else:
        logger.info("✅ Analysis Processor is running")
    
    # Log queue status
    logger.info("")
    logger.info("📋 Queue Status:")
    logger.info(f"   ⏳ Pending requests: {status['pending_requests']}")
    logger.info(f"   ✅ Completed requests: {status['completed_requests']}")
    logger.info(f"   📊 Latest results: {status['latest_results_count']} tickers")
    
    if status['latest_results_time']:
        logger.info(f"   📅 Latest update: {status['latest_results_time'][:19]}")
    
    # Save status for UI to display
    save_status(status)
    
    # Final status
    logger.info("")
    if status["processor_running"]:
        logger.info("✅ Monitor complete - Processor is healthy")
    else:
        logger.warning("⚠️  Monitor complete - Processor may have issues")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Monitor shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


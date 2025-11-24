"""
Base Windows Service Wrapper
Provides core functionality for wrapping monitoring services as Windows services
"""

import sys
import os
import time
import traceback
from pathlib import Path
from loguru import logger

try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager
except ImportError:
    # pywin32 not installed
    win32serviceutil = None
    win32service = None
    win32event = None
    servicemanager = None


class WindowsServiceBase(win32serviceutil.ServiceFramework if win32serviceutil else object):
    """
    Base class for Windows services
    
    Subclasses should override:
    - _svc_name_: Service name
    - _svc_display_name_: Display name in Services panel
    - _svc_description_: Service description
    - run_service(): Main service logic
    """
    
    # Override these in subclass
    _svc_name_ = "BaseService"
    _svc_display_name_ = "Base Service"
    _svc_description_ = "Base Windows Service"
    
    def __init__(self, args):
        """Initialize the service"""
        if win32serviceutil is None:
            raise ImportError("pywin32 package required for Windows services. Install with: pip install pywin32")
        
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.running = False
        self.logging_initialized = False
    
    def _setup_logging(self):
        """Configure logging to file and Windows Event Log"""
        try:
            # Remove default logger
            logger.remove()
            
            # Log to file
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{self._svc_name_}.log"
            
            logger.add(
                str(log_file),
                rotation="50 MB",
                retention="30 days",
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                backtrace=True,
                diagnose=True
            )
            
            logger.info(f"Logging initialized for {self._svc_name_}")
        except Exception as e:
            # Log setup failed - use basic logging as fallback
            import logging
            logging.basicConfig(
                filename=f"c:/temp/{self._svc_name_}_fallback.log",
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logging.error(f"Loguru setup failed: {e}")
    
    def SvcStop(self):
        """Called when the service is asked to stop"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        self.running = False
        logger.info(f"{self._svc_name_} - Stop requested")
        
        # Log to Windows Event Log
        if servicemanager:
            servicemanager.LogInfoMsg(f'{self._svc_display_name_} - Stopping')
    
    def SvcDoRun(self):
        """Called when the service is asked to start"""
        # Report that we're starting up
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
        
        self.running = True
        
        # CRITICAL: Report service as RUNNING immediately
        # This prevents the "service did not respond in a timely fashion" error
        # Must be done BEFORE any heavy initialization including logging
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        
        # Log to Windows Event Log first
        if servicemanager:
            servicemanager.LogInfoMsg(f'{self._svc_display_name_} - Started')
        
        # NOW set up file logging (after we've told Windows we're running)
        if not self.logging_initialized:
            self._setup_logging()
            self.logging_initialized = True
        
        logger.info(f"{self._svc_name_} - Service starting")
        
        try:
            self.run_service()
        except Exception as e:
            logger.error(f"{self._svc_name_} - Fatal error: {e}")
            logger.error(traceback.format_exc())
            
            # Log to Windows Event Log
            if servicemanager:
                servicemanager.LogErrorMsg(f'{self._svc_display_name_} - Fatal error: {str(e)}')
        finally:
            self.running = False
            logger.info(f"{self._svc_name_} - Service stopped")
            
            # Log to Windows Event Log
            if servicemanager:
                servicemanager.LogInfoMsg(f'{self._svc_display_name_} - Stopped')
    
    def run_service(self):
        """
        Main service logic - override this in subclass
        
        Should run continuously and check self.running flag
        Example:
            while self.running:
                # Do work
                time.sleep(60)
        """
        raise NotImplementedError("Subclass must implement run_service()")
    
    def is_stop_requested(self):
        """Check if stop has been requested"""
        return not self.running or win32event.WaitForSingleObject(
            self.stop_event, 0
        ) == win32event.WAIT_OBJECT_0


def install_service(service_class):
    """Install a Windows service"""
    if len(sys.argv) == 1:
        # Called with no arguments - show usage
        print(f"\nUsage: python {sys.argv[0]} [install|update|remove|start|stop|restart]")
        print(f"\nService: {service_class._svc_display_name_}")
        print(f"Description: {service_class._svc_description_}")
        return
    
    try:
        win32serviceutil.HandleCommandLine(service_class)
    except Exception as e:
        print(f"Error managing service: {e}")
        print(traceback.format_exc())


def check_pywin32_installed():
    """Check if pywin32 is installed"""
    if win32serviceutil is None:
        print("\n❌ ERROR: pywin32 package not installed")
        print("\nTo install:")
        print("  pip install pywin32")
        print("\nAfter installing, run:")
        print("  python -m win32serviceutil")
        print("\nThen try again.")
        return False
    return True

"""
Minimal test service to verify Windows service framework is working
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.windows_service_base import WindowsServiceBase, install_service, check_pywin32_installed


class TestService(WindowsServiceBase):
    """Minimal test service"""
    
    _svc_name_ = "SentientTest"
    _svc_display_name_ = "Sentient Test Service"
    _svc_description_ = "Minimal test service to verify framework works"
    
    def run_service(self):
        """Just sleep and log"""
        import logging
        logging.basicConfig(filename='c:/temp/test_service.log', level=logging.INFO)
        logging.info("Test service started!")
        
        while self.running and not self.is_stop_requested():
            logging.info("Test service running...")
            time.sleep(10)
        
        logging.info("Test service stopped")


if __name__ == '__main__':
    if not check_pywin32_installed():
        sys.exit(1)
    
    install_service(TestService)

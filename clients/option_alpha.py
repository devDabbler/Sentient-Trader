"""Option Alpha webhook client."""

import os
import json
from loguru import logger
import requests
from datetime import datetime
from typing import Dict, Optional, Tuple



class OptionAlphaClient:
    """Client for sending signals to Option Alpha webhook"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('OPTION_ALPHA_WEBHOOK_URL')
        self.timeout = 10
    
    def send_signal(self, signal: Dict) -> Tuple[bool, str]:
        """Send signal to Option Alpha webhook"""
        try:
            if not self.webhook_url:
                return False, "Option Alpha webhook URL not configured"
            
            signal['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Sending signal to Option Alpha: {json.dumps(signal, indent=2)}")
            
            response = requests.post(
                self.webhook_url,
                json=signal,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            
            logger.info(f"Signal sent successfully. Response: {response.status_code}")
            return True, f"Signal sent successfully (Status: {response.status_code})"
            
        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error(error_msg)
            return False, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

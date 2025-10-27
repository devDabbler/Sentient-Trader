"""
Trading Mode Configuration System
Handles switching between paper trading and production trading modes
"""

import os
import logging
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    PRODUCTION = "production"

@dataclass
class TradingCredentials:
    """Trading credentials for a specific mode"""
    account_id: str
    access_token: str
    api_url: str
    mode: TradingMode

class TradingModeManager:
    """Manages trading mode configuration and credentials"""
    
    def __init__(self):
        self.current_mode = TradingMode.PAPER
        self.credentials = {}
        self._load_credentials()
    
    def _load_credentials(self):
        """Load credentials for both paper and production modes"""
        # Paper trading credentials
        paper_account_id = os.getenv('TRADIER_PAPER_ACCOUNT_ID') or os.getenv('TRADIER_ACCOUNT_ID')
        paper_access_token = os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or os.getenv('TRADIER_ACCESS_TOKEN')
        paper_api_url = os.getenv('TRADIER_PAPER_API_URL', 'https://sandbox.tradier.com')
        
        if paper_account_id and paper_access_token:
            self.credentials[TradingMode.PAPER] = TradingCredentials(
                account_id=paper_account_id,
                access_token=paper_access_token,
                api_url=paper_api_url,
                mode=TradingMode.PAPER
            )
            logger.info("✅ Paper trading credentials loaded")
        else:
            logger.warning("⚠️ Paper trading credentials not found")
        
        # Production trading credentials
        prod_account_id = os.getenv('TRADIER_PROD_ACCOUNT_ID')
        prod_access_token = os.getenv('TRADIER_PROD_ACCESS_TOKEN')
        prod_api_url = os.getenv('TRADIER_PROD_API_URL', 'https://api.tradier.com')
        
        if prod_account_id and prod_access_token:
            self.credentials[TradingMode.PRODUCTION] = TradingCredentials(
                account_id=prod_account_id,
                access_token=prod_access_token,
                api_url=prod_api_url,
                mode=TradingMode.PRODUCTION
            )
            logger.info("✅ Production trading credentials loaded")
        else:
            logger.warning("⚠️ Production trading credentials not found")
    
    def set_mode(self, mode: TradingMode) -> bool:
        """Set the current trading mode"""
        if mode not in self.credentials:
            logger.error(f"❌ No credentials available for {mode.value} mode")
            return False
        
        self.current_mode = mode
        logger.info(f"🔄 Switched to {mode.value} trading mode")
        return True
    
    def get_current_credentials(self) -> Optional[TradingCredentials]:
        """Get credentials for the current trading mode"""
        return self.credentials.get(self.current_mode)
    
    def get_mode(self) -> TradingMode:
        """Get the current trading mode"""
        return self.current_mode
    
    def is_paper_mode(self) -> bool:
        """Check if currently in paper trading mode"""
        return self.current_mode == TradingMode.PAPER
    
    def is_production_mode(self) -> bool:
        """Check if currently in production trading mode"""
        return self.current_mode == TradingMode.PRODUCTION
    
    def get_available_modes(self) -> list[TradingMode]:
        """Get list of available trading modes based on loaded credentials"""
        return list(self.credentials.keys())
    
    def validate_credentials(self, mode: TradingMode) -> Tuple[bool, str]:
        """Validate credentials for a specific mode"""
        if mode not in self.credentials:
            return False, f"No credentials available for {mode.value} mode"
        
        creds = self.credentials[mode]
        if not creds.account_id or not creds.access_token:
            return False, f"Missing credentials for {mode.value} mode"
        
        return True, f"Credentials valid for {mode.value} mode"
    
    def get_mode_display_info(self) -> Dict[str, str]:
        """Get display information for the current mode"""
        mode = self.current_mode
        creds = self.get_current_credentials()
        
        if not creds:
            return {
                "mode": "Unknown",
                "status": "❌ No credentials",
                "api_url": "N/A",
                "account_id": "N/A"
            }
        
        status = "✅ Ready" if creds else "❌ Not configured"
        return {
            "mode": mode.value.title(),
            "status": status,
            "api_url": creds.api_url,
            "account_id": creds.account_id[:8] + "..." if creds.account_id else "N/A"
        }

# Global instance
trading_mode_manager = TradingModeManager()

def get_trading_mode_manager() -> TradingModeManager:
    """Get the global trading mode manager instance"""
    return trading_mode_manager

def switch_to_paper_mode() -> bool:
    """Switch to paper trading mode"""
    return trading_mode_manager.set_mode(TradingMode.PAPER)

def switch_to_production_mode() -> bool:
    """Switch to production trading mode"""
    return trading_mode_manager.set_mode(TradingMode.PRODUCTION)

def is_paper_mode() -> bool:
    """Check if currently in paper trading mode"""
    return trading_mode_manager.is_paper_mode()

def is_production_mode() -> bool:
    """Check if currently in production trading mode"""
    return trading_mode_manager.is_production_mode()

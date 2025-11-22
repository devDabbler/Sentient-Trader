"""
Trading Mode Configuration System
Handles switching between paper trading and production trading modes
Supports both Tradier and Interactive Brokers (IBKR)
"""

import os
from loguru import logger
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import streamlit as st


class TradingMode(Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    PRODUCTION = "production"

class BrokerType(Enum):
    """Broker type enumeration"""
    TRADIER = "tradier"
    IBKR = "ibkr"

@dataclass
class TradingCredentials:
    """Trading credentials for a specific mode"""
    account_id: str
    access_token: str
    api_url: str
    mode: TradingMode

@dataclass
class IBKRCredentials:
    """IBKR-specific credentials"""
    host: str
    port: int
    client_id: int
    mode: TradingMode
    account_id: Optional[str] = None  # Will be populated after connection

class TradingModeManager:
    """Manages trading mode configuration and credentials for Tradier and IBKR"""
    
    def __init__(self):
        # Default to paper mode if not in session state
        if 'trading_mode' not in st.session_state:
            st.session_state.trading_mode = TradingMode.PAPER
        self.current_mode = st.session_state.trading_mode
        
        # Separate credentials for each broker
        self.tradier_credentials = {}
        self.ibkr_credentials = {}
        
        # For backwards compatibility, keep main credentials dict for Tradier
        self.credentials = self.tradier_credentials
        
        self._load_credentials()
        self._load_ibkr_credentials()
    
    def _load_credentials(self):
        """Load Tradier credentials for both paper and production modes"""
        # Paper trading credentials
        paper_account_id = (os.getenv('TRADIER_PAPER_ACCOUNT_ID') or os.getenv('TRADIER_ACCOUNT_ID') or '').strip()
        paper_access_token = (os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or os.getenv('TRADIER_ACCESS_TOKEN') or '').strip()
        paper_api_url = os.getenv('TRADIER_PAPER_API_URL', 'https://sandbox.tradier.com').strip()
        
        if paper_account_id and paper_access_token:
            self.tradier_credentials[TradingMode.PAPER] = TradingCredentials(
                account_id=paper_account_id,
                access_token=paper_access_token,
                api_url=paper_api_url,
                mode=TradingMode.PAPER
            )
            logger.info("✅ Tradier Paper trading credentials loaded")
        else:
            logger.warning("⚠️ Tradier Paper trading credentials not found")
        
        # Production trading credentials
        prod_account_id = os.getenv('TRADIER_PROD_ACCOUNT_ID', '').strip()
        prod_access_token = os.getenv('TRADIER_PROD_ACCESS_TOKEN', '').strip()
        prod_api_url = os.getenv('TRADIER_PROD_API_URL', 'https://api.tradier.com').strip()
        
        logger.info("🔍 Debug - prod_account_id: '{}' (len={})", str(prod_account_id), len(prod_account_id))
        logger.info("🔍 Debug - prod_access_token: '{}...' (len={})", str(prod_access_token[:10]), len(prod_access_token))
        
        if prod_account_id and prod_access_token:
            self.tradier_credentials[TradingMode.PRODUCTION] = TradingCredentials(
                account_id=prod_account_id,
                access_token=prod_access_token,
                api_url=prod_api_url,
                mode=TradingMode.PRODUCTION
            )
            logger.info("✅ Production trading credentials loaded")
            logger.info(f"🔍 Debug - credentials dict keys: {list(self.tradier_credentials.keys())}")
        else:
            logger.warning(f"⚠️ Production trading credentials not found - account_id: {bool(prod_account_id)}, token: {bool(prod_access_token)}")
    
    def _load_ibkr_credentials(self):
        """Load IBKR credentials for both paper and live trading modes"""
        # IBKR Paper trading credentials
        paper_host = os.getenv('IBKR_PAPER_HOST', '127.0.0.1').strip()
        paper_port_str = os.getenv('IBKR_PAPER_PORT', '').strip()
        paper_client_id_str = os.getenv('IBKR_PAPER_CLIENT_ID', '1').strip()
        
        # Use legacy IBKR_PORT if paper port not specified (defaults to 7497 for paper TWS or 4002 for paper Gateway)
        if not paper_port_str:
            paper_port_str = os.getenv('IBKR_PORT', '7497').strip()
        
        if paper_port_str:
            try:
                paper_port = int(paper_port_str)
                paper_client_id = int(paper_client_id_str)
                
                self.ibkr_credentials[TradingMode.PAPER] = IBKRCredentials(
                    host=paper_host,
                    port=paper_port,
                    client_id=paper_client_id,
                    mode=TradingMode.PAPER
                )
                logger.info(f"✅ IBKR Paper trading credentials loaded: {paper_host}:{paper_port} (client_id={paper_client_id})")
            except ValueError as e:
                logger.error(f"❌ Invalid IBKR paper port or client_id: {e}")
        else:
            logger.warning("⚠️ IBKR Paper trading credentials not found")
        
        # IBKR Live/Production trading credentials
        live_host = os.getenv('IBKR_LIVE_HOST', '127.0.0.1').strip()
        live_port_str = os.getenv('IBKR_LIVE_PORT', '').strip()
        live_client_id_str = os.getenv('IBKR_LIVE_CLIENT_ID', '1').strip()
        
        logger.info("🔍 Debug - IBKR live_port: '{}' (len={})", str(live_port_str), len(live_port_str))
        logger.info("🔍 Debug - IBKR live_host: '{}' (len={})", str(live_host), len(live_host))
        
        if live_port_str:
            try:
                live_port = int(live_port_str)
                live_client_id = int(live_client_id_str)
                
                self.ibkr_credentials[TradingMode.PRODUCTION] = IBKRCredentials(
                    host=live_host,
                    port=live_port,
                    client_id=live_client_id,
                    mode=TradingMode.PRODUCTION
                )
                logger.info(f"✅ IBKR Live trading credentials loaded: {live_host}:{live_port} (client_id={live_client_id})")
                logger.info(f"🔍 Debug - IBKR credentials dict keys: {list(self.ibkr_credentials.keys())}")
            except ValueError as e:
                logger.error(f"❌ Invalid IBKR live port or client_id: {e}")
        else:
            logger.warning(f"⚠️ IBKR Live trading credentials not found")
    
    def set_mode(self, mode: TradingMode) -> bool:
        """Set the current trading mode"""
        logger.info(f"🔍 Debug set_mode - Requested mode: {mode} (type: {type(mode)}), value: {mode.value}")
        logger.info(f"🔍 Debug set_mode - Available credentials: {list(self.credentials.keys())}")
        logger.info(f"🔍 Debug set_mode - Mode in credentials (direct): {mode in self.credentials}")
        
        # Check if mode exists by comparing enum values (handles enum identity issues)
        mode_found = False
        actual_key = None
        for key in self.credentials.keys():
            logger.info(f"🔍 Debug - Comparing {mode} == {key}: {mode == key}, id match: {id(mode) == id(key)}")
            if key.value == mode.value:
                mode_found = True
                actual_key = key
                break
        
        if not mode_found:
            logger.error(f"❌ No credentials available for {mode.value} mode")
            logger.error(f"❌ Credentials dict: {self.credentials}")
            return False
        
        # Use actual_key for both to maintain consistency
        self.current_mode = actual_key
        st.session_state.trading_mode = actual_key  # Persist to session state
        logger.info(f"🔄 Switched to {mode.value} trading mode")
        return True
    
    def get_current_credentials(self) -> Optional[TradingCredentials]:
        """Get credentials for the current trading mode"""
        # Always sync with session state to avoid stale data after rerun
        if 'trading_mode' in st.session_state:
            self.current_mode = st.session_state.trading_mode
        # Use value-based lookup to handle enum identity issues
        for key, creds in self.credentials.items():
            if key.value == self.current_mode.value:
                return creds
        return None
    
    def get_mode(self) -> TradingMode:
        """Get the current trading mode"""
        # Always sync with session state to avoid stale data after rerun
        if 'trading_mode' in st.session_state:
            self.current_mode = st.session_state.trading_mode
        return self.current_mode
    
    def is_paper_mode(self) -> bool:
        """Check if currently in paper trading mode"""
        # Always sync with session state to avoid stale data after rerun
        if 'trading_mode' in st.session_state:
            self.current_mode = st.session_state.trading_mode
        return self.current_mode == TradingMode.PAPER
    
    def is_production_mode(self) -> bool:
        """Check if currently in production trading mode"""
        # Always sync with session state to avoid stale data after rerun
        if 'trading_mode' in st.session_state:
            self.current_mode = st.session_state.trading_mode
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
        # Always sync with session state to avoid stale data after rerun
        if 'trading_mode' in st.session_state:
            self.current_mode = st.session_state.trading_mode
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
    
    def get_ibkr_credentials(self, mode: Optional[TradingMode] = None) -> Optional[IBKRCredentials]:
        """
        Get IBKR credentials for specified mode or current mode
        
        Args:
            mode: Trading mode (PAPER or PRODUCTION). If None, uses current mode.
        
        Returns:
            IBKRCredentials or None if not available
        """
        target_mode = mode if mode else self.current_mode
        
        # Use value-based lookup to handle enum identity issues
        for key, creds in self.ibkr_credentials.items():
            if key.value == target_mode.value:
                return creds
        return None
    
    def get_tradier_credentials(self, mode: Optional[TradingMode] = None) -> Optional[TradingCredentials]:
        """
        Get Tradier credentials for specified mode or current mode
        
        Args:
            mode: Trading mode (PAPER or PRODUCTION). If None, uses current mode.
        
        Returns:
            TradingCredentials or None if not available
        """
        target_mode = mode if mode else self.current_mode
        
        # Use value-based lookup to handle enum identity issues
        for key, creds in self.tradier_credentials.items():
            if key.value == target_mode.value:
                return creds
        return None
    
    def has_ibkr_credentials(self, mode: Optional[TradingMode] = None) -> bool:
        """Check if IBKR credentials are available for the specified or current mode"""
        return self.get_ibkr_credentials(mode) is not None
    
    def has_tradier_credentials(self, mode: Optional[TradingMode] = None) -> bool:
        """Check if Tradier credentials are available for the specified or current mode"""
        return self.get_tradier_credentials(mode) is not None
    
    def get_available_brokers(self, mode: Optional[TradingMode] = None) -> list[BrokerType]:
        """
        Get list of available brokers for the specified or current mode
        
        Args:
            mode: Trading mode to check. If None, uses current mode.
        
        Returns:
            List of available BrokerType enums
        """
        target_mode = mode if mode else self.current_mode
        brokers = []
        
        if self.has_tradier_credentials(target_mode):
            brokers.append(BrokerType.TRADIER)
        
        if self.has_ibkr_credentials(target_mode):
            brokers.append(BrokerType.IBKR)
        
        return brokers

# Global instance (lazily initialized)
_trading_mode_manager: Optional[TradingModeManager] = None

def get_trading_mode_manager() -> TradingModeManager:
    """Get the global trading mode manager instance (lazy initialization)"""
    global _trading_mode_manager
    
    # CRITICAL SAFETY: Ensure session state trading_mode exists and is valid before creating manager
    if 'trading_mode' not in st.session_state:
        st.session_state.trading_mode = TradingMode.PAPER
        logger.info("🔒 get_trading_mode_manager: Initialized trading_mode to PAPER (safe default)")
    else:
        # Validate trading_mode by comparing values (enum identity can change after Streamlit serialization)
        current_mode_value = getattr(st.session_state.trading_mode, 'value', None)
        if current_mode_value not in ['paper', 'production']:
            # Safety check: if trading_mode is corrupted/invalid, reset to PAPER
            logger.warning(f"⚠️ get_trading_mode_manager: Invalid trading_mode in session state: {st.session_state.trading_mode}. Resetting to PAPER for safety.")
            st.session_state.trading_mode = TradingMode.PAPER
        elif current_mode_value == 'paper':
            # Ensure it's the correct enum instance (fixes serialization issues)
            st.session_state.trading_mode = TradingMode.PAPER
        elif current_mode_value == 'production':
            # Ensure it's the correct enum instance
            st.session_state.trading_mode = TradingMode.PRODUCTION
    
    if _trading_mode_manager is None:
        _trading_mode_manager = TradingModeManager()
    else:
        # On Streamlit reruns, ensure the cached manager syncs with session state
        # This handles cases where session state changed but manager instance persisted
        if 'trading_mode' in st.session_state:
            current_session_mode = st.session_state.trading_mode
            if _trading_mode_manager.current_mode != current_session_mode:
                logger.info(f"🔄 get_trading_mode_manager: Syncing cached manager with session state ({current_session_mode})")
                _trading_mode_manager.current_mode = current_session_mode
    
    return _trading_mode_manager

def switch_to_paper_mode() -> bool:
    """Switch to paper trading mode"""
    return get_trading_mode_manager().set_mode(TradingMode.PAPER)

def switch_to_production_mode() -> bool:
    """Switch to production trading mode"""
    return get_trading_mode_manager().set_mode(TradingMode.PRODUCTION)

def is_paper_mode() -> bool:
    """Check if currently in paper trading mode"""
    return get_trading_mode_manager().is_paper_mode()

def is_production_mode() -> bool:
    """Check if currently in production trading mode"""
    return get_trading_mode_manager().is_production_mode()

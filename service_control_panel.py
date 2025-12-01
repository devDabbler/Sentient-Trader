"""
Sentient Trader - Service Control Panel

A simple web GUI to manage your VPS services from anywhere.
Password + TOTP (2FA) protected for security.

Usage (on VPS):
    streamlit run service_control_panel.py --server.port 8501 --server.address 0.0.0.0
    
Access from anywhere:
    http://YOUR_VPS_IP:8501

First time setup:
    1. Set CONTROL_PANEL_PASSWORD in .env (strong password)
    2. Set CONTROL_PANEL_TOTP_SECRET in .env (run: python -c "import pyotp; print(pyotp.random_base32())")
    3. Scan QR code with Google Authenticator / Authy
"""

import streamlit as st
import subprocess
import os
import json
import time
from pathlib import Path
from datetime import datetime
import importlib
from typing import Any, Optional
import platform
import binascii

# Try to import TOTP library dynamically to avoid static linter errors when dev env
# doesn't have these optional dependencies installed.  Uses importlib so Pylance
# doesn't raise unresolved import warnings, and sets fallbacks to None.
pyotp: Optional[Any] = None
qrcode: Optional[Any] = None
import io
import base64
TOTP_AVAILABLE = False
try:
    pyotp = importlib.import_module("pyotp")  # type: ignore
    qrcode = importlib.import_module("qrcode")  # type: ignore
    TOTP_AVAILABLE = True
except Exception:
    # If importlib failed to load optional deps, keep fallbacks and mark disabled
    pyotp = None
    qrcode = None
    TOTP_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================
# Password - set in environment variable or use default
ADMIN_PASSWORD = os.getenv("CONTROL_PANEL_PASSWORD", "admin")  # Set your own

# TOTP Secret for 2FA (generate with: python -c "import pyotp; print(pyotp.random_base32())")
def sanitize_totp_secret(secret: Optional[str]) -> Optional[str]:
    """Sanitize and validate TOTP secret from environment variable."""
    if not secret:
        return None
    
    # Strip whitespace and quotes
    secret = secret.strip().strip('"').strip("'")
    
    # Validate it's valid base32 (only A-Z, 2-7 allowed)
    if not secret:
        return None
    
    # Check if it contains only valid base32 characters
    valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ234567')
    if not all(c.upper() in valid_chars for c in secret):
        print(f"[WARNING] TOTP_SECRET contains invalid characters. Only A-Z and 2-7 are allowed in base32.")
        return None
    
    # Convert to uppercase (base32 is case-insensitive but pyotp expects uppercase)
    secret = secret.upper()
    
    # Validate length (should be at least 16 characters for TOTP)
    if len(secret) < 16:
        print(f"[WARNING] TOTP_SECRET is too short (minimum 16 characters).")
        return None
    
    return secret

_raw_totp_secret = os.getenv("CONTROL_PANEL_TOTP_SECRET")
TOTP_SECRET = sanitize_totp_secret(_raw_totp_secret)
TOTP_ENABLED = TOTP_AVAILABLE and TOTP_SECRET and len(TOTP_SECRET) >= 16

# Service intervals config file path
SERVICE_INTERVALS_FILE = Path(__file__).resolve().parent / "data" / "service_intervals.json"
SERVICE_WATCHLISTS_FILE = Path(__file__).resolve().parent / "data" / "service_watchlists.json"
SERVICE_DISCORD_FILE = Path(__file__).resolve().parent / "data" / "service_discord_settings.json"
ACTIVE_STRATEGY_FILE = Path(__file__).resolve().parent / "active_strategy.json"
ANALYSIS_REQUESTS_FILE = Path(__file__).resolve().parent / "data" / "analysis_requests.json"
ANALYSIS_RESULTS_FILE = Path(__file__).resolve().parent / "data" / "analysis_results.json"
AI_POSITIONS_FILE = Path(__file__).resolve().parent / "data" / "ai_crypto_positions.json"

# Default watchlists for each service type
DEFAULT_WATCHLISTS = {
    "crypto": ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD', 'DOGE/USD', 'SHIB/USD', 'PEPE/USD'],
    "stocks": ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'PLTR', 'SOFI', 'COIN'],
    "dex": ['solana', 'ethereum', 'base'],  # chains to monitor
}

# Analysis presets for quick mobile access
ANALYSIS_PRESETS = {
    # Crypto Analysis Modes (matches Quick Trade UI)
    "crypto_standard": {
        "name": "🔬 Crypto Standard",
        "emoji": "🔬",
        "tickers": None,  # Use watchlist
        "depth": "standard",
        "asset_type": "crypto",
        "analysis_mode": "standard",
        "description": "Single strategy + timeframe analysis"
    },
    "crypto_multi": {
        "name": "🎯 Crypto Multi-Config",
        "emoji": "🎯",
        "tickers": None,  # Use watchlist
        "depth": "multi",
        "asset_type": "crypto",
        "analysis_mode": "multi_config",
        "description": "Test Long/Short + all leverage levels"
    },
    "crypto_ultimate": {
        "name": "🚀 Crypto Ultimate",
        "emoji": "🚀",
        "tickers": None,  # Use watchlist
        "depth": "ultimate",
        "asset_type": "crypto",
        "analysis_mode": "ultimate",
        "description": "ALL strategies + directions + leverages"
    },
    # Quick scans
    "quick_crypto": {
        "name": "⚡ Quick Crypto (Top 3)",
        "emoji": "⚡",
        "tickers": ['BTC/USD', 'ETH/USD', 'SOL/USD'],
        "depth": "quick",
        "asset_type": "crypto",
        "analysis_mode": "standard",
        "description": "Fast scan of BTC, ETH, SOL"
    },
    # Stock presets - Analysis Modes (Standard/Multi/Ultimate)
    "stock_standard": {
        "name": "🔬 Stock Standard",
        "emoji": "🔬",
        "tickers": None,  # Use watchlist
        "depth": "standard",
        "asset_type": "stock",
        "analysis_mode": "standard",
        "description": "Single strategy analysis for stocks"
    },
    "stock_multi": {
        "name": "🎯 Stock Multi-Config",
        "emoji": "🎯",
        "tickers": None,  # Use watchlist
        "depth": "multi",
        "asset_type": "stock",
        "analysis_mode": "multi_config",
        "description": "Test Long/Short + multiple timeframes"
    },
    "stock_ultimate": {
        "name": "🚀 Stock Ultimate",
        "emoji": "🚀",
        "tickers": None,  # Use watchlist
        "depth": "ultimate",
        "asset_type": "stock",
        "analysis_mode": "ultimate",
        "description": "ALL strategies + directions + timeframes"
    },
    # Stock Quick Scans
    "stock_momentum": {
        "name": "📈 Stock Momentum",
        "emoji": "📈",
        "tickers": ['NVDA', 'TSLA', 'AMD', 'PLTR', 'COIN', 'MARA'],
        "depth": "medium",
        "asset_type": "stock",
        "analysis_mode": "standard",
        "description": "Momentum scan for popular stocks"
    },
    "orb_fvg_scan": {
        "name": "🎯 ORB+FVG Day Trade",
        "emoji": "🎯",
        "tickers": ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD'],
        "depth": "deep",
        "asset_type": "stock",
        "analysis_mode": "standard",
        "description": "Opening Range Breakout + Fair Value Gap"
    },
}

SERVICES = {
    "DEX Launch Monitor": {
        "name": "sentient-dex-launch",
        "description": "Scans for new token launches on DEX platforms (Solana, ETH, etc.)",
        "emoji": "🚀",
        "category": "crypto",
        "interval_key": "scan_interval_seconds",
        "interval_default": 30,
        "interval_min": 5,
        "interval_max": 3600  # Up to 1 hour
    },
    "Crypto Breakout Monitor": {
        "name": "sentient-crypto-breakout",
        "description": "Monitors crypto for breakout patterns and momentum",
        "emoji": "📈",
        "category": "crypto",
        "interval_key": "scan_interval_seconds",
        "interval_default": 180,
        "interval_min": 10,
        "interval_max": 3600  # Up to 1 hour
    },
    "AI Crypto Trader": {
        "name": "sentient-crypto-ai-trader",
        "description": "AI-powered crypto position manager (executes trades)",
        "emoji": "🤖",
        "category": "crypto",
        "interval_key": "check_interval_seconds",
        "interval_default": 60,
        "interval_min": 10,
        "interval_max": 3600  # Up to 1 hour
    },
    "Stock Monitor": {
        "name": "sentient-stock-monitor",
        "description": "Monitors stocks for trading opportunities",
        "emoji": "📊",
        "category": "stocks",
        "interval_key": "scan_interval_seconds",
        "interval_default": 300,
        "interval_min": 30,
        "interval_max": 3600  # Up to 1 hour
    },
    "AI Stock Trader": {
        "name": "sentient-stock-ai-trader",
        "description": "AI-powered stock position manager (executes trades via Tradier/IBKR)",
        "emoji": "🤖",
        "category": "stocks",
        "interval_key": "check_interval_seconds",
        "interval_default": 60,
        "interval_min": 30,
        "interval_max": 300  # Up to 5 minutes
    },
    "ORB FVG Scanner": {
        "name": "sentient-orb-fvg",
        "description": "15-min Opening Range Breakout + Fair Value Gap scanner (9:30 AM - 12:30 PM ET)",
        "emoji": "🎯",
        "category": "stocks",
        "interval_key": "scan_interval_seconds",
        "interval_default": 60,
        "interval_min": 30,
        "interval_max": 300  # Up to 5 minutes (intraday strategy)
    },
    "Discord Approval Bot": {
        "name": "sentient-discord-approval",
        "description": "Discord bot for trade approvals (recommended: keep auto-start)",
        "emoji": "💬",
        "category": "infrastructure"
    },
    "Analysis Queue Processor": {
        "name": "sentient-analysis-queue",
        "description": "Processes queued analysis requests from Control Panel/Discord",
        "emoji": "🧠",
        "category": "infrastructure",
        "interval_key": "check_interval_seconds",
        "interval_default": 10,
        "interval_min": 5,
        "interval_max": 60
    }
}

# ============================================================
# SERVICE INTERVAL FUNCTIONS
# ============================================================

def load_service_intervals() -> dict:
    """Load service intervals from JSON file"""
    try:
        if SERVICE_INTERVALS_FILE.exists():
            with open(SERVICE_INTERVALS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading service intervals: {e}")
    return {}


def save_service_intervals(intervals: dict) -> bool:
    """Save service intervals to JSON file"""
    try:
        SERVICE_INTERVALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SERVICE_INTERVALS_FILE, 'w') as f:
            json.dump(intervals, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving service intervals: {e}")
        return False


def get_service_interval(service_name: str, svc_info: dict) -> int:
    """Get the current interval for a service"""
    intervals = load_service_intervals()
    interval_key = svc_info.get("interval_key")
    min_interval = svc_info.get("interval_min", 10)
    max_interval = svc_info.get("interval_max", 3600)
    default_interval = svc_info.get("interval_default", min_interval)
    
    if not interval_key:
        return default_interval
    
    service_config = intervals.get(service_name, {})
    interval = service_config.get(interval_key, default_interval)
    
    # Ensure interval is within valid range (min to max)
    # Handle cases where interval might be 0, None, or invalid
    if not isinstance(interval, (int, float)) or interval <= 0:
        interval = default_interval
    
    return max(min_interval, min(int(interval), max_interval))


def set_service_interval(service_name: str, svc_info: dict, new_interval: int) -> bool:
    """Set the interval for a service and trigger a restart if running"""
    intervals = load_service_intervals()
    interval_key = svc_info.get("interval_key")
    if not interval_key:
        return False
    
    if service_name not in intervals:
        intervals[service_name] = {}
    
    intervals[service_name][interval_key] = new_interval
    
    if save_service_intervals(intervals):
        # Also update environment variable for running process
        env_var_name = f"{service_name.upper().replace('-', '_')}_{interval_key.upper()}"
        os.environ[env_var_name] = str(new_interval)
        return True
    return False


# ============================================================
# WATCHLIST MANAGEMENT
# ============================================================

def load_service_watchlists() -> dict:
    """Load service-specific watchlists from JSON file"""
    try:
        if SERVICE_WATCHLISTS_FILE.exists():
            with open(SERVICE_WATCHLISTS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading service watchlists: {e}")
    return {}


def save_service_watchlists(watchlists: dict) -> bool:
    """Save service watchlists to JSON file"""
    try:
        SERVICE_WATCHLISTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SERVICE_WATCHLISTS_FILE, 'w') as f:
            json.dump(watchlists, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving service watchlists: {e}")
        return False


def get_service_watchlist(service_name: str, category: str = "stocks") -> list:
    """Get watchlist for a specific service"""
    watchlists = load_service_watchlists()
    if service_name in watchlists:
        return watchlists[service_name].get("tickers", [])
    # Return default based on category
    return DEFAULT_WATCHLISTS.get(category, [])


def set_service_watchlist(service_name: str, tickers: list) -> bool:
    """Set watchlist for a specific service"""
    watchlists = load_service_watchlists()
    if service_name not in watchlists:
        watchlists[service_name] = {}
    watchlists[service_name]["tickers"] = tickers
    watchlists[service_name]["updated"] = datetime.now().isoformat()
    return save_service_watchlists(watchlists)


# ============================================================
# STRATEGY CONFIG MANAGEMENT
# ============================================================

def load_active_strategy() -> dict:
    """Load active strategy configuration"""
    try:
        if ACTIVE_STRATEGY_FILE.exists():
            with open(ACTIVE_STRATEGY_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading active strategy: {e}")
    return {"active_strategy": "PAPER_TRADING", "available_strategies": {}}


def save_active_strategy(strategy_data: dict) -> bool:
    """Save active strategy configuration"""
    try:
        strategy_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ACTIVE_STRATEGY_FILE, 'w') as f:
            json.dump(strategy_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving active strategy: {e}")
        return False


def switch_strategy(strategy_key: str) -> tuple:
    """Switch to a different strategy configuration"""
    strategy_data = load_active_strategy()
    available = strategy_data.get("available_strategies", {})
    
    if strategy_key not in available:
        return False, f"Strategy '{strategy_key}' not found"
    
    strategy_data["active_strategy"] = strategy_key
    strategy_data["config_file"] = available[strategy_key].get("config_file", "")
    
    if save_active_strategy(strategy_data):
        return True, f"Switched to {available[strategy_key].get('name', strategy_key)}"
    return False, "Failed to save strategy configuration"


# ============================================================
# ON-DEMAND ANALYSIS
# ============================================================

def queue_analysis_request(preset_key: str, custom_tickers: Optional[list] = None) -> bool:
    """Queue an analysis request for services to pick up"""
    try:
        ANALYSIS_REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing requests
        requests = []
        if ANALYSIS_REQUESTS_FILE.exists():
            with open(ANALYSIS_REQUESTS_FILE, 'r') as f:
                requests = json.load(f)
        
        # Add new request
        preset = ANALYSIS_PRESETS.get(preset_key, {})
        request = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "preset": preset_key,
            "tickers": custom_tickers or preset.get("tickers", []),
            "depth": preset.get("depth", "medium"),
            "asset_type": preset.get("asset_type", "crypto"),
            "analysis_mode": preset.get("analysis_mode", "standard"),
            "status": "pending",
            "created": datetime.now().isoformat(),
        }
        requests.append(request)
        
        # Keep only last 20 requests
        requests = requests[-20:]
        
        with open(ANALYSIS_REQUESTS_FILE, 'w') as f:
            json.dump(requests, f, indent=2)
        return True
    except Exception as e:
        print(f"Error queuing analysis: {e}")
        return False


def get_analysis_requests() -> list:
    """Get pending analysis requests"""
    try:
        if ANALYSIS_REQUESTS_FILE.exists():
            with open(ANALYSIS_REQUESTS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading analysis requests: {e}")
    return []


def clear_analysis_requests() -> bool:
    """Clear all analysis requests"""
    try:
        # Ensure parent directory exists
        ANALYSIS_REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ANALYSIS_REQUESTS_FILE, 'w') as f:
            json.dump([], f)
        print(f"[INFO] Cleared analysis requests file: {ANALYSIS_REQUESTS_FILE}")
        return True
    except Exception as e:
        print(f"Error clearing analysis requests: {e}")
        return False


def clear_analysis_results() -> bool:
    """Clear all analysis results"""
    try:
        # Ensure parent directory exists
        ANALYSIS_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ANALYSIS_RESULTS_FILE, 'w') as f:
            json.dump({}, f)
        print(f"[INFO] Cleared analysis results file: {ANALYSIS_RESULTS_FILE}")
        return True
    except Exception as e:
        print(f"Error clearing analysis results: {e}")
        return False


# ============================================================
# DISCORD SETTINGS MANAGEMENT
# ============================================================

def load_discord_settings() -> dict:
    """Load Discord settings for all services"""
    try:
        if SERVICE_DISCORD_FILE.exists():
            with open(SERVICE_DISCORD_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading discord settings: {e}")
    return {}


def save_discord_settings(settings: dict) -> bool:
    """Save Discord settings"""
    try:
        SERVICE_DISCORD_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SERVICE_DISCORD_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving discord settings: {e}")
        return False


def get_service_discord_settings(service_name: str) -> dict:
    """Get Discord settings for a specific service"""
    settings = load_discord_settings()
    default = {
        'enabled': True,
        'min_confidence': 70,
        'alert_types': ['signal', 'breakout', 'error'],
        'cooldown_minutes': 15,
    }
    return {**default, **settings.get(service_name, {})}


def set_service_discord_settings(service_name: str, new_settings: dict) -> bool:
    """Set Discord settings for a specific service"""
    all_settings = load_discord_settings()
    all_settings[service_name] = new_settings
    return save_discord_settings(all_settings)


# ============================================================
# ORCHESTRATOR INTEGRATION
# ============================================================

def get_orchestrator_dashboard() -> Optional[dict]:
    """Get orchestrator dashboard data (safe import)"""
    try:
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        return orch.get_dashboard_data()
    except Exception as e:
        print(f"Orchestrator not available: {e}")
        return None


def set_workflow_mode(mode: str) -> bool:
    """Set orchestrator workflow mode"""
    try:
        from services.service_orchestrator import get_orchestrator, WorkflowMode
        orch = get_orchestrator()
        return orch.set_mode(WorkflowMode(mode))
    except Exception as e:
        print(f"Failed to set mode: {e}")
        return False


def get_pending_alerts(asset_type: Optional[str] = None) -> list:
    """Get pending alerts from orchestrator"""
    try:
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        alerts = orch.get_pending_alerts(asset_type)
        return [a.to_dict() for a in alerts]
    except Exception as e:
        print(f"Failed to get alerts: {e}")
        return []


def approve_alert(alert_id: str, add_to_watchlist: bool = True) -> bool:
    """Approve an alert"""
    try:
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        return orch.approve_alert(alert_id, add_to_watchlist)
    except Exception as e:
        print(f"Failed to approve alert: {e}")
        return False


def reject_alert(alert_id: str) -> bool:
    """Reject an alert"""
    try:
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        return orch.reject_alert(alert_id)
    except Exception as e:
        print(f"Failed to reject alert: {e}")
        return False


def bulk_clear_alerts(asset_type: Optional[str] = None) -> int:
    """Clear all alerts (or by asset type) in one operation. Returns count cleared."""
    try:
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        return orch.bulk_clear_alerts(asset_type)
    except Exception as e:
        print(f"Failed to bulk clear alerts: {e}")
        return 0


def reject_all_alerts_for_symbol(symbol: str) -> int:
    """Reject all pending alerts for a specific symbol. Returns count rejected."""
    try:
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        return orch.reject_all_alerts_for_symbol(symbol)
    except Exception as e:
        print(f"Failed to reject alerts for {symbol}: {e}")
        return 0


def add_manual_alert(symbol: str, alert_type: str = "WATCH", asset_type: str = "crypto",
                     reasoning: str = "Manually added") -> bool:
    """Add a manual alert to the queue"""
    try:
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.add_alert(
            symbol=symbol,
            alert_type=alert_type,
            source="manual",
            asset_type=asset_type,
            reasoning=reasoning,
            confidence="MEDIUM"
        )
        return True
    except Exception as e:
        print(f"Failed to add alert: {e}")
        return False


# ============================================================
# ANALYSIS RESULTS
# ============================================================

def get_analysis_results() -> dict:
    """Get analysis results from all services"""
    try:
        if ANALYSIS_RESULTS_FILE.exists():
            with open(ANALYSIS_RESULTS_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                elif isinstance(data, list):
                    # Handle legacy list format if it exists
                    return {"legacy": {"results": data, "updated": datetime.now().isoformat()}}
    except Exception as e:
        print(f"Error loading analysis results: {e}")
    return {}


# ============================================================
# AI MONITOR EXCLUSIONS
# ============================================================

def load_ai_exclusions() -> list:
    """Load excluded pairs from AI position manager state"""
    try:
        if AI_POSITIONS_FILE.exists():
            with open(AI_POSITIONS_FILE, 'r') as f:
                state = json.load(f)
                return state.get("excluded_pairs", [])
    except Exception as e:
        print(f"Error loading AI exclusions: {e}")
    return []


def save_ai_exclusions(excluded_pairs: list) -> bool:
    """Save excluded pairs to AI position manager state"""
    try:
        state = {}
        if AI_POSITIONS_FILE.exists():
            with open(AI_POSITIONS_FILE, 'r') as f:
                state = json.load(f)
        
        state["excluded_pairs"] = excluded_pairs
        
        # Ensure directory exists
        AI_POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(AI_POSITIONS_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving AI exclusions: {e}")
        return False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_command(cmd: str) -> tuple:
    """Run a shell command and return (success, output)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def get_service_status(service_name: str) -> dict:
    """Get status of a service. Uses systemd/journalctl on Linux and sc/Get-Service on Windows."""
    if platform.system().lower().startswith('win'):
        windows_name_map = {
            'sentient-stock-monitor': 'SentientStockMonitor',
            'sentient-dex-launch': 'SentientDEXLaunch',
            'sentient-crypto-breakout': 'SentientCryptoBreakout',
            'sentient-discord-approval': 'SentientDiscordApproval',
            'sentient-crypto-ai-trader': 'SentientCryptoAI',
            'sentient-orb-fvg': 'SentientORBFVG'
        }
        svc_to_check = windows_name_map.get(service_name, service_name)
        success, output = run_command(f"sc query {svc_to_check}")
        is_active = success and ('RUNNING' in output)
        # Use sc qc to determine start type (auto/manual)
        success2, output2 = run_command(f"sc qc {svc_to_check}")
        enabled = False
        if success2 and 'AUTO_START' in output2.upper():
            enabled = True
        memory = 'N/A'
        # On Windows we don't have systemd status lines to parse memory from easily
        return {
            "active": is_active,
            "enabled": enabled,
            "status_text": "🟢 Running" if is_active else "🔴 Stopped",
            "boot_text": "Auto-start ON" if enabled else "Auto-start OFF",
            "memory": memory
        }
    else:
        # Linux (systemd)
        success, output = run_command(f"systemctl is-active {service_name}")
        is_active = output == "active"
        success2, output2 = run_command(f"systemctl is-enabled {service_name}")
        is_enabled = output2 == "enabled"
        success3, details = run_command(f"systemctl status {service_name} --no-pager -n 0")
        memory = "N/A"
        if "Memory:" in details:
            for line in details.split('\n'):
                if "Memory:" in line:
                    memory = line.split("Memory:")[1].strip().split()[0]
                    break
        return {
            "active": is_active,
            "enabled": is_enabled,
            "status_text": "🟢 Running" if is_active else "🔴 Stopped",
            "boot_text": "Auto-start ON" if is_enabled else "Auto-start OFF",
            "memory": memory
        }


def control_service(service_name: str, action: str) -> tuple:
    """Start, stop, restart, enable, or disable a service.
    Uses systemctl on Linux and sc/Get-Service on Windows."""
    if platform.system().lower().startswith('win'):
        # Map systemd-style names to Windows service names
        windows_name_map = {
            'sentient-stock-monitor': 'SentientStockMonitor',
            'sentient-dex-launch': 'SentientDEXLaunch',
            'sentient-crypto-breakout': 'SentientCryptoBreakout',
            'sentient-discord-approval': 'SentientDiscordApproval',
            'sentient-crypto-ai-trader': 'SentientCryptoAI',
            'sentient-orb-fvg': 'SentientORBFVG'
        }
        svc_to_control = windows_name_map.get(service_name, service_name)
        
        # On Windows try sc start/stop for built-in service control; nssm can also be used
        action_map = {
            'start': 'start',
            'stop': 'stop',
            'restart': 'stop'  # implement restart via stop+start
        }
        if action == 'restart':
            # perform stop then start
            ok1, out1 = run_command(f"sc stop {svc_to_control}")
            time.sleep(1)
            ok2, out2 = run_command(f"sc start {svc_to_control}")
            return ok1 and ok2, out1 + "\n" + out2
        elif action in ['start', 'stop']:
            return run_command(f"sc {action} {svc_to_control}")
        elif action == 'enable':
            return run_command(f"sc config {svc_to_control} start= auto")
        elif action == 'disable':
            return run_command(f"sc config {svc_to_control} start= disabled")
        else:
            return False, f"Unknown action: {action}"
    else:
        if action in ["start", "stop", "restart"]:
            cmd = f"sudo systemctl {action} {service_name}"
        elif action == "enable":
            cmd = f"sudo systemctl enable {service_name}"
        elif action == "disable":
            cmd = f"sudo systemctl disable {service_name}"
        else:
            return False, f"Unknown action: {action}"
        return run_command(cmd)


def get_service_logs(service_name: str, lines: int = 100) -> str:
    """Get recent logs for a service. Reads from log files first (both Windows and Linux), falls back to journalctl on Linux."""
    project_root = Path(__file__).resolve().parent
    
    # Map service to likely log filenames - MUST match systemd StandardOutput/StandardError paths
    log_map = {
        'sentient-stock-monitor': [
            'logs/stock_monitor_service.log',
            'logs/stock_monitor_error.log',
        ],
        'sentient-stock-ai-trader': [
            'logs/stock_ai_trader_service.log',
            'logs/stock_ai_trader_error.log',
        ],
        'sentient-dex-launch': [
            'logs/dex_launch_service.log',
            'logs/dex_launch_error.log'
        ],
        'sentient-crypto-breakout': [
            'logs/crypto_breakout_service.log',
            'logs/crypto_breakout_error.log'
        ],
        'sentient-crypto-ai-trader': [
            'logs/crypto_ai_position_manager_service.log',
            'logs/crypto_ai_position_manager_error.log'
        ],
        'sentient-discord-approval': [
            'logs/discord_approval_service.log',
            'logs/discord_approval_error.log',
        ],
        'sentient-orb-fvg': [
            'logs/orb_fvg_service.log',
            'logs/orb_fvg_error.log',
        ],
        'sentient-analysis-queue': [
            'logs/analysis_queue_service.log',
            'logs/analysis_queue_error.log',
        ],
    }
    
    candidates = log_map.get(service_name, [])
    
    # Try to find and read a log file first
    all_logs = []
    for candidate in candidates:
        # Handle both Windows and Linux path separators
        if platform.system().lower().startswith('win'):
            candidate = candidate.replace('/', '\\')
        log_path = project_root / candidate
        if log_path.exists():
            try:
                # Read file with explicit flush-friendly approach
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Seek to end to get file size, then read from appropriate position
                    f.seek(0, 2)  # Seek to end
                    file_size = f.tell()
                    
                    # For large files, read last ~100KB (should be plenty for recent logs)
                    read_size = min(file_size, 100 * 1024)
                    f.seek(max(0, file_size - read_size))
                    
                    # Skip partial first line if we didn't start at beginning
                    if file_size > read_size:
                        f.readline()  # Discard partial line
                    
                    file_lines = f.readlines()
                    
                if file_lines:
                    # Add header showing which file and timestamp
                    all_logs.append(f"=== {candidate} (last {min(lines, len(file_lines))} lines) ===\n")
                    all_logs.extend(file_lines[-lines:])
                    all_logs.append("\n")
            except Exception as e:
                all_logs.append(f"Error reading {candidate}: {e}\n")
    
    if all_logs:
        return ''.join(all_logs)
    
    # Fallback: On Linux, try journalctl if no log files found
    if not platform.system().lower().startswith('win'):
        # Try to get logs from journalctl with better formatting
        success, output = run_command(f"journalctl -u {service_name} -n {lines} --no-pager -o short-iso")
        if success and output and 'No entries' not in output:
            header = f"=== systemd journal for {service_name} (last {lines} lines) ===\n"
            header += f"=== Note: Log files not found - showing journal output ===\n\n"
            return header + output
        
        # Check if service exists at all
        success2, status = run_command(f"systemctl is-active {service_name}")
        if 'inactive' in status or 'dead' in status:
            return f"Service '{service_name}' is not running.\n\nNo log files found at:\n" + '\n'.join(f"  - {c}" for c in candidates)
        
        return f"No logs found for {service_name}.\n\nSearched files:\n" + '\n'.join(f"  - {c}" for c in candidates) + "\n\nJournal: " + output
    
    return f"No log files found for {service_name} (searched: {candidates})"


def clear_service_logs(service_name: str) -> tuple:
    """Clear/reset log files for a service. Returns (success, message)."""
    project_root = Path(__file__).resolve().parent
    
    # Map service to log filenames - MUST match systemd StandardOutput/StandardError paths
    log_map = {
        'sentient-stock-monitor': [
            'logs/stock_monitor_service.log',
            'logs/stock_monitor_error.log',
        ],
        'sentient-stock-ai-trader': [
            'logs/stock_ai_trader_service.log',
            'logs/stock_ai_trader_error.log',
        ],
        'sentient-dex-launch': [
            'logs/dex_launch_service.log',
            'logs/dex_launch_error.log'
        ],
        'sentient-crypto-breakout': [
            'logs/crypto_breakout_service.log',
            'logs/crypto_breakout_error.log'
        ],
        'sentient-crypto-ai-trader': [
            'logs/crypto_ai_position_manager_service.log',
            'logs/crypto_ai_position_manager_error.log'
        ],
        'sentient-discord-approval': [
            'logs/discord_approval_service.log',
            'logs/discord_approval_error.log',
        ],
        'sentient-orb-fvg': [
            'logs/orb_fvg_service.log',
            'logs/orb_fvg_error.log',
        ],
        'sentient-analysis-queue': [
            'logs/analysis_queue_service.log',
            'logs/analysis_queue_error.log',
        ],
    }
    
    candidates = log_map.get(service_name, [])
    cleared = []
    errors = []
    
    for candidate in candidates:
        if platform.system().lower().startswith('win'):
            candidate = candidate.replace('/', '\\')
        log_path = project_root / candidate
        
        if log_path.exists():
            try:
                # Truncate the file (clear contents but keep file)
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== Log cleared at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                cleared.append(candidate)
            except Exception as e:
                errors.append(f"{candidate}: {e}")
    
    if errors:
        return False, f"Errors: {'; '.join(errors)}"
    elif cleared:
        return True, f"Cleared: {', '.join(cleared)}"
    else:
        return False, "No log files found to clear"


# ============================================================
# AUTHENTICATION HELPERS
# ============================================================

AUTH_TOKENS_FILE = Path(__file__).resolve().parent / "data" / ".auth_tokens.json"

def load_auth_tokens() -> dict:
    """Load valid auth tokens"""
    try:
        if AUTH_TOKENS_FILE.exists():
            with open(AUTH_TOKENS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading auth tokens: {e}")
    return {}

def save_auth_tokens(tokens: dict):
    """Save auth tokens"""
    try:
        # Clean up expired tokens
        now = time.time()
        valid_tokens = {k: v for k, v in tokens.items() if v.get("expires", 0) > now}
        
        AUTH_TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AUTH_TOKENS_FILE, 'w') as f:
            json.dump(valid_tokens, f)
    except Exception as e:
        print(f"Error saving auth tokens: {e}")

def generate_auth_token() -> str:
    """Generate a secure random token"""
    import secrets
    return secrets.token_urlsafe(32)

def check_password():
    """Password + TOTP authentication with Persistence"""
    
    # Check for persistent session via URL token
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        
        # Check query params for token
        # Streamlit 1.30+ uses st.query_params
        try:
            query_params = st.query_params
        except:
            # Fallback for older versions
            query_params = st.experimental_get_query_params()
            
        auth_token = query_params.get("auth_token")
        if isinstance(auth_token, list):
            auth_token = auth_token[0]
            
        if auth_token:
            tokens = load_auth_tokens()
            token_data = tokens.get(auth_token)
            
            if token_data:
                if token_data.get("expires", 0) > time.time():
                    st.session_state.authenticated = True
                    st.session_state.auth_token = auth_token
                    # Update last used
                    token_data["last_used"] = time.time()
                    save_auth_tokens(tokens)
                    st.toast("✅ Auto-logged in via persistent session")
                else:
                    st.warning("⚠️ Session expired. Please log in again.")
    
    if "password_verified" not in st.session_state:
        st.session_state.password_verified = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Sentient Trader Control Panel")
        st.markdown("---")
        
        # Step 1: Password
        if not st.session_state.password_verified:
            password = st.text_input("🔑 Enter Password", type="password", key="pwd_input")
            
            if st.button("Verify Password", type="primary"):
                if password == ADMIN_PASSWORD:
                    st.session_state.password_verified = True
                    if not TOTP_ENABLED:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.rerun()
                else:
                    st.error("❌ Incorrect password")
                    print(f"[SECURITY] Failed login attempt at {datetime.now()}")
        
        # Step 2: TOTP (if enabled and password verified)
        elif TOTP_ENABLED and st.session_state.password_verified:
            st.success("✅ Password verified")
            st.markdown("---")
            
            # Show QR code option for first-time setup
            with st.expander("📱 First time? Scan QR Code"):
                # pyotp and qrcode were loaded dynamically above; assert for static checkers
                assert pyotp is not None, "pyotp module is required for TOTP functionality"
                assert qrcode is not None, "qrcode module is required for QR generation"
                try:
                    totp = pyotp.TOTP(TOTP_SECRET)
                    provisioning_uri = totp.provisioning_uri(
                        name="admin",
                        issuer_name="Sentient Trader"
                    )
                    
                    # Generate QR code
                    qr = qrcode.QRCode(version=1, box_size=5, border=2)
                    qr.add_data(provisioning_uri)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    
                    # Convert to base64 for display
                    buffer = io.BytesIO()
                    img.save(buffer)
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    
                    st.image(f"data:image/png;base64,{img_str}", caption="Scan with Google Authenticator / Authy")
                    st.caption(f"Or enter manually: `{TOTP_SECRET}`")
                except (binascii.Error, ValueError) as e:
                    st.error(f"❌ Invalid TOTP secret in configuration. Error: {str(e)}")
                    st.warning("Please regenerate your TOTP secret using: `python -c \"import pyotp; print(pyotp.random_base32())\"`")
                    st.caption(f"Current secret (first 20 chars): `{TOTP_SECRET[:20] if TOTP_SECRET else 'None'}...`")
            
            col_code, col_chk = st.columns([3, 1])
            with col_code:
                totp_code = st.text_input("📲 Enter 6-digit code from Authenticator", max_chars=6, key="totp_input")
            with col_chk:
                st.write("")
                st.write("")
                remember_me = st.checkbox("Remember Me (30 days)", value=True)
            
            if st.button("Verify Code", type="primary"):
                # runtime guard for static type checkers
                assert pyotp is not None, "pyotp module is required for TOTP functionality"
                try:
                    totp = pyotp.TOTP(TOTP_SECRET)
                    if totp.verify(totp_code, valid_window=1):
                        st.session_state.authenticated = True
                        
                        # Handle Remember Me
                        if remember_me:
                            token = generate_auth_token()
                            tokens = load_auth_tokens()
                            tokens[token] = {
                                "created": time.time(),
                                "expires": time.time() + (30 * 24 * 3600),  # 30 days
                                "user_agent": "user"
                            }
                            save_auth_tokens(tokens)
                            
                            # Set query param for bookmarking
                            try:
                                st.query_params["auth_token"] = token
                            except:
                                st.experimental_set_query_params(auth_token=token)
                                
                            st.session_state.auth_token = token
                            st.success("✅ Login successful! Bookmark this URL to stay logged in.")
                            time.sleep(1)
                        
                        st.rerun()
                    else:
                        st.error("❌ Invalid code. Try again.")
                        print(f"[SECURITY] Failed TOTP attempt at {datetime.now()}")
                except (binascii.Error, ValueError) as e:
                    st.error(f"❌ Configuration Error: Invalid TOTP secret. {str(e)}")
                    st.warning("The TOTP secret in your .env file is not valid base32 format.")
                    st.info("To fix: Regenerate a new secret using: `python -c \"import pyotp; print(pyotp.random_base32())\"`")
                    print(f"[ERROR] TOTP secret validation failed: {str(e)}")
        
        # No TOTP, password was enough
        elif st.session_state.password_verified and not TOTP_ENABLED:
            st.session_state.authenticated = True
            st.rerun()
        
        st.markdown("---")
        
        # Show security status
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Password required")
        with col2:
            if TOTP_ENABLED:
                st.success("✅ 2FA enabled")
            else:
                st.warning("⚠️ 2FA not configured")
                # If a secret exists but the optional deps are missing, show actionable error
                if TOTP_SECRET and not TOTP_AVAILABLE:
                    st.error(
                        "TOTP secret is set but required libraries (`pyotp`, `qrcode`) are not installed.\n"
                        "Install them with: `pip install -r requirements.txt` or `pip install pyotp qrcode[pil]`"
                    )
        
        if not TOTP_ENABLED:
            st.info("💡 To enable 2FA, set `CONTROL_PANEL_TOTP_SECRET` in your .env file")
        
        return False
    
    # Already authenticated
    # Ensure token is in URL if we have one in session (for bookmarking)
    if "auth_token" in st.session_state:
        try:
            if st.query_params.get("auth_token") != st.session_state.auth_token:
                st.query_params["auth_token"] = st.session_state.auth_token
        except:
            # Fallback
            pass
            
    return True


# ============================================================
# WATCHLIST UI
# ============================================================

def _render_ticker_category(tm, tickers_data: list, category_name: str, ticker_type: str, key_prefix: str):
    """
    Helper function to render a ticker category (stocks/crypto) with add/remove functionality.
    
    Args:
        tm: TickerManager instance
        tickers_data: List of ticker dicts from Supabase
        category_name: Display name for the category (e.g., "Stocks", "Crypto")
        ticker_type: Type filter for adding new tickers (e.g., "stock", "crypto")
        key_prefix: Unique prefix for Streamlit widget keys
    """
    # Add new ticker
    with st.expander(f"➕ Add {category_name}"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_ticker = st.text_input("Symbol", key=f"{key_prefix}_new_ticker").upper()
        with col2:
            if ticker_type == "stock":
                sub_type = st.selectbox("Sub-type", ["stock", "penny_stock"], key=f"{key_prefix}_sub_type")
            else:
                sub_type = ticker_type
        
        if st.button(f"Add {category_name}", type="primary", key=f"{key_prefix}_add_btn"):
            if new_ticker:
                if tm and tm.add_ticker(new_ticker, ticker_type=sub_type if ticker_type == "stock" else ticker_type):
                    st.success(f"Added {new_ticker} to {category_name}")
                    st.rerun()
                else:
                    st.error("Failed to add ticker")
    
    # List/Manage tickers for this category
    if tickers_data:
        ticker_list = [t['ticker'] for t in tickers_data]
        
        # Quick stats
        st.caption(f"📊 {len(ticker_list)} {category_name.lower()} in your watchlist")
        
        # Quick action buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✅ Select All", key=f"{key_prefix}_select_all", use_container_width=True):
                st.session_state[f"{key_prefix}_selected"] = ticker_list.copy()
                st.rerun()
        with btn_col2:
            if st.button("❌ Clear Selection", key=f"{key_prefix}_clear_all", use_container_width=True):
                st.session_state[f"{key_prefix}_selected"] = []
                st.rerun()
        
        # Initialize session state for selections
        if f"{key_prefix}_selected" not in st.session_state:
            st.session_state[f"{key_prefix}_selected"] = ticker_list.copy()
        
        selected = st.multiselect(
            f"Current {category_name}",
            options=ticker_list,
            default=st.session_state.get(f"{key_prefix}_selected", ticker_list),
            key=f"{key_prefix}_ticker_list"
        )
        
        # Check for removals
        to_remove = set(ticker_list) - set(selected)
        if to_remove:
            st.warning(f"⚠️ {len(to_remove)} ticker(s) will be removed")
            if st.button(f"🗑️ Remove {len(to_remove)} ticker(s)?", key=f"{key_prefix}_remove_btn", type="secondary"):
                for t in to_remove:
                    if tm:
                        tm.remove_ticker(t)
                st.success(f"Removed {len(to_remove)} {category_name.lower()}")
                st.rerun()
    else:
        st.info(f"No {category_name.lower()} found in your watchlist. Add some above!")


def render_watchlist_manager():
    """Render the Watchlist Manager section with separate Stock and Crypto tabs"""
    st.header("📋 Watchlist Manager")
    
    # Try to import TickerManager
    try:
        from services.ticker_manager import TickerManager
        tm = TickerManager()
        supabase_available = tm.test_connection()
    except Exception as e:
        print(f"TickerManager import failed: {e}")
        supabase_available = False
        tm = None

    tab1, tab2, tab3 = st.tabs(["📈 My Tickers (Supabase)", "⚙️ Service Watchlists", "🚫 AI Exclusions"])
    
    # Tab 1: Global "My Tickers" (Supabase) - Now with Stock/Crypto separation
    with tab1:
        if supabase_available:
            st.caption("✅ Connected to Supabase 'saved_tickers'")
            
            # Fetch all tickers once
            if tm:
                all_tickers = tm.get_all_tickers(limit=1000)
            else:
                all_tickers = []
            
            # Separate by type
            stock_tickers = [t for t in all_tickers if t.get('type') in ['stock', 'penny_stock', None]]
            crypto_tickers = [t for t in all_tickers if t.get('type') == 'crypto']
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total", len(all_tickers))
            with col2:
                st.metric("📈 Stocks", len(stock_tickers))
            with col3:
                st.metric("🪙 Crypto", len(crypto_tickers))
            
            st.markdown("---")
            
            # Sub-tabs for Stocks and Crypto
            stock_tab, crypto_tab = st.tabs(["📈 Stocks", "🪙 Crypto"])
            
            with stock_tab:
                _render_ticker_category(
                    tm=tm,
                    tickers_data=stock_tickers,
                    category_name="Stocks",
                    ticker_type="stock",
                    key_prefix="sb_stocks"
                )
            
            with crypto_tab:
                _render_ticker_category(
                    tm=tm,
                    tickers_data=crypto_tickers,
                    category_name="Crypto",
                    ticker_type="crypto",
                    key_prefix="sb_crypto"
                )
        else:
            st.warning("⚠️ Supabase connection not available. Ensure credentials are in .env")
            st.info("Use 'Service Watchlists' tab for local configuration.")

    # Tab 2: Service Watchlists (Synced)
    with tab2:
        st.caption("Service watchlists (Synced with Supabase + Local JSON backup)")
        
        service_names = list(SERVICES.keys())
        selected_service = st.selectbox("Select Service", service_names)
        
        if selected_service:
            svc_info = SERVICES[selected_service]
            svc_key = svc_info['name']
            # Define Supabase watchlist name for this service
            sup_watchlist_name = f"service_{svc_key}"
            
            # 1. Load current watchlist (Try Supabase first, then Local)
            current_watchlist = []
            source = "Local"
            
            if tm and supabase_available:
                # Try to get from Supabase
                sup_tickers = tm.get_watchlist_tickers(sup_watchlist_name)
                if sup_tickers:
                    current_watchlist = sup_tickers
                    source = "Supabase ☁️"
                else:
                    # Fallback to local if Supabase empty (first time sync?)
                    current_watchlist = get_service_watchlist(svc_key, svc_info.get('category', 'stocks'))
                    if current_watchlist:
                        source = "Local 📂 (Will sync to Supabase on save)"
            else:
                current_watchlist = get_service_watchlist(svc_key, svc_info.get('category', 'stocks'))
            
            st.caption(f"Source: {source}")
            
            # Custom input
            col1, col2 = st.columns([3, 1])
            with col1:
                custom_add = st.text_input("Add Custom Ticker", key=f"custom_{svc_key}").upper()
            with col2:
                if st.button("Add", key=f"btn_add_{svc_key}"):
                    if custom_add and custom_add not in current_watchlist:
                        current_watchlist.append(custom_add)
                        
                        # Save to Local
                        set_service_watchlist(svc_key, current_watchlist)
                        
                        # Save to Supabase
                        if tm and supabase_available:
                            # Ensure watchlist exists
                            tm.create_watchlist(sup_watchlist_name, description=f"Watchlist for {selected_service}")
                            tm.add_to_watchlist(sup_watchlist_name, custom_add)
                            
                        st.success(f"Added {custom_add}")
                        st.rerun()
            
            # Multiselect with clear/all
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("Select All Default", key=f"all_{svc_key}"):
                    default = DEFAULT_WATCHLISTS.get(svc_info.get('category', 'stocks'), [])
                    new_list = list(set(current_watchlist + default))
                    
                    # Save Local
                    set_service_watchlist(svc_key, new_list)
                    
                    # Save Supabase (Bulk add)
                    if tm and supabase_available:
                        tm.create_watchlist(sup_watchlist_name, description=f"Watchlist for {selected_service}")
                        for t in new_list:
                            if t not in current_watchlist:
                                tm.add_to_watchlist(sup_watchlist_name, t)
                                
                    st.rerun()
            with col_act2:
                if st.button("Clear All", key=f"clear_{svc_key}"):
                    # Save Local
                    set_service_watchlist(svc_key, [])
                    
                    # Save Supabase (Delete and Recreate empty)
                    if tm and supabase_available:
                         tm.delete_watchlist(sup_watchlist_name)
                         tm.create_watchlist(sup_watchlist_name, description=f"Watchlist for {selected_service}")
                         
                    st.rerun()

            # Combine with defaults for options, but selection is current_watchlist
            options = list(set(current_watchlist + DEFAULT_WATCHLISTS.get(svc_info.get('category', 'stocks'), [])))
            
            updated_list = st.multiselect(
                f"Watchlist for {selected_service}",
                options=options,
                default=current_watchlist,
                key=f"multi_{svc_key}"
            )
            
            # Save changes if different
            if set(updated_list) != set(current_watchlist):
                # 1. Update Local JSON (Source of Truth for running services)
                set_service_watchlist(svc_key, updated_list)
                
                # 2. Update Supabase (Sync for other devices)
                if tm and supabase_available:
                    # Ensure watchlist exists
                    tm.create_watchlist(sup_watchlist_name, description=f"Watchlist for {selected_service}")
                    
                    # Calculate diff
                    to_add = set(updated_list) - set(current_watchlist)
                    to_remove = set(current_watchlist) - set(updated_list)
                    
                    # Apply diff
                    for t in to_add:
                        tm.add_to_watchlist(sup_watchlist_name, t)
                    for t in to_remove:
                        tm.remove_from_watchlist(sup_watchlist_name, t)
                        
                    st.toast(f"Watchlist updated & synced to Supabase! (+{len(to_add)} / -{len(to_remove)})")
                else:
                    st.toast("Watchlist updated locally!")
                    
                time.sleep(0.5)
                st.rerun()

    # Tab 3: AI Monitor Exclusions
    with tab3:
        st.caption("🚫 Manage coins permanently excluded from AI monitoring")
        st.info("ℹ️ **Note:** Pending trade approvals are managed in the main Trading App. To clear all pending approvals from here, you can **Restart** the 'AI Crypto Trader' service.")
        
        # Load exclusions
        excluded = load_ai_exclusions()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_exclude = st.text_input("Exclude Pair (e.g., BTC/USD)", key="new_exclude").upper()
        with col2:
            if st.button("Add Exclusion", type="primary"):
                if new_exclude and new_exclude not in excluded:
                    excluded.append(new_exclude)
                    if save_ai_exclusions(excluded):
                        st.success(f"Excluded {new_exclude}")
                        st.rerun()
                    else:
                        st.error("Failed to save exclusion")
        
        if excluded:
            st.write("### Excluded Pairs")
            
            # Display as removable tags/multiselect
            to_keep = st.multiselect(
                "Currently Excluded",
                options=excluded,
                default=excluded,
                key="excluded_multiselect_v2",
                help="Deselect to re-include pair in monitoring"
            )
            
            # Detect changes
            if set(to_keep) != set(excluded):
                if st.button("Apply Changes"):
                    if save_ai_exclusions(to_keep):
                        st.success("Updated exclusions list")
                        # Offer to restart service
                        if st.button("Restart AI Trader Service?"):
                            control_service("sentient-crypto-ai-trader", "restart")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to save changes")
        else:
            st.info("No pairs currently excluded.")


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Sentient Trader Control",
        page_icon="🤖",
        layout="wide"
    )
    
    # Check authentication
    if not check_password():
        return
    
    # Header
    st.title("🤖 Sentient Trader Control Panel")
    
    # Logout button in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        if st.button("🚪 Logout"):
            # Clear session state
            st.session_state.authenticated = False
            st.session_state.password_verified = False
            
            # Clear auth token if present
            if "auth_token" in st.session_state:
                token = st.session_state.auth_token
                # Remove from file
                tokens = load_auth_tokens()
                if token in tokens:
                    del tokens[token]
                    save_auth_tokens(tokens)
                del st.session_state.auth_token
            
            # Clear query params
            try:
                st.query_params.clear()
            except:
                st.experimental_set_query_params()
                
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"**Last refresh:** {datetime.now().strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        # Auto-refresh option - uses query param trick
        st.markdown("---")
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=False, key="auto_refresh", 
                                 help="Auto-refresh every 30 seconds")
        if auto_refresh:
            time.sleep(30)
            st.rerun()

    # Tabs
    tab_workflow, tab_status, tab_watchlist, tab_analysis, tab_risk, tab_discord, tab_logs = st.tabs([
        "🎯 Workflow",
        "📊 Service Status", 
        "📋 Watchlists",
        "🔍 Analysis",
        "💰 Risk Profile",
        "💬 Discord", 
        "📝 Logs"
    ])
    
    # ============================================================
    # WORKFLOW TAB - New unified control center
    # ============================================================
    with tab_workflow:
        st.markdown("### 🎯 Workflow Control Center")
        st.caption("One-click modes to manage all services safely")
        
        # Get orchestrator data
        orch_data = get_orchestrator_dashboard()
        
        if orch_data:
            current_mode = orch_data.get("mode", "stopped")
            
            # Mode indicator
            mode_colors = {
                "stopped": "🔴",
                "safe": "🟡",
                "discovery": "🟢",
                "active": "🔵",
                "aggressive": "🟣"
            }
            st.info(f"{mode_colors.get(current_mode, '⚪')} **Current Mode: {current_mode.upper()}**")
            
            # Mode buttons
            st.markdown("#### Quick Modes")
            mode_cols = st.columns(5)
            
            modes = [
                ("stopped", "⏹️ Stop All", "All services stopped"),
                ("safe", "🛡️ Safe", "Only AI Position Manager (monitor existing)"),
                ("discovery", "🔍 Discovery", "Scanners ON, alerts queue for review"),
                ("active", "🚀 Active", "Full crypto automation with approvals"),
                ("aggressive", "⚡ Aggressive", "Fast intervals, all services")
            ]
            
            for i, (mode_key, label, desc) in enumerate(modes):
                with mode_cols[i]:
                    is_current = current_mode == mode_key
                    if st.button(
                        label,
                        key=f"mode_{mode_key}",
                        type="primary" if is_current else "secondary",
                        use_container_width=True,
                        disabled=is_current
                    ):
                        if set_workflow_mode(mode_key):
                            st.toast(f"✅ Switched to {label}")
                            # Apply mode to actual services
                            if mode_key == "stopped":
                                for svc in SERVICES.values():
                                    control_service(svc["name"], "stop")
                            elif mode_key == "safe":
                                for svc_name, svc in SERVICES.items():
                                    if svc["name"] in ["sentient-crypto-ai-trader", "sentient-discord-approval"]:
                                        control_service(svc["name"], "start")
                                    else:
                                        control_service(svc["name"], "stop")
                            elif mode_key == "discovery":
                                for svc_name, svc in SERVICES.items():
                                    if svc.get("category") == "crypto" or svc["name"] == "sentient-discord-approval":
                                        control_service(svc["name"], "start")
                                    else:
                                        control_service(svc["name"], "stop")
                            elif mode_key == "active":
                                for svc_name, svc in SERVICES.items():
                                    if svc.get("category") in ["crypto", "infrastructure"]:
                                        control_service(svc["name"], "start")
                            elif mode_key == "aggressive":
                                for svc in SERVICES.values():
                                    control_service(svc["name"], "start")
                            time.sleep(1)
                            st.rerun()
                    st.caption(desc)
            
            st.markdown("---")
            
            # ============================================================
            # ALERT QUEUE
            # ============================================================
            st.markdown("### 📥 Alert Queue")
            st.caption("Alerts from Discord and scanners waiting for your review")
            
            # Add manual alert
            with st.expander("➕ Add Alert Manually"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    manual_symbol = st.text_input("Symbol", placeholder="BTC, ETH, NVDA", key="manual_alert_symbol").upper()
                with col2:
                    manual_type = st.selectbox("Type", ["WATCH", "ENTRY", "BREAKOUT"], key="manual_alert_type")
                with col3:
                    manual_asset = st.selectbox("Asset", ["crypto", "stock"], key="manual_alert_asset")
                
                manual_reason = st.text_input("Reasoning (optional)", placeholder="Saw on Twitter...", key="manual_alert_reason")
                
                if st.button("Add to Queue", type="primary", key="add_manual_alert"):
                    if manual_symbol:
                        if add_manual_alert(manual_symbol, manual_type, manual_asset, manual_reason or "Manually added"):
                            st.toast(f"✅ Added {manual_symbol} to queue")
                            st.rerun()
                        else:
                            st.error("Failed to add alert")
                    else:
                        st.warning("Enter a symbol")
            
            # Get pending alerts
            pending_crypto = get_pending_alerts("crypto")
            pending_stocks = get_pending_alerts("stock")
            
            # Bulk actions row
            bulk_col1, bulk_col2, bulk_col3 = st.columns([1, 1, 2])
            with bulk_col1:
                if st.button("🗑️ Clear All Crypto", key="clear_all_crypto", use_container_width=True, disabled=len(pending_crypto)==0):
                    cleared = bulk_clear_alerts("crypto")
                    st.toast(f"❌ Cleared {cleared} crypto alerts")
                    time.sleep(0.3)
                    st.rerun()
            with bulk_col2:
                if st.button("🗑️ Clear All Stocks", key="clear_all_stocks", use_container_width=True, disabled=len(pending_stocks)==0):
                    cleared = bulk_clear_alerts("stock")
                    st.toast(f"❌ Cleared {cleared} stock alerts")
                    time.sleep(0.3)
                    st.rerun()
            with bulk_col3:
                st.caption(f"Total: {len(pending_crypto) + len(pending_stocks)} pending alerts")
            
            # Use stable tab labels (counts shown inside tabs, not in labels)
            alert_tab1, alert_tab2 = st.tabs(["🪙 Crypto", "📈 Stocks"])
            
            def render_alert_actions(alert, asset_type):
                """Render action buttons for an alert"""
                # Analysis mode selector
                analysis_modes = {
                    "crypto": [("🔬 Standard", "crypto_standard"), ("🎯 Multi", "crypto_multi"), ("🚀 Ultimate", "crypto_ultimate")],
                    "stock": [("📈 Momentum", "stock_momentum"), ("🎯 ORB+FVG", "orb_fvg_scan")]
                }
                
                # Count how many alerts exist for this symbol
                symbol = alert['symbol']
                all_pending = get_pending_alerts(asset_type)
                symbol_alert_count = sum(1 for a in all_pending if a['symbol'].upper() == symbol.upper())
                
                col_acts = st.columns(6)
                
                with col_acts[0]:
                        if st.button("✅ Watchlist", key=f"approve_{alert['id']}", use_container_width=True, help="Add to Watchlist"):
                            success = False
                            with st.spinner("Adding..."):
                                success = approve_alert(alert['id'])
                            
                            if success:
                                st.rerun()
                
                # Analysis buttons - show mode options
                modes = analysis_modes.get(asset_type, analysis_modes["crypto"])
                with col_acts[1]:
                    if st.button(modes[0][0], key=f"analyze_std_{alert['id']}", use_container_width=True, help="Standard analysis"):
                        if queue_analysis_request(modes[0][1], [alert['symbol']]):
                            st.toast(f"🔬 Standard analysis queued for {alert['symbol']}")
                        else:
                            st.error("Failed to queue analysis")
                
                with col_acts[2]:
                    if st.button(modes[1][0], key=f"analyze_multi_{alert['id']}", use_container_width=True, help="Multi-config analysis"):
                        if queue_analysis_request(modes[1][1], [alert['symbol']]):
                            st.toast(f"🎯 Multi analysis queued for {alert['symbol']}")
                        else:
                            st.error("Failed to queue analysis")
                
                with col_acts[3]:
                    if asset_type == "crypto" and len(modes) > 2:
                        if st.button(modes[2][0], key=f"analyze_ult_{alert['id']}", use_container_width=True, help="Ultimate analysis"):
                            if queue_analysis_request(modes[2][1], [alert['symbol']]):
                                st.toast(f"🚀 Ultimate analysis queued for {alert['symbol']}")
                            else:
                                st.error("Failed to queue analysis")
                    else:
                        # Trade button for stocks or placeholder
                        if st.button("🚀 Trade", key=f"trade_{alert['id']}", use_container_width=True, help="Open Trade Setup"):
                            st.info("⚠️ Use 'Quick Trade' tab in main app to execute")
                
                with col_acts[4]:
                    if st.button("🗑️", key=f"reject_{alert['id']}", use_container_width=True, help="Dismiss This Alert"):
                        if reject_alert(alert['id']):
                            st.toast(f"❌ {alert['symbol']} dismissed")
                            time.sleep(0.5)
                            st.rerun()
                
                with col_acts[5]:
                    # Show button to remove all alerts for this symbol if there are multiple
                    if symbol_alert_count > 1:
                        help_text = f"Remove all {symbol_alert_count} alerts for {symbol}"
                        if st.button("🗑️ All", key=f"reject_all_{alert['id']}", use_container_width=True, help=help_text):
                            count = reject_all_alerts_for_symbol(symbol)
                            if count > 0:
                                st.toast(f"❌ Removed {count} alert(s) for {symbol}")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("Failed to remove alerts")
                    else:
                        st.empty()  # Empty space if only one alert

            with alert_tab1:
                # Show count inside the tab
                st.caption(f"**{len(pending_crypto)}** pending crypto alert(s)")
                if pending_crypto:
                    for alert in pending_crypto:
                        alert_container = st.empty()
                        with alert_container.container():
                            # Determine color/icon
                            conf = alert.get('confidence', 'MEDIUM')
                            icon = "🔥" if conf == "HIGH" else "⚠️" if conf == "LOW" else "🔔"
                            color = "red" if conf == "LOW" else "orange" if conf == "MEDIUM" else "green"
                            
                            with st.expander(f"{icon} **{alert['symbol']}** - {alert['alert_type']} ({conf})", expanded=True):
                                # Alert Details
                                st.markdown(f"**Reasoning:** {alert.get('reasoning', 'No details')}")
                                
                                # Price Info
                                cols = st.columns(3)
                                with cols[0]:
                                    price = alert.get('price')
                                    st.write(f"**Price:** {f'${price:.4f}' if price else 'N/A'}")
                                with cols[1]:
                                    target = alert.get('target')
                                    st.write(f"**Target:** {f'${target:.4f}' if target else 'N/A'}")
                                with cols[2]:
                                    stop = alert.get('stop_loss')
                                    st.write(f"**Stop:** {f'${stop:.4f}' if stop else 'N/A'}")
                                
                                st.caption(f"Source: {alert['source']} | Time: {alert['timestamp'].split('T')[1][:8]}")
                                
                                # Actions
                                st.markdown("---")
                                render_alert_actions(alert, "crypto")
                else:
                    st.info("No pending crypto alerts. Alerts from Discord and scanners will appear here.")
            
            with alert_tab2:
                # Show count inside the tab
                st.caption(f"**{len(pending_stocks)}** pending stock alert(s)")
                if pending_stocks:
                    for alert in pending_stocks:
                        conf = alert.get('confidence', 'MEDIUM')
                        icon = "🔥" if conf == "HIGH" else "⚠️" if conf == "LOW" else "🔔"
                        
                        with st.expander(f"{icon} **{alert['symbol']}** - {alert['alert_type']} ({conf})", expanded=True):
                            st.markdown(f"**Reasoning:** {alert.get('reasoning', 'No details')}")
                            
                            cols = st.columns(3)
                            with cols[0]:
                                price = alert.get('price')
                                st.write(f"**Price:** {f'${price:.2f}' if price else 'N/A'}")
                            with cols[1]:
                                target = alert.get('target')
                                st.write(f"**Target:** {f'${target:.2f}' if target else 'N/A'}")
                            with cols[2]:
                                stop = alert.get('stop_loss')
                                st.write(f"**Stop:** {f'${stop:.2f}' if stop else 'N/A'}")
                                
                            st.caption(f"Source: {alert['source']} | Time: {alert['timestamp'].split('T')[1][:8]}")
                            
                            st.markdown("---")
                            render_alert_actions(alert, "stock")
                else:
                    st.info("No pending stock alerts.")
            
            st.markdown("---")
            
            # ============================================================
            # SERVICE HEALTH SUMMARY
            # ============================================================
            st.markdown("### 🏥 Service Health")
            
            svc_data = orch_data.get("services", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Running", f"{svc_data.get('running', 0)}/{svc_data.get('total', 0)}")
            with col2:
                st.metric("Errors", svc_data.get('error', 0))
            with col3:
                alerts_data = orch_data.get("alerts", {})
                st.metric("Pending Alerts", alerts_data.get('pending_total', 0))
            
            # Quick service status
            with st.expander("📊 Service Details"):
                for svc_name, svc_info in svc_data.get("details", {}).items():
                    state = svc_info.get("state", "unknown")
                    state_emoji = "🟢" if state == "running" else "🔴" if state == "stopped" else "🟡"
                    health = svc_info.get("health", {})
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"{state_emoji} **{svc_info.get('display_name', svc_name)}**")
                    with col2:
                        st.caption(f"✅ {health.get('success_count', 0)} runs")
                    with col3:
                        st.caption(f"❌ {health.get('error_count', 0)} errors")
        else:
            st.warning("⚠️ Orchestrator not available. Using legacy service control.")
            st.info("The orchestrator provides unified service management. Check if `services/service_orchestrator.py` exists.")
    
    with tab_watchlist:
        render_watchlist_manager()
    
    with tab_status:
        st.markdown("### Service Status")
        
        cols = st.columns(3)
        
        for i, (svc_label, svc_info) in enumerate(SERVICES.items()):
            svc_name = svc_info['name']
            with cols[i % 3]:
                st.markdown(f"#### {svc_info['emoji']} {svc_label}")
                st.caption(svc_info['description'])
                
                status = get_service_status(svc_name)
                
                # Status indicator
                st.markdown(f"**Status:** {status['status_text']}")
                st.markdown(f"**Boot:** {status['boot_text']}")
                if status['memory'] != 'N/A':
                    st.markdown(f"**Memory:** {status['memory']}")
                
                # Controls
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Start", key=f"tab_start_{svc_name}", disabled=status['active']):
                        ok, msg = control_service(svc_name, "start")
                        if ok:
                            st.success("Started")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Error: {msg}")
                
                with c2:
                    if st.button("Stop", key=f"tab_stop_{svc_name}", disabled=not status['active']):
                        ok, msg = control_service(svc_name, "stop")
                        if ok:
                            st.success("Stopped")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Error: {msg}")
                
                with c3:
                    if st.button("Restart", key=f"tab_restart_{svc_name}"):
                        ok, msg = control_service(svc_name, "restart")
                        if ok:
                            st.success("Restarted")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Error: {msg}")
                
                # Interval config - only show for services that support intervals
                if svc_info.get("interval_key"):
                    current_interval = get_service_interval(svc_name, svc_info)
                    interval_min = svc_info.get("interval_min", 10)
                    interval_max = svc_info.get("interval_max", 3600)
                    # Ensure current_interval is within valid range
                    current_interval = max(interval_min, min(current_interval, interval_max))
                    new_interval = st.number_input(
                        "Scan Interval (sec)", 
                        min_value=interval_min,
                        max_value=interval_max,
                        value=current_interval,
                        key=f"tab_interval_{svc_name}"
                    )
                    
                    if new_interval != current_interval:
                        if st.button("Update Interval", key=f"tab_update_{svc_name}"):
                            if set_service_interval(svc_name, svc_info, int(new_interval)):
                                st.success("Interval updated")
                                time.sleep(1)
                                st.rerun()
                
                # Stock Discovery Config - only for Stock Monitor
                if svc_name == "sentient-stock-monitor":
                    st.markdown("---")
                    try:
                        from ui.discovery_config_ui import render_discovery_config_panel
                        render_discovery_config_panel()
                    except ImportError as e:
                        st.warning(f"Discovery config UI not available: {e}")
                    except Exception as e:
                        st.error(f"Error loading discovery config: {e}")
                
                st.markdown("---")

    with tab_analysis:
        
        st.markdown("---")
        st.markdown("### 🎛️ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start\nAll", use_container_width=True):
                for svc in SERVICES.values():
                    control_service(svc["name"], "start")
                st.rerun()
        with col2:
            if st.button("⏹️ Stop\nAll", use_container_width=True):
                for svc in SERVICES.values():
                    control_service(svc["name"], "stop")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Overview")
        running = sum(1 for s in SERVICES.values() if get_service_status(s["name"])["active"])
        st.metric("Services Running", f"{running}/{len(SERVICES)}")
        
        # ============================================================
        # STRATEGY CONFIG (Mobile-Friendly)
        # ============================================================
        st.markdown("---")
        st.markdown("### 🎯 Active Strategy")
        
        strategy_data = load_active_strategy()
        current_strategy = strategy_data.get("active_strategy", "PAPER_TRADING")
        available_strategies = strategy_data.get("available_strategies", {})
        
        # Show current strategy
        current_info = available_strategies.get(current_strategy, {})
        st.info(f"**{current_info.get('name', current_strategy)}**")
        st.caption(current_info.get('description', ''))
        
        # Strategy switcher dropdown
        strategy_options = list(available_strategies.keys())
        if strategy_options:
            selected = st.selectbox(
                "Switch Strategy",
                options=strategy_options,
                index=strategy_options.index(current_strategy) if current_strategy in strategy_options else 0,
                format_func=lambda x: available_strategies.get(x, {}).get('name', x),
                key="strategy_selector",
                label_visibility="collapsed"
            )
            
            if selected and selected != current_strategy:
                if st.button("🔄 Apply Strategy", type="primary", use_container_width=True):
                    success, msg = switch_strategy(str(selected))
                    if success:
                        st.toast(f"✅ {msg}")
                        # Restart relevant services
                        control_service("sentient-stock-monitor", "restart")
                        st.rerun()
                    else:
                        st.error(msg)
        
        # ============================================================
        # QUICK ANALYSIS (Mobile-Friendly)
        # ============================================================
        st.markdown("---")
        st.markdown("### 🔬 Quick Analysis")
        st.caption("Trigger on-demand scans from your phone!")
        
        # Crypto Analysis Modes - THE BIG 3
        st.markdown("#### 🪙 Crypto Analysis")
        crypto_col1, crypto_col2, crypto_col3 = st.columns(3)
        
        with crypto_col1:
            if st.button("🔬 Standard", key="crypto_standard_btn", use_container_width=True, type="primary"):
                if queue_analysis_request("crypto_standard"):
                    st.toast("✅ Standard crypto analysis queued!")
            st.caption("Single strategy")
        
        with crypto_col2:
            if st.button("🎯 Multi", key="crypto_multi_btn", use_container_width=True):
                if queue_analysis_request("crypto_multi"):
                    st.toast("✅ Multi-config analysis queued!")
            st.caption("Long/Short + Leverage")
        
        with crypto_col3:
            if st.button("🚀 Ultimate", key="crypto_ultimate_btn", use_container_width=True):
                if queue_analysis_request("crypto_ultimate"):
                    st.toast("✅ Ultimate analysis queued!")
            st.caption("ALL combinations")
        
        # Quick scan button
        if st.button("⚡ Quick Scan (BTC/ETH/SOL)", key="quick_crypto_btn", use_container_width=True):
            if queue_analysis_request("quick_crypto"):
                st.toast("✅ Quick crypto scan queued!")
        
        st.markdown("#### 📈 Stock Analysis")
        stock_col1, stock_col2 = st.columns(2)
        
        with stock_col1:
            if st.button("📈 Momentum Scan", key="stock_momentum_btn", use_container_width=True):
                if queue_analysis_request("stock_momentum"):
                    st.toast("✅ Stock momentum scan queued!")
        
        with stock_col2:
            if st.button("🎯 ORB+FVG", key="orb_fvg_btn", use_container_width=True):
                if queue_analysis_request("orb_fvg_scan"):
                    st.toast("✅ ORB+FVG scan queued!")
        
        # Custom analysis input
        with st.expander("📝 Custom Analysis"):
            custom_tickers = st.text_input(
                "Tickers (comma-separated)",
                placeholder="BTC/USD, ETH/USD or NVDA, TSLA",
                key="custom_analysis_tickers"
            )
            custom_col1, custom_col2 = st.columns(2)
            with custom_col1:
                custom_asset = st.selectbox("Asset Type", ["crypto", "stock"], key="custom_asset_type")
            with custom_col2:
                custom_mode = st.selectbox("Analysis Mode", ["standard", "multi_config", "ultimate"], key="custom_analysis_mode")
            
            if st.button("🚀 Run Custom", use_container_width=True, type="primary"):
                tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
                if tickers:
                    if queue_analysis_request("custom", tickers):
                        st.toast(f"✅ Custom analysis for {len(tickers)} tickers queued!")
                else:
                    st.warning("Enter at least one ticker")
        
        # Show pending requests count
        requests = get_analysis_requests()
        pending = [r for r in requests if r.get("status") == "pending"]
        if pending:
            st.caption(f"📋 {len(pending)} pending request(s)")
            if st.button("🗑️ Clear Queue", key="clear_analysis_queue"):
                if clear_analysis_requests():
                    st.toast("✅ Queue cleared!")
                time.sleep(0.3)
                st.rerun()
    
    # Main content - Service Cards by Category
    st.markdown("---")
    
    # Group services by category
    categories = {
        "crypto": ("🪙 Crypto Services", []),
        "stocks": ("📈 Stock Services", []),
        "infrastructure": ("🔧 Infrastructure", [])
    }
    
    for display_name, svc_info in SERVICES.items():
        cat = svc_info.get("category", "infrastructure")
        categories[cat][1].append((display_name, svc_info))
    
    for cat_key, (cat_title, services) in categories.items():
        if not services:
            continue
            
        st.markdown(f"## {cat_title}")
        
        for display_name, svc_info in services:
            service_name = svc_info["name"]
            status = get_service_status(service_name)
            
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    st.markdown(f"### {svc_info['emoji']} {display_name}")
                    st.caption(svc_info["description"])
                    st.markdown(f"**Status:** {status['status_text']} | **Memory:** {status['memory']}")
                    st.caption(f"📌 {status['boot_text']}")
                
                with col2:
                    st.markdown("**Controls**")
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    
                    with btn_col1:
                        if st.button("▶️", key=f"main_start_{service_name}", help="Start"):
                            success, msg = control_service(service_name, "start")
                            if success:
                                st.toast(f"✅ {display_name} started!")
                            else:
                                st.error(msg)
                            st.rerun()
                    
                    with btn_col2:
                        if st.button("⏹️", key=f"main_stop_{service_name}", help="Stop"):
                            success, msg = control_service(service_name, "stop")
                            if success:
                                st.toast(f"⏹️ {display_name} stopped!")
                            else:
                                st.error(msg)
                            st.rerun()
                    
                    with btn_col3:
                        if st.button("🔄", key=f"main_restart_{service_name}", help="Restart"):
                            success, msg = control_service(service_name, "restart")
                            if success:
                                st.toast(f"🔄 {display_name} restarted!")
                            else:
                                st.error(msg)
                            st.rerun()
                
                with col3:
                    st.markdown("**Auto-Start**")
                    
                    # Show recommendation for Discord bot
                    if service_name == "sentient-discord-approval":
                        st.caption("💡 Recommended: Keep ON")
                    
                    if status["enabled"]:
                        if st.button("🔕 Disable", key=f"main_disable_{service_name}"):
                            control_service(service_name, "disable")
                            st.rerun()
                    else:
                        if st.button("🔔 Enable", key=f"main_enable_{service_name}"):
                            control_service(service_name, "enable")
                            st.rerun()
                
                # Expandable logs section
                with st.expander(f"📜 View Logs - {display_name}"):
                    log_col1, log_col2, log_col3 = st.columns([3, 1, 1])
                    with log_col2:
                        # Line count selector
                        log_lines = st.selectbox(
                            "Lines",
                            options=[50, 100, 200, 500],
                            index=1,  # Default to 100
                            key=f"log_lines_{service_name}",
                            label_visibility="collapsed"
                        )
                    with log_col3:
                        if st.button("🗑️ Clear", key=f"clear_logs_{service_name}", help="Clear log files to start fresh"):
                            success, msg = clear_service_logs(service_name)
                            if success:
                                st.toast(f"✅ {msg}")
                            else:
                                st.error(msg)
                            st.rerun()
                    
                    # Refresh button in its own row
                    if st.button("🔄 Refresh Logs", key=f"refresh_logs_{service_name}", use_container_width=True):
                        st.rerun()
                    
                    logs = get_service_logs(service_name, log_lines)
                    
                    # Use a container with fixed height for scrollable logs
                    st.code(logs, language="log", line_numbers=False)
                    
                    # Show log file info
                    st.caption(f"💡 Tip: Click 'Refresh Logs' to see latest entries. Showing last {log_lines} lines.")
                
                # Expandable settings section (only for services with configurable intervals)
                if svc_info.get("interval_key"):
                    with st.expander(f"⚙️ Settings - {display_name}"):
                        current_interval = get_service_interval(service_name, svc_info)
                        interval_min = svc_info.get("interval_min", 5)
                        interval_max = svc_info.get("interval_max", 3600)
                        # Ensure current_interval is within valid range
                        current_interval = max(interval_min, min(current_interval, interval_max))
                        interval_key = svc_info.get("interval_key", "scan_interval_seconds")
                        
                        st.markdown(f"**⏱️ Scan/Check Interval**")
                        
                        # Show current interval with nice formatting
                        if current_interval >= 60:
                            current_display = f"{current_interval}s ({current_interval//60}m {current_interval%60}s)"
                        else:
                            current_display = f"{current_interval}s"
                        st.info(f"Current: **{current_display}**")
                        
                        # Two input methods: slider for quick adjust, number input for precise
                        input_col1, input_col2 = st.columns([2, 1])
                        
                        with input_col1:
                            # Slider for quick adjustment
                            slider_val = st.slider(
                                "Quick Adjust (drag)",
                                min_value=interval_min,
                                max_value=min(600, interval_max),  # Cap slider at 10 min for usability
                                value=min(current_interval, 600),
                                step=5,
                                key=f"main_slider_{service_name}",
                                help="Drag for quick adjustment (5-600s)"
                            )
                        
                        with input_col2:
                            # Number input for precise custom values
                            custom_val = st.number_input(
                                "Custom (seconds)",
                                min_value=interval_min,
                                max_value=interval_max,
                                value=current_interval,
                                step=1,
                                key=f"main_custom_{service_name}",
                                help=f"Enter any value from {interval_min}s to {interval_max}s"
                            )
                        
                        # Determine which value to use (custom takes precedence if changed)
                        if f"main_last_custom_{service_name}" not in st.session_state:
                            st.session_state[f"main_last_custom_{service_name}"] = current_interval
                        
                        # Use custom if it was changed, otherwise use slider
                        if custom_val != st.session_state[f"main_last_custom_{service_name}"]:
                            new_interval = custom_val
                            st.session_state[f"main_last_custom_{service_name}"] = custom_val
                        else:
                            new_interval = slider_val
                        
                        # Show what will be applied
                        if new_interval != current_interval:
                            if new_interval >= 60:
                                new_display = f"{new_interval}s ({new_interval//60}m {new_interval%60}s)"
                            else:
                                new_display = f"{new_interval}s"
                            st.warning(f"📝 New value: **{new_display}** (click Apply to save)")
                        
                        # Apply button
                        col_apply, col_reset = st.columns([1, 1])
                        with col_apply:
                            if st.button("💾 Apply & Restart", key=f"main_apply_interval_{service_name}", type="primary"):
                                if set_service_interval(service_name, svc_info, int(new_interval)):
                                    # Restart the service to apply new interval
                                    success, msg = control_service(service_name, "restart")
                                    if success:
                                        st.toast(f"✅ Interval set to {new_interval}s and service restarted!")
                                    else:
                                        st.warning(f"Interval saved but restart failed: {msg}")
                                    st.rerun()
                                else:
                                    st.error("Failed to save interval")
                        
                        with col_reset:
                            default_val = svc_info.get("interval_default", 60)
                            if st.button(f"↩️ Reset to Default ({default_val}s)", key=f"main_reset_{service_name}"):
                                set_service_interval(service_name, svc_info, default_val)
                                control_service(service_name, "restart")
                                st.rerun()
                        
                        # Quick presets
                        st.markdown("---")
                        st.markdown("**⚡ Quick Presets:**")
                        preset_cols = st.columns(5)
                        
                        presets = [
                            ("10s", 10, "🏎️"),
                            ("30s", 30, "🚀"),
                            ("1m", 60, "⚖️"),
                            ("5m", 300, "🐢"),
                            ("15m", 900, "😴"),
                        ]
                        
                        for i, (label, seconds, emoji) in enumerate(presets):
                            with preset_cols[i]:
                                if seconds >= interval_min and seconds <= interval_max:
                                    if st.button(f"{emoji} {label}", key=f"main_preset_{seconds}_{service_name}"):
                                        set_service_interval(service_name, svc_info, seconds)
                                        control_service(service_name, "restart")
                                        st.rerun()
                        
                        # Info about intervals
                        st.caption(f"💡 Range: {interval_min}s - {interval_max}s ({interval_max//60} min). Lower = more responsive but higher API usage.")
                        
                        # ============================================================
                        # WATCHLIST / TICKER FILTERING
                        # ============================================================
                        st.markdown("---")
                        st.markdown("**📋 Watchlist / Ticker Filter**")
                        
                        category = svc_info.get("category", "stocks")
                        current_watchlist = get_service_watchlist(service_name, category)
                        
                        # Default ticker lists (fallback)
                        default_stock_tickers = [
                            'SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT',
                            'META', 'AMZN', 'GOOGL', 'PLTR', 'SOFI', 'COIN', 'MARA',
                            'RIOT', 'HOOD', 'NFLX', 'CRM', 'SHOP'
                        ]
                        default_crypto_tickers = [
                            'BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'LINK/USD', 
                            'DOGE/USD', 'SHIB/USD', 'PEPE/USD', 'XRP/USD', 'ADA/USD',
                            'DOT/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD', 'ARB/USD'
                        ]
                        default_dex_chains = ['solana', 'ethereum', 'base', 'arbitrum', 'polygon']
                        
                        # Try to fetch tickers from Supabase based on category
                        supabase_tickers = []
                        try:
                            from services.ticker_manager import TickerManager
                            svc_tm = TickerManager()
                            if svc_tm.test_connection():
                                if category == "crypto" and service_name != "sentient-dex-launch":
                                    # Fetch crypto tickers from Supabase
                                    crypto_data = svc_tm.get_all_tickers(ticker_type='crypto', limit=1000)
                                    if crypto_data:
                                        supabase_tickers = [t['ticker'] for t in crypto_data]
                                elif category == "stocks":
                                    # Fetch stock tickers from Supabase (includes penny_stock type)
                                    stock_data = svc_tm.get_all_tickers(ticker_type='stock', limit=1000)
                                    penny_data = svc_tm.get_all_tickers(ticker_type='penny_stock', limit=1000)
                                    if stock_data:
                                        supabase_tickers.extend([t['ticker'] for t in stock_data])
                                    if penny_data:
                                        supabase_tickers.extend([t['ticker'] for t in penny_data])
                        except Exception as e:
                            print(f"Failed to fetch tickers from Supabase: {e}")
                        
                        # Determine final ticker list: Supabase tickers + defaults
                        if category == "crypto" and service_name != "sentient-dex-launch":
                            all_tickers = list(set(supabase_tickers + default_crypto_tickers))
                        elif service_name == "sentient-dex-launch":
                            all_tickers = default_dex_chains
                        else:
                            all_tickers = list(set(supabase_tickers + default_stock_tickers))
                        
                        # Sort for consistent display
                        all_tickers.sort()
                        
                        # Show data source info
                        if supabase_tickers:
                            st.caption(f"☁️ Loaded {len(supabase_tickers)} tickers from Supabase + defaults")
                        else:
                            st.caption("📂 Using default ticker list (Supabase not connected or empty)")
                        
                        # Quick action buttons (mobile-friendly)
                        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                        with btn_col1:
                            if st.button("✅ All", key=f"main_watchlist_all_{service_name}", use_container_width=True):
                                set_service_watchlist(service_name, all_tickers)
                                st.rerun()
                        with btn_col2:
                            if st.button("❌ Clear", key=f"main_watchlist_clear_{service_name}", use_container_width=True):
                                set_service_watchlist(service_name, [])
                                st.rerun()
                        with btn_col3:
                            if st.button("🔝 Top 5", key=f"main_watchlist_top_{service_name}", use_container_width=True):
                                set_service_watchlist(service_name, all_tickers[:5])
                                st.rerun()
                        with btn_col4:
                            # Sync from Supabase button - use all Supabase tickers
                            if supabase_tickers:
                                if st.button("☁️ Sync", key=f"main_watchlist_sync_{service_name}", use_container_width=True, help="Use all tickers from Supabase"):
                                    set_service_watchlist(service_name, supabase_tickers)
                                    st.toast(f"✅ Synced {len(supabase_tickers)} tickers from Supabase!")
                                    st.rerun()
                        
                        # Multiselect for tickers
                        # Combine all_tickers with current_watchlist to ensure custom additions are available as options
                        available_options = list(set(all_tickers + current_watchlist))
                        
                        selected_tickers = st.multiselect(
                            "Select tickers to monitor",
                            options=available_options,
                            default=[t for t in current_watchlist if t in available_options],
                            key=f"main_watchlist_select_{service_name}",
                            help="Select which tickers this service should scan"
                        )
                        
                        # Custom ticker input
                        custom_input = st.text_input(
                            "Add custom tickers (comma-separated)",
                            placeholder="SMCI, RDDT, MSTR",
                            key=f"main_custom_tickers_{service_name}"
                        )
                        
                        # Save watchlist button
                        if st.button("💾 Save Watchlist", key=f"main_save_watchlist_{service_name}", type="primary", use_container_width=True):
                            # Combine selected and custom tickers
                            final_tickers = list(selected_tickers)
                            if custom_input:
                                custom_list = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
                                final_tickers.extend([t for t in custom_list if t not in final_tickers])
                            
                            # 1. Save to Local JSON
                            if set_service_watchlist(service_name, final_tickers):
                                
                                # 2. Sync to Supabase (if available)
                                try:
                                    from services.ticker_manager import TickerManager
                                    tm = TickerManager()
                                    if tm.test_connection():
                                        sup_watchlist_name = f"service_{service_name}"
                                        # Recreate watchlist to match exact state
                                        tm.delete_watchlist(sup_watchlist_name)
                                        tm.create_watchlist(sup_watchlist_name, description=f"Watchlist for {svc_label}")
                                        for t in final_tickers:
                                            tm.add_to_watchlist(sup_watchlist_name, t)
                                        st.toast("✅ Synced to Supabase!")
                                except Exception as e:
                                    print(f"Failed to sync to Supabase: {e}")
                                
                                st.toast(f"✅ Saved {len(final_tickers)} tickers!")
                                # Restart service to apply
                                control_service(service_name, "restart")
                                st.rerun()
                            else:
                                st.error("Failed to save watchlist")
                        
                        st.caption(f"📊 Currently monitoring: {len(current_watchlist)} ticker(s)")
                        
                        # ============================================================
                        # DISCORD ALERT SETTINGS
                        # ============================================================
                        st.markdown("---")
                        st.markdown("**🔔 Discord Alert Settings**")
                        
                        discord_settings = get_service_discord_settings(service_name)
                        
                        # Enable/Disable toggle
                        alerts_enabled = st.checkbox(
                            "Enable Discord Alerts",
                            value=discord_settings.get('enabled', True),
                            key=f"main_discord_enabled_{service_name}",
                            help="Turn Discord notifications on/off for this service"
                        )
                        
                        if alerts_enabled:
                            # Minimum confidence slider
                            min_confidence = st.slider(
                                "Min Confidence for Alerts",
                                min_value=50,
                                max_value=95,
                                value=discord_settings.get('min_confidence', 70),
                                step=5,
                                key=f"main_discord_confidence_{service_name}",
                                help="Only send alerts for signals above this confidence level"
                            )
                            
                            # Alert cooldown
                            cooldown = st.number_input(
                                "Cooldown (minutes)",
                                min_value=1,
                                max_value=120,
                                value=discord_settings.get('cooldown_minutes', 15),
                                key=f"main_discord_cooldown_{service_name}",
                                help="Minimum time between alerts for the same ticker"
                            )
                            
                            # Alert types
                            all_alert_types = ['signal', 'breakout', 'error', 'summary']
                            current_types = discord_settings.get('alert_types', ['signal', 'breakout', 'error'])
                            alert_types = st.multiselect(
                                "Alert Types",
                                options=all_alert_types,
                                default=[t for t in current_types if t in all_alert_types],
                                key=f"main_discord_types_{service_name}",
                                help="Which types of alerts to send"
                            )
                        else:
                            min_confidence = 70
                            cooldown = 15
                            alert_types = ['signal', 'breakout', 'error']
                        
                        # Save Discord settings button
                        if st.button("💾 Save Discord Settings", key=f"main_save_discord_{service_name}", use_container_width=True):
                            new_discord_settings = {
                                'enabled': alerts_enabled,
                                'min_confidence': min_confidence,
                                'cooldown_minutes': int(cooldown),
                                'alert_types': alert_types,
                            }
                            if set_service_discord_settings(service_name, new_discord_settings):
                                st.toast(f"✅ Discord settings saved!")
                                control_service(service_name, "restart")
                                st.rerun()
                            else:
                                st.error("Failed to save Discord settings")
                
                st.markdown("---")
    
    # ============================================================
    # ANALYSIS REQUESTS QUEUE
    # ============================================================
    st.markdown("## 📊 Analysis Queue")
    
    requests = get_analysis_requests()
    
    if requests:
        # Show recent requests
        for req in reversed(requests[-5:]):  # Show last 5
            status_emoji = "⏳" if req.get("status") == "pending" else "✅"
            preset = ANALYSIS_PRESETS.get(req.get("preset", ""), {})
            preset_name = preset.get("name", req.get("preset", "Custom"))
            tickers = req.get("tickers") or []
            ticker_display = ", ".join(tickers[:3]) + (f" +{len(tickers)-3}" if len(tickers) > 3 else "")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"{status_emoji} **{preset_name}**")
            with col2:
                st.caption(f"📈 {ticker_display}")
            with col3:
                created = req.get("created", "")[:16].replace("T", " ")
                st.caption(created)
        
        col_clear1, col_clear2 = st.columns(2)
        with col_clear1:
            if st.button("🗑️ Clear All Requests", use_container_width=True):
                if clear_analysis_requests():
                    st.toast("✅ Analysis queue cleared!")
                else:
                    st.error("❌ Failed to clear queue")
                time.sleep(0.3)
                st.rerun()
        with col_clear2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                if clear_analysis_results():
                    st.toast("✅ Analysis results cleared!")
                else:
                    st.error("❌ Failed to clear results")
                time.sleep(0.3)
                st.rerun()
    else:
        st.info("No analysis requests queued. Use the sidebar to trigger scans from your phone! 📱")
    
    st.markdown("---")
    
    # ============================================================
    # ANALYSIS RESULTS VIEWER (with Auto-Refresh)
    # ============================================================
    # Create columns for header and refresh control
    res_col1, res_col2, res_col3 = st.columns([2, 1, 1])
    with res_col1:
        st.markdown("## 📈 Recent Analysis Results")
    with res_col2:
        if st.button("🔄 Refresh", key="refresh_results"):
            st.rerun()
    with res_col3:
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="auto_refresh_results")
        if auto_refresh:
            st.session_state.last_refresh_time = time.time()
    
    # Auto-refresh logic
    if auto_refresh and 'last_refresh_time' in st.session_state:
        elapsed = time.time() - st.session_state.last_refresh_time
        if elapsed > 15:  # Refresh every 15 seconds
            st.session_state.last_refresh_time = time.time()
            st.rerun()
    
    analysis_results = get_analysis_results()
    
    # Build service label map for better organization
    SERVICE_LABELS = {
        'latest': '⏱️ Latest Results',
        'queue_latest': '⏱️ Latest Queue',
        'Crypto Standard': '🔍 Crypto Standard',
        'Crypto Multi-Config': '🎯 Crypto Multi-Config',
        'Crypto Ultimate': '🚀 Crypto Ultimate',
        'Stock Momentum': '📈 Stock Momentum',
        'queue_': '📊 Queue Analysis'
    }
    
    def get_service_label(service_key: str) -> str:
        """Get a better label for the service"""
        # Check for exact matches first
        if service_key in SERVICE_LABELS:
            return SERVICE_LABELS[service_key]
        
        # Check for partial matches
        for pattern, label in SERVICE_LABELS.items():
            if pattern in service_key.lower():
                return label
        
        # Clean up the raw key
        clean_label = service_key.replace('_', ' ').title()
        return f"📊 {clean_label}"
    
    if analysis_results:
        # Initialize session state for stable tab keys - prioritize special keys
        if 'analysis_tab_keys' not in st.session_state:
            all_keys = list(analysis_results.keys())
            # Sort with 'latest' first, then 'queue_latest', then others
            special_keys = [k for k in all_keys if k in ['latest', 'queue_latest']]
            other_keys = [k for k in all_keys if k not in ['latest', 'queue_latest']]
            st.session_state.analysis_tab_keys = special_keys + sorted(other_keys)
        
        # Update keys if they've changed, but maintain order
        current_keys = list(analysis_results.keys())
        if set(current_keys) != set(st.session_state.analysis_tab_keys):
            # Merge: keep existing order, add new keys at end
            existing_keys = [k for k in st.session_state.analysis_tab_keys if k in current_keys]
            new_keys = [k for k in current_keys if k not in st.session_state.analysis_tab_keys]
            st.session_state.analysis_tab_keys = existing_keys + new_keys
        
        # Create tabs with improved labels
        tab_labels = [get_service_label(key) for key in st.session_state.analysis_tab_keys] + ["📋 All Results"]
        result_tabs = st.tabs(tab_labels)
        
        # Map services to tabs (show message if service has no current data)
        for i, service in enumerate(st.session_state.analysis_tab_keys):
            with result_tabs[i]:
                if service not in analysis_results:
                    st.info(f"No current results for {get_service_label(service)}. Results will appear here when available.")
                    continue
                
                data = analysis_results[service]
                if not isinstance(data, dict):
                    st.error(f"Invalid data format for {service}")
                    continue
                    
                results = data.get('results', [])
                updated = str(data.get('updated', 'Unknown'))[:19].replace('T', ' ')
                count = data.get('count', len(results))
                
                if not isinstance(results, list):
                    st.warning(f"Invalid results list for {service}")
                    results = []
                
                # Enhanced header with status indicators
                st.markdown(f"### {get_service_label(service)}")
                
                # Status badges
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.caption(f"📅 Updated: {updated}")
                with col2:
                    st.metric("Total", len(results), f"{count if count else len(results)} stored")
                with col3:
                    if results:
                        st.metric("Latest", results[-1].get('ticker', 'N/A'))
                
                st.divider()
                
                if results:
                    # Display last 15 results in a better format
                    for idx, result in enumerate(results[-15:], 1):
                        ticker = result.get('ticker', result.get('symbol', 'N/A'))
                        signal = result.get('signal', result.get('action', 'N/A'))
                        confidence = result.get('confidence', result.get('score', 0))
                        price = result.get('price', result.get('entry_price', 0))
                        timestamp = result.get('analysis_time', '')[:19] if result.get('analysis_time') else ''
                        reasoning = result.get('reasoning', '')
                        
                        # Get more detailed fields for stock analysis
                        entry_point = result.get('suggested_entry')
                        stop_loss = result.get('suggested_stop')
                        target = result.get('suggested_target')
                        risk_reward = result.get('risk_reward_ratio')
                        urgency = result.get('urgency', '')
                        
                        # Format confidence for display (avoid f-string format issues)
                        if isinstance(confidence, (int, float)):
                            confidence_str = f"{confidence:.0f}"
                        else:
                            confidence_str = str(confidence)
                        
                        # Color based on signal
                        if signal in ['LONG', 'BUY', 'BULLISH', 'ENTER_NOW']:
                            signal_color = "🟢"
                        elif signal in ['SHORT', 'SELL', 'BEARISH', 'DO_NOT_ENTER']:
                            signal_color = "🔴"
                        else:
                            signal_color = "🟡"
                        
                        # Urgency indicator
                        urgency_emoji = "🔥" if urgency == "HIGH" else "⏱️" if urgency == "MEDIUM" else "✅"
                        
                        # Create an expandable result card
                        with st.expander(f"{signal_color} **{ticker}** → {signal} ({confidence_str}%)", expanded=False):
                            result_col1, result_col2 = st.columns([1, 2])
                            
                            with result_col1:
                                st.markdown(f"**Signal:** {signal_color} {signal}")
                                st.markdown(f"**Confidence:** {confidence_str}%")
                                if urgency:
                                    st.markdown(f"**Urgency:** {urgency_emoji} {urgency}")
                                if isinstance(price, (int, float)) and price > 0:
                                    st.markdown(f"**Price:** ${price:.2f}")
                                st.caption(f"📅 {timestamp}")
                            
                            with result_col2:
                                if reasoning:
                                    st.markdown(f"**Analysis:** {reasoning}")
                                if entry_point is not None and signal not in ['DO_NOT_ENTER']:
                                    entry_str = f"{entry_point:.2f}" if isinstance(entry_point, (int, float)) else str(entry_point)
                                    st.markdown(f"**Entry:** ${entry_str}")
                                if stop_loss is not None:
                                    stop_str = f"{stop_loss:.2f}" if isinstance(stop_loss, (int, float)) else str(stop_loss)
                                    st.markdown(f"**Stop Loss:** ${stop_str}")
                                if target is not None:
                                    target_str = f"{target:.2f}" if isinstance(target, (int, float)) else str(target)
                                    st.markdown(f"**Target:** ${target_str}")
                                if risk_reward is not None and signal not in ['DO_NOT_ENTER']:
                                    rr_str = f"{risk_reward:.2f}" if isinstance(risk_reward, (int, float)) else str(risk_reward)
                                    st.markdown(f"**Risk/Reward:** {rr_str}")
                            
                            # Divider between results
                            st.divider()
                else:
                    st.info("📭 No results yet. Analysis requests will appear here once processed.")
        
        # "All Results" tab - combined view
        with result_tabs[-1]:
            st.markdown("### 📋 All Results (Latest)")
            
            all_results = []
            for service, data in analysis_results.items():
                if service in ['latest', 'queue_latest']:  # Skip meta keys
                    continue
                if not isinstance(data, dict):
                    continue
                
                results = data.get('results', [])
                if not isinstance(results, list):
                    continue
                    
                for result in results:
                    if isinstance(result, dict):
                        result['_service'] = get_service_label(service)
                        all_results.append(result)
            
            if all_results:
                # Sort by timestamp (most recent first)
                try:
                    all_results.sort(key=lambda x: x.get('analysis_time', ''), reverse=True)
                except:
                    pass
                
                st.caption(f"📊 Total: **{len(all_results)}** results across all services")
                st.divider()
                
                for result in all_results[:30]:  # Show top 30 most recent
                    ticker = result.get('ticker', result.get('symbol', 'N/A'))
                    signal = result.get('signal', result.get('action', 'N/A'))
                    confidence = result.get('confidence', result.get('score', 0))
                    service = result.get('_service', 'Unknown')
                    timestamp = result.get('analysis_time', '')[:19] if result.get('analysis_time') else ''
                    
                    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 0.8, 2, 1.5])
                    with col1:
                        st.write(f"**{ticker}**")
                    with col2:
                        st.write(signal)
                    with col3:
                        if isinstance(confidence, (int, float)):
                            st.write(f"{confidence:.0f}%")
                        else:
                            st.write(str(confidence))
                    with col4:
                        st.caption(service)
                    with col5:
                        st.caption(timestamp)
            else:
                st.info("📭 No results from any service yet")
        
        # Control buttons
        st.divider()
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            if st.button("🔄 Force Refresh Now", key="force_refresh_results"):
                st.rerun()
        
        with button_col2:
            if st.button("🗑️ Clear All Results", key="clear_all_results"):
                clear_analysis_results()
                st.success("✅ All analysis results cleared!")
                time.sleep(1)
                st.rerun()
        
        with button_col3:
            st.caption("⚠️ Use sparingly")
    else:
        st.info("📭 No analysis results yet.\n\n**To get started:**\n1. Queue an analysis from the sidebar\n2. Or wait for background services to generate results\n3. Results will appear here automatically")
    
    # ============================================================
    # RISK PROFILE TAB
    # ============================================================
    with tab_risk:
        try:
            from ui.risk_profile_ui import render_risk_profile_config
            render_risk_profile_config()
        except ImportError as e:
            st.error(f"Could not load Risk Profile UI: {e}")
            st.info("Make sure ui/risk_profile_ui.py exists")
        except Exception as e:
            st.error(f"Error rendering Risk Profile: {e}")
    
    st.markdown("---")
    
    # System Info
    with st.expander("🖥️ System Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            if platform.system() == "Windows":
                # Windows uptime (boot time)
                try:
                    import psutil
                    boot_time = datetime.fromtimestamp(psutil.boot_time())
                    uptime = datetime.now() - boot_time
                    st.metric("Server Uptime", str(uptime).split('.')[0])
                except ImportError:
                    st.metric("Server Uptime", "N/A (install psutil)")
            else:
                success, uptime = run_command("uptime -p")
                st.metric("Server Uptime", uptime if success else "N/A")
        
        with col2:
            if platform.system() == "Windows":
                 try:
                    import psutil
                    mem = psutil.virtual_memory()
                    st.metric("Memory Usage", f"{mem.percent}%")
                 except ImportError:
                    st.metric("Memory Usage", "N/A (install psutil)")
            else:
                success, memory = run_command("free -h | grep Mem | awk '{print $3\"/\"$2}'")
                st.metric("Memory Usage", memory if success else "N/A")


if __name__ == "__main__":
    main()

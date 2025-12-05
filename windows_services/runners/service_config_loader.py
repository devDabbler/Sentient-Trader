"""
Service Configuration Loader

Shared utility for service runners to load configuration from the Control Panel.
Reads from data/service_watchlists.json and data/analysis_requests.json.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Project root (two levels up from runners/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def load_service_watchlist(service_name: str, default_tickers: Optional[List[str]] = None) -> List[str]:
    """
    Load watchlist for a specific service from Control Panel config.
    
    Args:
        service_name: e.g., 'sentient-stock-monitor', 'sentient-crypto-breakout'
        default_tickers: Fallback tickers if no config found
        
    Returns:
        List of ticker symbols to scan
    """
    watchlist_file = PROJECT_ROOT / 'data' / 'service_watchlists.json'
    
    try:
        if watchlist_file.exists():
            content = watchlist_file.read_text().strip()
            if content:
                try:
                    watchlists = json.loads(content)
                    if isinstance(watchlists, dict):
                        service_config = watchlists.get(service_name, {})
                        tickers = service_config.get('tickers', [])
                        if tickers:
                            return tickers
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[service_config_loader] Error loading watchlist: {e}")
    
    return default_tickers or []


def load_discord_settings(service_name: str) -> Dict:
    """
    Load Discord alert settings for a specific service.
    
    Args:
        service_name: e.g., 'sentient-stock-monitor'
        
    Returns:
        Dict with discord settings (webhook_url, enabled, min_confidence, etc.)
    """
    settings_file = PROJECT_ROOT / 'data' / 'service_discord_settings.json'
    
    default_settings = {
        'enabled': True,
        'min_confidence': 70,
        'alert_types': ['signal', 'breakout', 'error'],
        'webhook_url': None,  # Uses default from .env if None
        'cooldown_minutes': 15,
    }
    
    try:
        if settings_file.exists():
            content = settings_file.read_text().strip()
            if content:
                try:
                    all_settings = json.loads(content)
                    if isinstance(all_settings, dict):
                        service_settings = all_settings.get(service_name, {})
                        # Merge with defaults
                        return {**default_settings, **service_settings}
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[service_config_loader] Error loading discord settings: {e}")
    
    return default_settings


def get_pending_analysis_requests() -> List[Dict]:
    """
    Get pending analysis requests from Control Panel queue.
    
    Returns:
        List of pending request dicts
    """
    requests_file = PROJECT_ROOT / 'data' / 'analysis_requests.json'
    
    try:
        if requests_file.exists():
            content = requests_file.read_text().strip()
            if content:  # Only parse if file has content
                requests = json.loads(content)
                if isinstance(requests, list):
                    # Return only pending requests
                    return [r for r in requests if r.get('status') == 'pending']
    except json.JSONDecodeError:
        print(f"[service_config_loader] Corrupted analysis_requests.json")
    except Exception as e:
        print(f"[service_config_loader] Error loading analysis requests: {e}")
    
    return []


def mark_analysis_complete(request_id: str, results: Optional[Dict] = None) -> bool:
    """
    Mark an analysis request as complete and store results.
    
    Args:
        request_id: ID of the request to mark complete
        results: Optional dict of analysis results
        
    Returns:
        True if successful
    """
    import os
    requests_file = PROJECT_ROOT / 'data' / 'analysis_requests.json'
    
    try:
        requests = []
        if requests_file.exists():
            content = requests_file.read_text().strip()
            if content:
                try:
                    requests = json.loads(content)
                    if not isinstance(requests, list):
                        requests = []
                except json.JSONDecodeError:
                    requests = []
        
        # Find and update the request
        found = False
        for req in requests:
            if req.get('id') == request_id:
                req['status'] = 'complete'
                req['completed'] = datetime.now().isoformat()
                if results:
                    req['results'] = results
                found = True
                break
        
        if not found:
            print(f"[service_config_loader] Request {request_id} not found")
            return False
        
        # Write with explicit flush for Windows file system
        with open(requests_file, 'w') as f:
            json.dump(requests, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Force OS to write to disk
        
        return True
        
    except Exception as e:
        print(f"[service_config_loader] Error marking analysis complete: {e}")
        return False


def save_analysis_results(service_name: str, results: List[Dict]) -> bool:
    """
    Save analysis results for display in Control Panel.
    
    Args:
        service_name: Service that generated the results
        results: List of result dicts
        
    Returns:
        True if successful
    """
    results_file = PROJECT_ROOT / 'data' / 'analysis_results.json'
    
    try:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        if results_file.exists():
            content = results_file.read_text().strip()
            if content:
                try:
                    all_results = json.loads(content)
                    if not isinstance(all_results, dict):
                        all_results = {}
                except json.JSONDecodeError:
                    all_results = {}
        
        # Store results for this service
        all_results[service_name] = {
            'results': results[-50:],  # Keep last 50
            'updated': datetime.now().isoformat(),
            'count': len(results)
        }
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        return True
        
    except Exception as e:
        print(f"[service_config_loader] Error saving analysis results: {e}")
        return False


def get_analysis_results(service_name: Optional[str] = None) -> Dict:
    """
    Get analysis results from all services or a specific one.
    
    Args:
        service_name: Optional service name filter
        
    Returns:
        Dict of results by service
    """
    results_file = PROJECT_ROOT / 'data' / 'analysis_results.json'
    
    try:
        if results_file.exists():
            content = results_file.read_text().strip()
            if content:
                try:
                    all_results = json.loads(content)
                    if not isinstance(all_results, dict):
                        return {}
                    if service_name:
                        return all_results.get(service_name, {})
                    return all_results
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[service_config_loader] Error loading analysis results: {e}")
    
    return {}


# Analysis presets (shared with service_control_panel.py)
ANALYSIS_PRESETS = {
    # Crypto Analysis Modes
    "crypto_standard": {
        "name": "🔬 Crypto Standard",
        "tickers": None,
        "depth": "standard",
        "asset_type": "crypto",
        "analysis_mode": "standard",
    },
    "crypto_multi": {
        "name": "🎯 Crypto Multi-Config",
        "tickers": None,
        "depth": "multi",
        "asset_type": "crypto",
        "analysis_mode": "multi_config",
    },
    "crypto_ultimate": {
        "name": "🚀 Crypto Ultimate",
        "tickers": None,
        "depth": "ultimate",
        "asset_type": "crypto",
        "analysis_mode": "ultimate",
    },
    "quick_crypto": {
        "name": "⚡ Quick Crypto",
        "tickers": ['BTC/USD', 'ETH/USD', 'SOL/USD'],
        "depth": "quick",
        "asset_type": "crypto",
        "analysis_mode": "standard",
    },
    # Stock Analysis Modes
    "stock_standard": {
        "name": "🔬 Stock Standard",
        "tickers": None,
        "depth": "standard",
        "asset_type": "stock",
        "analysis_mode": "standard",
        "description": "Single strategy + timeframe analysis",
    },
    "stock_multi": {
        "name": "🎯 Stock Multi-Config",
        "tickers": None,
        "depth": "multi",
        "asset_type": "stock",
        "analysis_mode": "multi_config",
        "description": "Test Long/Short + multiple timeframes",
    },
    "stock_ultimate": {
        "name": "🚀 Stock Ultimate",
        "tickers": None,
        "depth": "ultimate",
        "asset_type": "stock",
        "analysis_mode": "ultimate",
        "description": "ALL strategies + directions + timeframes",
    },
    "stock_momentum": {
        "name": "📈 Stock Momentum",
        "tickers": ['NVDA', 'TSLA', 'AMD', 'PLTR', 'COIN', 'MARA'],
        "depth": "medium",
        "asset_type": "stock",
        "analysis_mode": "standard",
    },
    "orb_fvg_scan": {
        "name": "🎯 ORB+FVG Day Trade",
        "tickers": ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD'],
        "depth": "deep",
        "asset_type": "stock",
        "analysis_mode": "standard",
    },
}


def queue_analysis_request(
    preset_key: str, 
    custom_tickers: Optional[List[str]] = None,
    asset_type: Optional[str] = None,
    analysis_mode: Optional[str] = None,
    source: Optional[str] = None
) -> bool:
    """
    Queue an analysis request for services to pick up.
    
    Args:
        preset_key: Key from ANALYSIS_PRESETS or 'custom'
        custom_tickers: Optional list of tickers (overrides preset)
        asset_type: Optional asset type override ('crypto' or 'stock')
        analysis_mode: Optional analysis mode override ('standard', 'multi', 'ultimate')
        source: Optional source identifier for channel routing (e.g., 'dex_hunter', 'dex_monitor')
        
    Returns:
        True if successful
    """
    import os
    requests_file = PROJECT_ROOT / 'data' / 'analysis_requests.json'
    
    try:
        requests_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing requests (handle empty/corrupted file gracefully)
        requests = []
        if requests_file.exists():
            try:
                content = requests_file.read_text().strip()
                if content:  # Only parse if file has content
                    requests = json.loads(content)
                    if not isinstance(requests, list):
                        requests = []
            except json.JSONDecodeError:
                # File is corrupted, start fresh
                print(f"[service_config_loader] Corrupted analysis_requests.json, resetting")
                requests = []
        
        # Add new request
        preset = ANALYSIS_PRESETS.get(preset_key, {})
        
        # Use overrides if provided, otherwise use preset defaults
        final_asset_type = asset_type or preset.get("asset_type", "crypto")
        final_analysis_mode = analysis_mode or preset.get("analysis_mode", "standard")
        final_tickers = custom_tickers or preset.get("tickers", [])
        
        # DUPLICATE DETECTION: Check if an identical pending request already exists
        # This prevents double-clicks from creating duplicate analysis runs
        for existing in requests:
            if (existing.get("status") == "pending" and
                existing.get("preset") == preset_key and
                existing.get("tickers") == final_tickers and
                existing.get("asset_type") == final_asset_type and
                existing.get("analysis_mode") == final_analysis_mode):
                print(f"[service_config_loader] Duplicate pending request detected, skipping")
                return True  # Return True so UI doesn't show error
        
        # Use microsecond precision for unique IDs (prevents same-second collisions)
        request = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{datetime.now().microsecond:06d}",
            "preset": preset_key,
            "tickers": final_tickers,
            "depth": preset.get("depth", "medium"),
            "asset_type": final_asset_type,
            "analysis_mode": final_analysis_mode,
            "source": source or "",
            "status": "pending",
            "created": datetime.now().isoformat(),
        }
        requests.append(request)
        
        # Keep only last 20 requests
        requests = requests[-20:]
        
        # Write with explicit flush for Windows file system
        with open(requests_file, 'w') as f:
            json.dump(requests, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        return True
        
    except Exception as e:
        print(f"[service_config_loader] Error queuing analysis: {e}")
        return False

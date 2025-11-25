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
            with open(watchlist_file, 'r') as f:
                watchlists = json.load(f)
            
            service_config = watchlists.get(service_name, {})
            tickers = service_config.get('tickers', [])
            
            if tickers:
                return tickers
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
            with open(settings_file, 'r') as f:
                all_settings = json.load(f)
            
            service_settings = all_settings.get(service_name, {})
            # Merge with defaults
            return {**default_settings, **service_settings}
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
            with open(requests_file, 'r') as f:
                requests = json.load(f)
            
            # Return only pending requests
            return [r for r in requests if r.get('status') == 'pending']
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
    requests_file = PROJECT_ROOT / 'data' / 'analysis_requests.json'
    
    try:
        requests = []
        if requests_file.exists():
            with open(requests_file, 'r') as f:
                requests = json.load(f)
        
        # Find and update the request
        for req in requests:
            if req.get('id') == request_id:
                req['status'] = 'complete'
                req['completed'] = datetime.now().isoformat()
                if results:
                    req['results'] = results
                break
        
        with open(requests_file, 'w') as f:
            json.dump(requests, f, indent=2)
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
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        
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
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            
            if service_name:
                return all_results.get(service_name, {})
            return all_results
    except Exception as e:
        print(f"[service_config_loader] Error loading analysis results: {e}")
    
    return {}

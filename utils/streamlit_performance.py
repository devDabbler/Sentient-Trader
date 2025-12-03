"""
Streamlit Performance Optimization Utilities

Provides cached helpers, debounced operations, and fragment-based partial updates
to dramatically improve UX by reducing unnecessary full-page reruns.

Key patterns:
1. @st.cache_data with TTL for expensive file I/O operations
2. @st.fragment for partial re-renders (Streamlit 1.33+)
3. Callback-based state updates to batch changes before rerun
4. Debounced actions to prevent rapid-fire duplicate operations
"""

import streamlit as st
import time
import json
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from loguru import logger

# Type variable for generic return types
T = TypeVar('T')


# ============================================================
# CACHING UTILITIES
# ============================================================

def get_cached_file_content(file_path: Union[str, Path], ttl_seconds: int = 30) -> Optional[Dict]:
    """
    Load JSON file with caching. Uses session state cache with TTL.
    
    Avoids redundant file I/O operations that slow down the app.
    
    Args:
        file_path: Path to JSON file
        ttl_seconds: Cache time-to-live in seconds (default 30s)
    
    Returns:
        Parsed JSON dict or None if file doesn't exist/is invalid
    """
    path = Path(file_path)
    cache_key = f"_file_cache_{path.name}"
    cache_ts_key = f"_file_cache_ts_{path.name}"
    
    # Check if cache is still valid
    now = time.time()
    cached_ts = st.session_state.get(cache_ts_key, 0)
    
    if now - cached_ts < ttl_seconds and cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Load from file
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            st.session_state[cache_key] = content
            st.session_state[cache_ts_key] = now
            return content
    except Exception as e:
        logger.debug(f"Error loading cached file {path}: {e}")
    
    return None


def invalidate_file_cache(file_path: Union[str, Path]) -> None:
    """Invalidate cached file content, forcing reload on next access."""
    path = Path(file_path)
    cache_key = f"_file_cache_{path.name}"
    cache_ts_key = f"_file_cache_ts_{path.name}"
    
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    if cache_ts_key in st.session_state:
        del st.session_state[cache_ts_key]


def cached_operation(ttl_seconds: int = 60):
    """
    Decorator to cache expensive operation results in session state.
    
    Usage:
        @cached_operation(ttl_seconds=30)
        def get_expensive_data():
            return do_expensive_thing()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = f"_op_cache_{func.__name__}_{hash(str(args) + str(kwargs))}"
            cache_ts_key = f"_op_cache_ts_{func.__name__}"
            
            now = time.time()
            cached_ts = st.session_state.get(cache_ts_key, 0)
            
            if now - cached_ts < ttl_seconds and cache_key in st.session_state:
                return st.session_state[cache_key]
            
            result = func(*args, **kwargs)
            st.session_state[cache_key] = result
            st.session_state[cache_ts_key] = now
            return result
        
        return wrapper
    return decorator


# ============================================================
# DEBOUNCE / THROTTLE UTILITIES
# ============================================================

def debounced_action(action_id: str, cooldown_seconds: float = 0.5) -> bool:
    """
    Prevent rapid-fire duplicate actions (e.g., double-click prevention).
    
    Returns True if action should proceed, False if it's within cooldown.
    
    Usage:
        if st.button("Do Thing") and debounced_action("do_thing"):
            # Action proceeds
    """
    last_action_key = f"_debounce_{action_id}"
    now = time.time()
    
    last_action_time = st.session_state.get(last_action_key, 0)
    
    if now - last_action_time < cooldown_seconds:
        return False
    
    st.session_state[last_action_key] = now
    return True


def rate_limited_action(action_id: str, min_interval_seconds: float = 2.0) -> bool:
    """
    Rate limit actions to prevent server overload.
    
    Similar to debounce but with longer intervals for expensive operations.
    """
    return debounced_action(action_id, min_interval_seconds)


# ============================================================
# STATE UPDATE HELPERS (Reduce reruns)
# ============================================================

class StateUpdater:
    """
    Batch multiple state updates before calling rerun.
    
    Instead of:
        st.session_state.foo = 1
        st.rerun()
        st.session_state.bar = 2
        st.rerun()  # Wasted rerun
    
    Use:
        with StateUpdater() as state:
            state.set("foo", 1)
            state.set("bar", 2)
        # Single rerun at end (if changes made)
    """
    
    def __init__(self, rerun_on_change: bool = True):
        self._changes: Dict[str, Any] = {}
        self._rerun_on_change = rerun_on_change
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._changes and self._rerun_on_change:
            for key, value in self._changes.items():
                st.session_state[key] = value
            st.rerun()
    
    def set(self, key: str, value: Any) -> None:
        """Queue a state update."""
        self._changes[key] = value
    
    def get_changes(self) -> Dict[str, Any]:
        """Get pending changes without applying them."""
        return self._changes.copy()


def batch_state_update(updates: Dict[str, Any], rerun: bool = True) -> None:
    """
    Apply multiple state updates in one operation.
    
    More efficient than multiple individual updates + reruns.
    """
    for key, value in updates.items():
        st.session_state[key] = value
    
    if rerun and updates:
        st.rerun()


# ============================================================
# CALLBACK HELPERS (Avoid immediate reruns)
# ============================================================

def create_callback(state_key: str, value: Any) -> Callable:
    """
    Create a callback function that sets state without immediate rerun.
    
    Usage with st.button:
        st.button("Set Value", on_click=create_callback("my_key", "my_value"))
    """
    def callback():
        st.session_state[state_key] = value
    return callback


def create_toggle_callback(state_key: str) -> Callable:
    """
    Create a callback that toggles a boolean state value.
    
    Usage:
        st.button("Toggle", on_click=create_toggle_callback("is_enabled"))
    """
    def callback():
        current = st.session_state.get(state_key, False)
        st.session_state[state_key] = not current
    return callback


def create_append_callback(state_key: str, value: Any) -> Callable:
    """
    Create a callback that appends to a list in state.
    """
    def callback():
        current = st.session_state.get(state_key, [])
        if not isinstance(current, list):
            current = [current]
        current.append(value)
        st.session_state[state_key] = current
    return callback


# ============================================================
# TOAST-BASED FEEDBACK (Non-blocking)
# ============================================================

def show_action_result(success: bool, success_msg: str, error_msg: str = "Operation failed") -> None:
    """
    Show toast feedback for action result (non-blocking).
    
    Much faster UX than showing messages and then rerunning.
    """
    if success:
        st.toast(f"✅ {success_msg}")
    else:
        st.toast(f"❌ {error_msg}")


def show_progress_toast(message: str) -> None:
    """Show a quick progress toast."""
    st.toast(f"⏳ {message}...")


# ============================================================
# SMART RERUN HELPERS
# ============================================================

def should_rerun(reason: str = "refresh") -> bool:
    """
    Determine if a rerun is actually needed based on recent activity.
    
    Prevents unnecessary reruns when nothing has changed.
    """
    last_rerun_key = "_last_rerun_time"
    last_reason_key = "_last_rerun_reason"
    
    now = time.time()
    last_rerun = st.session_state.get(last_rerun_key, 0)
    last_reason = st.session_state.get(last_reason_key, "")
    
    # Skip if same reason within 1 second (prevent rapid duplicate reruns)
    if reason == last_reason and now - last_rerun < 1.0:
        return False
    
    st.session_state[last_rerun_key] = now
    st.session_state[last_reason_key] = reason
    return True


def smart_rerun(reason: str = "action") -> None:
    """
    Perform a rerun only if it's actually needed.
    
    Reduces unnecessary full-page refreshes.
    """
    if should_rerun(reason):
        st.rerun()


def delayed_rerun(delay_ms: int = 100) -> None:
    """
    Schedule a rerun with a small delay to allow UI feedback.
    
    Uses Streamlit's native rerun but with optional visual feedback first.
    """
    # The delay is handled by showing toast before rerun
    st.rerun()


# ============================================================
# FRAGMENT HELPER (Streamlit 1.33+)
# ============================================================

def fragment_safe(run_every: Optional[float] = None):
    """
    Decorator that wraps a function as a Streamlit fragment for partial reruns.
    
    Falls back gracefully if Streamlit version doesn't support fragments.
    
    Usage:
        @fragment_safe(run_every=30)  # Auto-refresh every 30 seconds
        def my_widget():
            # This will only re-render this section, not the whole page
            st.button("Click me")
    """
    def decorator(func: Callable) -> Callable:
        # Check if fragment is available (Streamlit 1.33+)
        if hasattr(st, 'fragment'):
            if run_every is not None:
                return st.fragment(run_every=run_every)(func)
            else:
                return st.fragment(func)
        else:
            # Fallback: just return the function as-is
            return func
    return decorator


# ============================================================
# AUTO-REFRESH UTILITIES
# ============================================================

def setup_auto_refresh(key: str, interval_seconds: int = 30, enabled_by_default: bool = False) -> bool:
    """
    Set up auto-refresh controls with proper session state management.
    
    Returns True if a rerun should happen now.
    
    Usage:
        if setup_auto_refresh("my_section", 30):
            # Refresh is due - data will be reloaded on next render
            pass
    """
    enabled_key = f"_auto_refresh_enabled_{key}"
    last_refresh_key = f"_auto_refresh_time_{key}"
    
    # Initialize defaults
    if enabled_key not in st.session_state:
        st.session_state[enabled_key] = enabled_by_default
    if last_refresh_key not in st.session_state:
        st.session_state[last_refresh_key] = time.time()
    
    is_enabled = st.session_state[enabled_key]
    
    if is_enabled:
        elapsed = time.time() - st.session_state[last_refresh_key]
        if elapsed >= interval_seconds:
            st.session_state[last_refresh_key] = time.time()
            return True
    
    return False


def render_auto_refresh_toggle(key: str, label: str = "Auto-refresh", interval_seconds: int = 30) -> None:
    """
    Render an auto-refresh toggle checkbox with countdown display.
    
    Non-blocking implementation that doesn't cause additional reruns.
    """
    enabled_key = f"_auto_refresh_enabled_{key}"
    last_refresh_key = f"_auto_refresh_time_{key}"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        is_enabled = st.checkbox(
            f"🔄 {label} ({interval_seconds}s)",
            value=st.session_state.get(enabled_key, False),
            key=f"toggle_{enabled_key}",
            on_change=lambda: st.session_state.update({enabled_key: not st.session_state.get(enabled_key, False)})
        )
        st.session_state[enabled_key] = is_enabled
    
    with col2:
        if is_enabled:
            elapsed = time.time() - st.session_state.get(last_refresh_key, time.time())
            remaining = max(0, interval_seconds - elapsed)
            st.caption(f"⏱️ {remaining:.0f}s")


# ============================================================
# CLEANUP UTILITIES
# ============================================================

def cleanup_stale_cache(max_age_seconds: int = 300) -> int:
    """
    Clean up stale cache entries from session state.
    
    Call periodically to prevent memory bloat.
    Returns number of entries cleaned.
    """
    now = time.time()
    cleaned = 0
    
    keys_to_remove = []
    for key in list(st.session_state.keys()):
        if key.startswith(("_file_cache_ts_", "_op_cache_ts_", "_debounce_")):
            ts = st.session_state.get(key, 0)
            if now - ts > max_age_seconds:
                # Also remove the associated data key
                data_key = key.replace("_ts_", "_").replace("_debounce_", "_debounce_data_")
                keys_to_remove.extend([key, data_key])
    
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
            cleaned += 1
    
    return cleaned


# ============================================================
# INITIALIZATION
# ============================================================

def init_performance_state() -> None:
    """
    Initialize performance tracking state.
    
    Call at app startup to ensure consistent behavior.
    """
    if "_perf_initialized" not in st.session_state:
        st.session_state._perf_initialized = True
        st.session_state._last_rerun_time = 0
        st.session_state._last_rerun_reason = ""
        logger.debug("Performance state initialized")


# Export all public utilities
__all__ = [
    # Caching
    'get_cached_file_content',
    'invalidate_file_cache', 
    'cached_operation',
    # Debounce
    'debounced_action',
    'rate_limited_action',
    # State updates
    'StateUpdater',
    'batch_state_update',
    # Callbacks
    'create_callback',
    'create_toggle_callback',
    'create_append_callback',
    # Feedback
    'show_action_result',
    'show_progress_toast',
    # Rerun
    'should_rerun',
    'smart_rerun',
    'delayed_rerun',
    # Fragment
    'fragment_safe',
    # Auto-refresh
    'setup_auto_refresh',
    'render_auto_refresh_toggle',
    # Cleanup
    'cleanup_stale_cache',
    'init_performance_state',
]


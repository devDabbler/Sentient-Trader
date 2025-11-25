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
TOTP_SECRET = os.getenv("CONTROL_PANEL_TOTP_SECRET")
TOTP_ENABLED = TOTP_AVAILABLE and TOTP_SECRET and len(TOTP_SECRET) >= 16

# Service intervals config file path
SERVICE_INTERVALS_FILE = Path(__file__).parent / "data" / "service_intervals.json"

SERVICES = {
    "DEX Launch Monitor": {
        "name": "sentient-dex-launch",
        "description": "Scans for new token launches on DEX platforms (Solana, ETH, etc.)",
        "emoji": "🚀",
        "category": "crypto",
        "interval_key": "scan_interval_seconds",
        "interval_default": 30,
        "interval_min": 10,
        "interval_max": 300
    },
    "Crypto Breakout Monitor": {
        "name": "sentient-crypto-breakout",
        "description": "Monitors crypto for breakout patterns and momentum",
        "emoji": "📈",
        "category": "crypto",
        "interval_key": "scan_interval_seconds",
        "interval_default": 180,
        "interval_min": 60,
        "interval_max": 600
    },
    "AI Crypto Trader": {
        "name": "sentient-crypto-ai-trader",
        "description": "AI-powered crypto position manager (executes trades)",
        "emoji": "🤖",
        "category": "crypto",
        "interval_key": "check_interval_seconds",
        "interval_default": 60,
        "interval_min": 30,
        "interval_max": 300
    },
    "Stock Monitor": {
        "name": "sentient-stock-monitor",
        "description": "Monitors stocks for trading opportunities",
        "emoji": "📊",
        "category": "stocks",
        "interval_key": "scan_interval_seconds",
        "interval_default": 300,
        "interval_min": 60,
        "interval_max": 900
    },
    "Discord Approval Bot": {
        "name": "sentient-discord-approval",
        "description": "Discord bot for trade approvals (recommended: keep auto-start)",
        "emoji": "💬",
        "category": "infrastructure"
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
    if not interval_key:
        return 0
    
    service_config = intervals.get(service_name, {})
    return service_config.get(interval_key, svc_info.get("interval_default", 60))


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
            'sentient-crypto-ai-trader': 'SentientCryptoAI'
        }
        svc_to_control = windows_name_map.get(service_name, service_name)
        # Use sc query for Windows
        # Map systemd-style service names to Windows service names where possible
        windows_name_map = {
            'sentient-stock-monitor': 'SentientStockMonitor',
            'sentient-dex-launch': 'SentientDEXLaunch',
            'sentient-crypto-breakout': 'SentientCryptoBreakout',
            'sentient-discord-approval': 'SentientDiscordApproval',
            'sentient-crypto-ai-trader': 'SentientCryptoAI'
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
            'sentient-crypto-ai-trader': 'SentientCryptoAI'
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
    project_root = Path(__file__).parent
    
    # Map service to likely log filenames - MUST match systemd StandardOutput/StandardError paths
    log_map = {
        'sentient-stock-monitor': [
            'logs/stock_monitor_service.log',
            'logs/stock_monitor_error.log',
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
        ]
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
    project_root = Path(__file__).parent
    
    # Map service to log filenames - MUST match systemd StandardOutput/StandardError paths
    log_map = {
        'sentient-stock-monitor': [
            'logs/stock_monitor_service.log',
            'logs/stock_monitor_error.log',
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
        ]
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


def check_password():
    """Password + TOTP authentication"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
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
            
            totp_code = st.text_input("📲 Enter 6-digit code from Authenticator", max_chars=6, key="totp_input")
            
            if st.button("Verify Code", type="primary"):
                # runtime guard for static type checkers
                assert pyotp is not None, "pyotp module is required for TOTP functionality"
                totp = pyotp.TOTP(TOTP_SECRET)
                if totp.verify(totp_code, valid_window=1):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Invalid code. Try again.")
                    print(f"[SECURITY] Failed TOTP attempt at {datetime.now()}")
        
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
    
    return True


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
            st.session_state.authenticated = False
            st.rerun()
        
        st.markdown("---")
        st.markdown(f"**Last refresh:** {datetime.now().strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        # Auto-refresh option - uses query param trick
        st.markdown("---")
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=False, key="auto_refresh", 
                                   help="Enable automatic page refresh every 30 seconds")
        if auto_refresh:
            # Use streamlit's built-in auto-rerun with a placeholder
            st.markdown("_Auto-refresh enabled (30s)_")
            # Add a meta refresh tag via markdown
            st.markdown(
                """
                <meta http-equiv="refresh" content="30">
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.markdown("### 🎛️ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start All"):
                for svc in SERVICES.values():
                    control_service(svc["name"], "start")
                st.rerun()
        with col2:
            if st.button("⏹️ Stop All"):
                for svc in SERVICES.values():
                    control_service(svc["name"], "stop")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Overview")
        running = sum(1 for s in SERVICES.values() if get_service_status(s["name"])["active"])
        st.metric("Services Running", f"{running}/{len(SERVICES)}")
    
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
                        if st.button("▶️", key=f"start_{service_name}", help="Start"):
                            success, msg = control_service(service_name, "start")
                            if success:
                                st.toast(f"✅ {display_name} started!")
                            else:
                                st.error(msg)
                            st.rerun()
                    
                    with btn_col2:
                        if st.button("⏹️", key=f"stop_{service_name}", help="Stop"):
                            success, msg = control_service(service_name, "stop")
                            if success:
                                st.toast(f"⏹️ {display_name} stopped!")
                            else:
                                st.error(msg)
                            st.rerun()
                    
                    with btn_col3:
                        if st.button("🔄", key=f"restart_{service_name}", help="Restart"):
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
                        if st.button("🔕 Disable", key=f"disable_{service_name}"):
                            control_service(service_name, "disable")
                            st.rerun()
                    else:
                        if st.button("🔔 Enable", key=f"enable_{service_name}"):
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
                        interval_min = svc_info.get("interval_min", 10)
                        interval_max = svc_info.get("interval_max", 600)
                        interval_key = svc_info.get("interval_key", "scan_interval_seconds")
                        
                        st.markdown(f"**⏱️ Scan/Check Interval**")
                        st.caption(f"How often the service checks for updates (in seconds)")
                        
                        # Show current interval
                        st.info(f"Current: **{current_interval}s** ({current_interval/60:.1f} min)")
                        
                        # Slider for new interval
                        new_interval = st.slider(
                            "New Interval (seconds)",
                            min_value=interval_min,
                            max_value=interval_max,
                            value=current_interval,
                            step=10,
                            key=f"interval_{service_name}",
                            help=f"Range: {interval_min}s - {interval_max}s"
                        )
                        
                        # Apply button
                        col_apply, col_info = st.columns([1, 2])
                        with col_apply:
                            if st.button("💾 Apply & Restart", key=f"apply_interval_{service_name}", type="primary"):
                                if set_service_interval(service_name, svc_info, new_interval):
                                    # Restart the service to apply new interval
                                    success, msg = control_service(service_name, "restart")
                                    if success:
                                        st.toast(f"✅ Interval set to {new_interval}s and service restarted!")
                                    else:
                                        st.warning(f"Interval saved but restart failed: {msg}")
                                    st.rerun()
                                else:
                                    st.error("Failed to save interval")
                        
                        with col_info:
                            st.caption(f"💡 Changes require a service restart to take effect")
                        
                        # Show quick presets
                        st.markdown("**Quick Presets:**")
                        preset_col1, preset_col2, preset_col3 = st.columns(3)
                        
                        with preset_col1:
                            if st.button("🚀 Fast (30s)", key=f"preset_fast_{service_name}"):
                                preset_val = max(30, interval_min)
                                set_service_interval(service_name, svc_info, preset_val)
                                control_service(service_name, "restart")
                                st.rerun()
                        
                        with preset_col2:
                            if st.button("⚖️ Normal (60s)", key=f"preset_normal_{service_name}"):
                                set_service_interval(service_name, svc_info, 60)
                                control_service(service_name, "restart")
                                st.rerun()
                        
                        with preset_col3:
                            if st.button("🐢 Slow (120s)", key=f"preset_slow_{service_name}"):
                                set_service_interval(service_name, svc_info, 120)
                                control_service(service_name, "restart")
                                st.rerun()
                
                st.markdown("---")
    
    # System Info
    with st.expander("🖥️ System Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            success, uptime = run_command("uptime -p")
            st.metric("Server Uptime", uptime if success else "N/A")
        
        with col2:
            success, memory = run_command("free -h | grep Mem | awk '{print $3\"/\"$2}'")
            st.metric("Memory Usage", memory if success else "N/A")


if __name__ == "__main__":
    main()

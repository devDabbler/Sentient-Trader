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
from datetime import datetime

# Try to import TOTP library
try:
    import pyotp
    import qrcode
    import io
    import base64
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================
# Password - set in environment variable or use default
ADMIN_PASSWORD = os.getenv("CONTROL_PANEL_PASSWORD", "admin")  # Set your own

# TOTP Secret for 2FA (generate with: python -c "import pyotp; print(pyotp.random_base32())")
TOTP_SECRET = os.getenv("CONTROL_PANEL_TOTP_SECRET")
TOTP_ENABLED = TOTP_AVAILABLE and TOTP_SECRET and len(TOTP_SECRET) >= 16

SERVICES = {
    "DEX Launch Monitor": {
        "name": "sentient-dex-launch",
        "description": "Scans for new token launches on DEX platforms (Solana, ETH, etc.)",
        "emoji": "🚀",
        "category": "crypto"
    },
    "Crypto Breakout Monitor": {
        "name": "sentient-crypto-breakout",
        "description": "Monitors crypto for breakout patterns and momentum",
        "emoji": "📈",
        "category": "crypto"
    },
    "AI Crypto Trader": {
        "name": "sentient-crypto-ai-trader",
        "description": "AI-powered crypto position manager (executes trades)",
        "emoji": "🤖",
        "category": "crypto"
    },
    "Stock Monitor": {
        "name": "sentient-stock-monitor",
        "description": "Monitors stocks for trading opportunities",
        "emoji": "📊",
        "category": "stocks"
    },
    "Discord Approval Bot": {
        "name": "sentient-discord-approval",
        "description": "Discord bot for trade approvals (recommended: keep auto-start)",
        "emoji": "💬",
        "category": "infrastructure"
    }
}

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
    """Get status of a systemd service"""
    # Check if active
    success, output = run_command(f"systemctl is-active {service_name}")
    is_active = output == "active"
    
    # Check if enabled (starts on boot)
    success2, output2 = run_command(f"systemctl is-enabled {service_name}")
    is_enabled = output2 == "enabled"
    
    # Get uptime/status details
    success3, details = run_command(f"systemctl status {service_name} --no-pager -n 0")
    
    # Parse memory usage if available
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
    """Start, stop, restart, enable, or disable a service"""
    if action in ["start", "stop", "restart"]:
        cmd = f"sudo systemctl {action} {service_name}"
    elif action == "enable":
        cmd = f"sudo systemctl enable {service_name}"
    elif action == "disable":
        cmd = f"sudo systemctl disable {service_name}"
    else:
        return False, f"Unknown action: {action}"
    
    return run_command(cmd)


def get_service_logs(service_name: str, lines: int = 50) -> str:
    """Get recent logs for a service"""
    success, output = run_command(f"journalctl -u {service_name} -n {lines} --no-pager")
    return output if success else f"Error fetching logs: {output}"


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
                    logs = get_service_logs(service_name, 30)
                    st.code(logs, language="log")
                
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

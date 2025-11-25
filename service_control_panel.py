"""
Sentient Trader - Service Control Panel

A simple web GUI to manage your VPS services from anywhere.
Password protected for security.

Usage (on VPS):
    streamlit run service_control_panel.py --server.port 8501 --server.address 0.0.0.0
    
Access from anywhere:
    http://YOUR_VPS_IP:8501
"""

import streamlit as st
import subprocess
import os
from datetime import datetime

# ============================================================
# CONFIGURATION - Change these!
# ============================================================
ADMIN_PASSWORD = os.getenv("CONTROL_PANEL_PASSWORD", "sentient2025")  # Change this!

SERVICES = {
    "DEX Launch Monitor": {
        "name": "sentient-dex-launch",
        "description": "Scans for new token launches on DEX platforms",
        "emoji": "🚀"
    },
    "Crypto Breakout Monitor": {
        "name": "sentient-crypto-breakout",
        "description": "Monitors crypto for breakout patterns",
        "emoji": "📈"
    },
    "Stock Monitor": {
        "name": "sentient-stock-monitor",
        "description": "Monitors stocks for trading opportunities",
        "emoji": "📊"
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_command(cmd: str) -> tuple[bool, str]:
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


def control_service(service_name: str, action: str) -> tuple[bool, str]:
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
    """Simple password authentication"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Sentient Trader Control Panel")
        st.markdown("---")
        
        password = st.text_input("Enter Password", type="password")
        
        if st.button("Login", type="primary"):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        
        st.markdown("---")
        st.caption("Access your trading services from anywhere")
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
    
    # Main content - Service Cards
    st.markdown("---")
    
    for display_name, svc_info in SERVICES.items():
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
                            st.success("Started!")
                        else:
                            st.error(msg)
                        st.rerun()
                
                with btn_col2:
                    if st.button("⏹️", key=f"stop_{service_name}", help="Stop"):
                        success, msg = control_service(service_name, "stop")
                        if success:
                            st.success("Stopped!")
                        else:
                            st.error(msg)
                        st.rerun()
                
                with btn_col3:
                    if st.button("🔄", key=f"restart_{service_name}", help="Restart"):
                        success, msg = control_service(service_name, "restart")
                        if success:
                            st.success("Restarted!")
                        else:
                            st.error(msg)
                        st.rerun()
            
            with col3:
                st.markdown("**Auto-Start**")
                
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

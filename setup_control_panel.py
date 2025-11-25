"""
Setup script to install the Service Control Panel on your VPS.

Run this on your VPS after copying the files:
    python setup_control_panel.py
"""

import subprocess
import os

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

print("=" * 60)
print("🤖 Sentient Trader - Control Panel Setup")
print("=" * 60)

# 1. Create systemd service for the control panel
service_content = """[Unit]
Description=Sentient Trader Control Panel
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/sentient-trader
Environment="PATH=/root/sentient-trader/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="CONTROL_PANEL_PASSWORD=sentient2025"
ExecStart=/root/sentient-trader/venv/bin/streamlit run service_control_panel.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

print("\n1. Creating systemd service file...")
with open("/etc/systemd/system/sentient-control-panel.service", "w") as f:
    f.write(service_content)
print("   ✅ Created /etc/systemd/system/sentient-control-panel.service")

# 2. Open firewall port
print("\n2. Opening firewall port 8501...")
run_cmd("ufw allow 8501/tcp")
print("   ✅ Port 8501 opened")

# 3. Reload systemd and enable service
print("\n3. Enabling control panel service...")
run_cmd("systemctl daemon-reload")
run_cmd("systemctl enable sentient-control-panel")
run_cmd("systemctl start sentient-control-panel")
print("   ✅ Service enabled and started")

# 4. Get VPS IP
print("\n4. Getting your VPS IP address...")
result = subprocess.run("curl -s ifconfig.me", shell=True, capture_output=True, text=True)
vps_ip = result.stdout.strip()

print("\n" + "=" * 60)
print("🎉 SETUP COMPLETE!")
print("=" * 60)
print(f"\n🌐 Access your control panel at:")
print(f"   http://{vps_ip}:8501")
print(f"\n🔐 Default password: sentient2025")
print(f"   (Change it in /etc/systemd/system/sentient-control-panel.service)")
print("\n📋 Useful commands:")
print("   sudo systemctl status sentient-control-panel  # Check status")
print("   sudo systemctl restart sentient-control-panel # Restart")
print("   sudo journalctl -u sentient-control-panel -f  # View logs")
print("=" * 60)

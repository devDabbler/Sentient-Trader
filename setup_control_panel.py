"""
Setup script to install the Service Control Panel on your VPS.

Run this on your VPS after copying the files:
    python setup_control_panel.py
"""

import subprocess
import os
import secrets
import string

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

def generate_strong_password(length=24):
    """Generate a strong random password"""
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(chars) for _ in range(length))

def generate_totp_secret():
    """Generate a TOTP secret (base32)"""
    try:
        import pyotp
        return pyotp.random_base32()
    except ImportError:
        # Fallback: generate base32 manually
        import base64
        return base64.b32encode(secrets.token_bytes(20)).decode('utf-8')

print("=" * 60)
print("🤖 Sentient Trader - Control Panel Setup")
print("=" * 60)

# 0. Install required packages
print("\n0. Installing required packages...")
run_cmd("pip install pyotp qrcode[pil]")

# 1. Get credentials from user
print("\n1. Setting up credentials...")
print("   Enter your desired password (any length, 2FA adds security):")
password = input("   Password: ").strip()
if not password:
    password = "admin"
    print(f"   Using default password: {password}")

totp_secret = generate_totp_secret()
print(f"   ✅ Generated TOTP secret for 2FA")

# 2. Create/update .env file with credentials
env_file = "/root/sentient-trader/.env"
print(f"\n2. Adding credentials to {env_file}...")

# Read existing .env
env_content = ""
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        env_content = f.read()

# Remove old control panel settings if they exist
lines = env_content.split('\n')
lines = [l for l in lines if not l.startswith('CONTROL_PANEL_')]
env_content = '\n'.join(lines)

# Add new credentials
new_settings = f"""
# ============================================================
# Service Control Panel (added by setup_control_panel.py)
# ============================================================
CONTROL_PANEL_PASSWORD={password}
CONTROL_PANEL_TOTP_SECRET={totp_secret}
"""

with open(env_file, 'a') as f:
    f.write(new_settings)

print(f"   ✅ Credentials saved to .env")

# 3. Create systemd service for the control panel
service_content = f"""[Unit]
Description=Sentient Trader Control Panel
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/sentient-trader
EnvironmentFile=/root/sentient-trader/.env
Environment="PATH=/root/sentient-trader/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/root/sentient-trader/venv/bin/streamlit run service_control_panel.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

print("\n3. Creating systemd service file...")
with open("/etc/systemd/system/sentient-control-panel.service", "w") as f:
    f.write(service_content)
print("   ✅ Created /etc/systemd/system/sentient-control-panel.service")

# 4. Open firewall port
print("\n4. Opening firewall port 8501...")
run_cmd("ufw allow 8501/tcp")
print("   ✅ Port 8501 opened")

# 5. Reload systemd and enable service
print("\n5. Enabling control panel service...")
run_cmd("systemctl daemon-reload")
run_cmd("systemctl enable sentient-control-panel")
run_cmd("systemctl restart sentient-control-panel")
print("   ✅ Service enabled and started")

# 6. Get VPS IP
print("\n6. Getting your VPS IP address...")
result = subprocess.run("curl -s ifconfig.me", shell=True, capture_output=True, text=True)
vps_ip = result.stdout.strip()

print("\n" + "=" * 60)
print("🎉 SETUP COMPLETE!")
print("=" * 60)

print(f"""
🌐 Access your control panel at:
   http://{vps_ip}:8501

🔐 Login Credentials (SAVE THESE!):
   Password: {password}
   
📱 Two-Factor Authentication (2FA):
   TOTP Secret: {totp_secret}
   
   To set up 2FA:
   1. Open Google Authenticator / Authy on your phone
   2. Add new account → Scan QR code (shown on login page)
   3. Or manually enter the secret above

⚠️  IMPORTANT: Save these credentials securely!
   They are also stored in: /root/sentient-trader/.env

📋 Useful commands:
   sudo systemctl status sentient-control-panel  # Check status
   sudo systemctl restart sentient-control-panel # Restart
   sudo journalctl -u sentient-control-panel -f  # View logs
""")
print("=" * 60)

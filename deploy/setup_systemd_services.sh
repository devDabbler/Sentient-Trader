#!/bin/bash
# Create systemd service files for background services
# Updated: Now includes EnvironmentFile for .env loading (fallback for load_dotenv)

set -e

USER=$(whoami)
# Handle root user (home is /root, not /home/root)
if [ "$USER" = "root" ]; then
    PROJECT_DIR="/root/sentient-trader"
else
    PROJECT_DIR="/home/$USER/sentient-trader"
fi

echo "Creating systemd service files..."

# Create systemd-compatible env file from .env (strips quotes and comments)
echo "Creating systemd-compatible environment file..."
if [ -f "$PROJECT_DIR/.env" ]; then
    # Convert .env to systemd format:
    # 1. Remove comment lines (starting with #)
    # 2. Remove empty lines
    # 3. Remove 'export ' prefix if present
    # 4. Remove surrounding quotes from values (both single and double)
    grep -v '^\s*#' "$PROJECT_DIR/.env" | grep -v '^\s*$' | sed 's/^export //' | sed "s/=\"/=/" | sed "s/\"$//" | sed "s/='/=/" | sed "s/'$//" > "$PROJECT_DIR/.env.systemd"
    
    # Count variables
    VAR_COUNT=$(wc -l < "$PROJECT_DIR/.env.systemd")
    chmod 600 "$PROJECT_DIR/.env.systemd"
    echo "  ✅ Created $PROJECT_DIR/.env.systemd ($VAR_COUNT variables)"
    
    # Verify critical variables exist
    if grep -q "DISCORD_WEBHOOK_URL" "$PROJECT_DIR/.env.systemd"; then
        echo "  ✅ DISCORD_WEBHOOK_URL found"
    else
        echo "  ⚠️  WARNING: DISCORD_WEBHOOK_URL not found in .env!"
    fi
    if grep -q "KRAKEN_API_KEY" "$PROJECT_DIR/.env.systemd"; then
        echo "  ✅ KRAKEN_API_KEY found"
    else
        echo "  ⚠️  WARNING: KRAKEN_API_KEY not found in .env!"
    fi
else
    echo "  ⚠️  WARNING: $PROJECT_DIR/.env not found!"
    echo "     Services may not have required API keys (Discord, OpenAI, Kraken, etc.)"
    touch "$PROJECT_DIR/.env.systemd"
fi

# Stock Monitor Service
sudo tee /etc/systemd/system/sentient-stock-monitor.service > /dev/null << EOF
[Unit]
Description=Sentient Trader - Stock Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=-$PROJECT_DIR/.env.systemd
ExecStart=$PROJECT_DIR/venv/bin/python3 windows_services/runners/run_stock_monitor_simple.py
Restart=always
RestartSec=10
StandardOutput=append:$PROJECT_DIR/logs/stock_monitor_service.log
StandardError=append:$PROJECT_DIR/logs/stock_monitor_error.log

[Install]
WantedBy=multi-user.target
EOF

# Crypto Breakout Service
sudo tee /etc/systemd/system/sentient-crypto-breakout.service > /dev/null << EOF
[Unit]
Description=Sentient Trader - Crypto Breakout Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=-$PROJECT_DIR/.env.systemd
ExecStart=$PROJECT_DIR/venv/bin/python3 windows_services/runners/run_crypto_breakout_simple.py
Restart=always
RestartSec=10
StandardOutput=append:$PROJECT_DIR/logs/crypto_breakout_service.log
StandardError=append:$PROJECT_DIR/logs/crypto_breakout_error.log

[Install]
WantedBy=multi-user.target
EOF

# DEX Launch Service
sudo tee /etc/systemd/system/sentient-dex-launch.service > /dev/null << EOF
[Unit]
Description=Sentient Trader - DEX Launch Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=-$PROJECT_DIR/.env.systemd
ExecStart=$PROJECT_DIR/venv/bin/python3 windows_services/runners/run_dex_launch_simple.py
Restart=always
RestartSec=10
StandardOutput=append:$PROJECT_DIR/logs/dex_launch_service.log
StandardError=append:$PROJECT_DIR/logs/dex_launch_error.log

[Install]
WantedBy=multi-user.target
EOF

# AI Crypto Position Manager Service (OPTIONAL - Only enable if you want AI to execute trades)
sudo tee /etc/systemd/system/sentient-crypto-ai-trader.service > /dev/null << EOF
[Unit]
Description=Sentient Trader - AI Crypto Position Manager
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=-$PROJECT_DIR/.env.systemd
ExecStart=$PROJECT_DIR/venv/bin/python3 windows_services/runners/run_crypto_ai_position_manager_simple.py
Restart=always
RestartSec=10
StandardOutput=append:$PROJECT_DIR/logs/crypto_ai_position_manager_service.log
StandardError=append:$PROJECT_DIR/logs/crypto_ai_position_manager_error.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable services (start on boot)
echo "Enabling services..."
sudo systemctl enable sentient-stock-monitor
sudo systemctl enable sentient-crypto-breakout
sudo systemctl enable sentient-dex-launch
# AI trader NOT auto-enabled - must be started manually

# Start services
echo "Starting services..."
sudo systemctl start sentient-stock-monitor
sudo systemctl start sentient-crypto-breakout
sudo systemctl start sentient-dex-launch
sudo systemctl start sentient-crypto-ai-trader

echo ""
echo "=========================================="
echo "  ✅ SERVICES CONFIGURED!"
echo "=========================================="
echo ""
echo "Service Status:"
sudo systemctl status sentient-stock-monitor --no-pager -l
sudo systemctl status sentient-crypto-breakout --no-pager -l
sudo systemctl status sentient-dex-launch --no-pager -l
sudo systemctl status sentient-crypto-ai-trader --no-pager -l
echo ""
echo "ℹ️  AI Trading Service: STARTED (but NOT auto-boot)"
echo "   Service name: sentient-crypto-ai-trader"
echo "   Will NOT restart on server reboot - start manually with:"
echo "   sudo systemctl start sentient-crypto-ai-trader"
echo ""
echo "Useful Commands:"
echo "  View logs:     tail -f $PROJECT_DIR/logs/stock_monitor_service.log"
echo "  Restart:       sudo systemctl restart sentient-stock-monitor"
echo "  Stop:          sudo systemctl stop sentient-stock-monitor"
echo "  Check status:  sudo systemctl status sentient-stock-monitor"
echo ""
echo "AI Trading Commands (if enabled):"
echo "  View AI logs:  tail -f $PROJECT_DIR/logs/crypto_ai_position_manager_service.log"
echo "  Start AI:      sudo systemctl start sentient-crypto-ai-trader"
echo "  Stop AI:       sudo systemctl stop sentient-crypto-ai-trader"
echo ""

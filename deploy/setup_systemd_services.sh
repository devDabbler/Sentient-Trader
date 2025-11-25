#!/bin/bash
# Create systemd service files for background services

set -e

USER=$(whoami)
PROJECT_DIR="/home/$USER/sentient-trader"

echo "Creating systemd service files..."

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

# OPTIONAL: Enable AI trading (ONLY if you want AI to execute trades automatically)
# Uncomment the lines below to enable AI position management:
# sudo systemctl enable sentient-crypto-ai-trader

# Start services
echo "Starting services..."
sudo systemctl start sentient-stock-monitor
sudo systemctl start sentient-crypto-breakout
sudo systemctl start sentient-dex-launch

# OPTIONAL: Start AI trading service
# Uncomment to start AI position manager:
# sudo systemctl start sentient-crypto-ai-trader

echo ""
echo "=========================================="
echo "  ✅ SERVICES CONFIGURED!"
echo "=========================================="
echo ""
echo "Service Status:"
sudo systemctl status sentient-stock-monitor --no-pager -l
sudo systemctl status sentient-crypto-breakout --no-pager -l
sudo systemctl status sentient-dex-launch --no-pager -l
echo ""
echo "⚠️  OPTIONAL: AI Trading Service (Currently DISABLED)"
echo "   To enable AI auto-trading, edit this script and uncomment the AI service lines"
echo "   Service name: sentient-crypto-ai-trader"
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

#!/bin/bash
# First-time server setup for Sentient Trader (Non-interactive)
# Run this ONCE on a fresh Digital Ocean droplet

set -e

echo "=========================================="
echo "  Sentient Trader - First Time Setup"
echo "=========================================="
echo ""
echo "Installing dependencies..."
echo ""

echo "Step 1: Updating system packages..."
apt-get update
apt-get upgrade -y

echo ""
echo "Step 2: Installing Python and build tools..."
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3.11-dev \
    build-essential \
    git \
    unzip \
    wget \
    curl \
    software-properties-common

echo ""
echo "Step 3: Creating project directory..."
PROJECT_DIR="/root/sentient-trader"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo ""
echo "Step 4: Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

echo ""
echo "Step 5: Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 6: Creating required directories..."
mkdir -p logs
mkdir -p data
mkdir -p backups

echo ""
echo "Step 7: Creating placeholder .env file..."
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# API Keys - REPLACE WITH YOUR ACTUAL KEYS!
OPENROUTER_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
DISCORD_WEBHOOK_URL=your_webhook_here

# Broker API Keys
TRADIER_API_KEY=your_key_here
IBKR_USERNAME=your_username_here
IBKR_PASSWORD=your_password_here
KRAKEN_API_KEY=your_key_here
KRAKEN_API_SECRET=your_secret_here

# Environment
ENVIRONMENT=production

# Logging
LOG_LEVEL=INFO
EOF
    echo "  ✓ Created .env template"
else
    echo "  ⚠ .env already exists, skipping..."
fi

echo ""
echo "=========================================="
echo "  ✅ FIRST-TIME SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Python virtual env: $PROJECT_DIR/venv"
echo ""
echo "NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. CLONE YOUR REPO:"
echo "   cd /root/sentient-trader"
echo "   git clone https://github.com/devDabbler/Sentient-Trader.git ."
echo ""
echo "2. CONFIGURE ENVIRONMENT:"
echo "   nano .env"
echo "   (Add your actual API keys)"
echo ""
echo "3. INSTALL DEPENDENCIES:"
echo "   source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "4. START SERVICES:"
echo "   chmod +x deploy/setup_systemd_services.sh"
echo "   ./deploy/setup_systemd_services.sh"
echo ""
echo "5. CHECK SERVICE STATUS:"
echo "   sudo systemctl status sentient-stock-monitor"
echo "   tail -f logs/stock_monitor_service.log"
echo ""

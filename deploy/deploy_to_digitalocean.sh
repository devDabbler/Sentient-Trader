#!/bin/bash
# Deploy Sentient Trader to DigitalOcean Droplet
# Run this ON THE SERVER after uploading your code

set -e

echo "=========================================="
echo "  Sentient Trader - Server Setup"
echo "=========================================="
echo ""

# Update system
echo "1. Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python 3.11+ and dependencies
echo "2. Installing Python and dependencies..."
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# Create project directory
echo "3. Setting up project directory..."
cd /home/$USER
PROJECT_DIR="/home/$USER/sentient-trader"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
echo "4. Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python packages
echo "5. Installing Python requirements..."
# You'll need to upload your requirements.txt
pip install --upgrade pip
pip install loguru requests python-dotenv yfinance pandas numpy

# Create necessary directories
echo "6. Creating log directories..."
mkdir -p logs
mkdir -p data

# Create .env file template
echo "7. Creating environment file template..."
cat > .env << 'EOF'
# API Keys - FILL THESE IN!
OPENROUTER_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
DISCORD_WEBHOOK_URL=your_webhook_here

# Other settings
ENVIRONMENT=production
EOF

echo ""
echo "=========================================="
echo "  ✅ BASIC SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo "1. Upload your code to: $PROJECT_DIR"
echo "2. Edit .env file with your API keys: nano .env"
echo "3. Run the systemd setup script"
echo ""

#!/bin/bash
# Download historical data from Kraken for backtesting
# Usage: ./download_freqtrade_data.sh [days]
# Example: ./download_freqtrade_data.sh 60

cd /root/Sentient-Trader

# Activate virtual environment
source venv/bin/activate

# Load environment variables
set -a
source .env
set +a

DAYS=${1:-30}
END_DATE=$(date +%Y%m%d)
START_DATE=$(date -d "-${DAYS} days" +%Y%m%d)

echo "Downloading Kraken data for backtesting"
echo "Date range: $START_DATE - $END_DATE"
echo "Pairs: BTC/USD, ETH/USD, SOL/USD, XRP/USD, ADA/USD, AVAX/USD, DOT/USD, LINK/USD, MATIC/USD, ATOM/USD"
echo ""

freqtrade download-data \
    --exchange kraken \
    --pairs BTC/USD ETH/USD SOL/USD XRP/USD ADA/USD AVAX/USD DOT/USD LINK/USD MATIC/USD ATOM/USD \
    --timeframes 5m 15m 1h 4h \
    --userdir freqtrade_userdata \
    --timerange "${START_DATE}-${END_DATE}"

echo ""
echo "Download complete. Data saved to freqtrade_userdata/data/kraken/"

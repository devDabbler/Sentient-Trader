#!/bin/bash
# Download historical data from Kraken for backtesting
# Usage: ./download_freqtrade_data.sh [days] [pairs]
# Example: ./download_freqtrade_data.sh 30
# Example: ./download_freqtrade_data.sh 7 "BTC/USD ETH/USD"
# Example: ./download_freqtrade_data.sh 14 "SOL/USD XRP/USD DOGE/USD"

cd /root/sentient-trader

# Activate virtual environment
source venv/bin/activate

# Load environment variables
set -a
source .env
set +a

DAYS=${1:-30}
DEFAULT_PAIRS="BTC/USD ETH/USD SOL/USD"
PAIRS=${2:-$DEFAULT_PAIRS}
END_DATE=$(date +%Y%m%d)
START_DATE=$(date -d "-${DAYS} days" +%Y%m%d)

echo "Downloading Kraken data for backtesting"
echo "Date range: $START_DATE - $END_DATE"
echo "Pairs: $PAIRS"
echo "Note: Kraken uses --dl-trades (slower than klines)"
echo ""

freqtrade download-data \
    --exchange kraken \
    --pairs $PAIRS \
    --timeframes 5m 15m 1h 4h \
    --userdir freqtrade_userdata \
    --timerange "${START_DATE}-${END_DATE}" \
    --dl-trades

echo ""
echo "Download complete. Data saved to freqtrade_userdata/data/kraken/"

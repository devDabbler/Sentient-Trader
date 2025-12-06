#!/bin/bash
# Backtest Freqtrade strategy
# Usage: ./backtest_freqtrade.sh [strategy] [days]
# Example: ./backtest_freqtrade.sh SentientStrategy 30
# Example: ./backtest_freqtrade.sh SentientFreqAIStrategy 60

cd /root/Sentient-Trader

# Activate virtual environment
source venv/bin/activate

# Load environment variables
set -a
source .env
set +a

STRATEGY=${1:-SentientStrategy}
DAYS=${2:-30}
END_DATE=$(date +%Y%m%d)
START_DATE=$(date -d "-${DAYS} days" +%Y%m%d)

echo "========================================="
echo "Freqtrade Backtest"
echo "========================================="
echo "Strategy: $STRATEGY"
echo "Date range: $START_DATE - $END_DATE"
echo "Exchange: Kraken"
echo ""

# Check if FreqAI strategy
if [[ "$STRATEGY" == *"FreqAI"* ]]; then
    FREQAI_ARGS="--freqaimodel LightGBMRegressor"
    echo "FreqAI enabled with LightGBMRegressor"
else
    FREQAI_ARGS=""
fi

echo ""
echo "Starting backtest..."
echo ""

freqtrade backtesting \
    --config freqtrade_userdata/config.json \
    --strategy "$STRATEGY" \
    --userdir freqtrade_userdata \
    --timerange "${START_DATE}-${END_DATE}" \
    $FREQAI_ARGS \
    --export trades

echo ""
echo "Backtest complete. Results exported to freqtrade_userdata/backtest_results/"

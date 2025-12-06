#!/bin/bash
# Start Freqtrade in paper trading mode (dry-run)
# Usage: ./start_freqtrade.sh [strategy]
# Example: ./start_freqtrade.sh SentientStrategy
# Example: ./start_freqtrade.sh SentientFreqAIStrategy

cd /root/Sentient-Trader

# Activate virtual environment
source venv/bin/activate

# Load environment variables
set -a
source .env
set +a

# Default strategy
STRATEGY=${1:-SentientStrategy}

# Check if FreqAI strategy
if [[ "$STRATEGY" == *"FreqAI"* ]]; then
    FREQAI_ARGS="--freqaimodel LightGBMRegressor"
else
    FREQAI_ARGS=""
fi

echo "Starting Freqtrade with strategy: $STRATEGY"
echo "Mode: Paper Trading (dry-run)"
echo "Exchange: Kraken"
echo "Timestamp: $(date)"

# Create log directory if not exists
mkdir -p logs

# Run Freqtrade
freqtrade trade \
    --config freqtrade_userdata/config.json \
    --strategy "$STRATEGY" \
    --userdir freqtrade_userdata \
    $FREQAI_ARGS \
    --dry-run \
    2>&1 | tee -a "logs/freqtrade_$(date +%Y%m%d).log"

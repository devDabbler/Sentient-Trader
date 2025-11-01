# Sentient Trader Platform - GitHub Copilot Instructions

## Repository Overview

This is **Sentient Trader**, a comprehensive AI-driven options and stock trading platform built with Python and Streamlit. The platform provides real-time market analysis, automated trading capabilities, technical indicators, news integration, and intelligent strategy recommendations for both manual and automated trading.

**Key Technologies:**
- **Frontend:** Streamlit web application (main interface)
- **Backend:** Python 3.9+ with asyncio for concurrent operations
- **APIs:** Tradier (live/paper trading), yfinance (market data), various news APIs
- **AI/ML:** OpenAI, Anthropic Claude, Google Gemini for trading signals and analysis
- **Database:** SQLite for trade journaling, Supabase for cloud storage
- **Integrations:** Option Alpha webhooks, Discord notifications, Reddit/social sentiment

## Project Architecture

### Core Application Structure
- **`app.py`** - Main Streamlit application (10,000+ lines) with tabbed interface
- **`services/`** - Core business logic (trading signals, scanners, analysis)
- **`analyzers/`** - Market analysis modules (technical, news, comprehensive)
- **`clients/`** - External API integrations (Tradier, Supabase, Option Alpha)
- **`ui/`** - Streamlit UI components and helpers
- **`utils/`** - Utility functions and data processing
- **`tests/`** - Pytest test suite with asyncio support

### Configuration System
The platform uses multiple configuration files for different trading modes:
- **`config_paper_trading.py`** - Safe paper trading settings
- **`config_live_trading.py`** - Production trading with strict risk limits
- **`config_background_trader.py`** - Automated background trader settings
- **`config_warrior_scalping.py`** - High-frequency scalping configuration
- **`config_swing_trader.py`** - Swing trading strategy settings
- **`config_options_premium.py`** - Options premium strategy config

### Trading Modes & Strategies
1. **WARRIOR_SCALPING** - Gap & Go strategy with market-wide scanning
2. **SLOW_SCALPER** - PDT-safe intraday trading with cash rotation
3. **MICRO_SWING** - Key level rejections with T+2 settlement tracking
4. **OPTIONS** - Wheel strategy and premium collection
5. **STOCKS** - Traditional equity trading
6. **ALL** - Multi-strategy approach with dynamic selection

## Build and Run Instructions

### Environment Setup
```bash
# 1. Install Python 3.9+ and create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables (.env file required)
# Copy .env.example to .env and fill in API keys:
# - TRADIER_ACCOUNT_ID, TRADIER_ACCESS_TOKEN
# - OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
# - DISCORD_WEBHOOK_URL, SUPABASE_URL, SUPABASE_KEY
```

### Running the Application
```bash
# Main Streamlit application
streamlit run app.py

# Background automated trader
python run_autotrader_background.py

# Switch trading modes (Windows batch files)
switch_to_paper_trading.bat    # Safe testing
switch_to_live_trading.bat     # Production trading
```

### Testing
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_warrior_market_scanner.py -v
pytest tests/test_trading_signals.py -v

# Run with coverage
pytest --cov=services --cov=analyzers
```

### Development Workflow
1. **Always test in paper trading mode first** - Use `config_paper_trading.py` for development
2. **Check trading hours** - Platform enforces market hours (9:30 AM - 4:00 PM ET)
3. **Monitor logs** - Check `logs/` directory and `trading_signals.log` for debugging
4. **Test API connections** - Use built-in connection validators before trading

## Key Development Guidelines

### Code Style & Patterns
- **Async/Await:** Most trading operations use asyncio for non-blocking execution
- **Error Handling:** Comprehensive try/catch blocks with detailed logging
- **Type Hints:** Use typing annotations for better code clarity
- **Dataclasses:** Prefer dataclasses for structured data (TradingSignal, TopTrade, etc.)
- **Enum Classes:** Use enums for trading modes, signal types, and status values

### Critical Safety Rules
1. **Never bypass trading hour checks** - Always validate market hours before execution
2. **Respect rate limits** - APIs have limits, use proper delays and error handling
3. **Validate all orders** - Check account balance, position limits, and risk parameters
4. **Paper trading first** - Test all new features in paper mode before live trading
5. **PDT compliance** - Implement proper Pattern Day Trader rule adherence

### Configuration Management
- **Environment Variables:** Sensitive data (API keys) must be in `.env` file
- **Trading Configs:** Strategy-specific settings in `config_*.py` files
- **Capital Management:** Always implement position sizing and risk limits
- **Watchlist Management:** Use `services/watchlist_manager.py` for ticker lists

### Common Patterns
```python
# Trading signal generation
signal = AITradingSignalGenerator.generate_signal(ticker, analysis_data)

# Async data fetching
async def fetch_market_data(ticker: str) -> Dict:
    # Use aiohttp or asyncio-compatible libraries

# Streamlit caching for expensive operations
@st.cache_data(ttl=300)  # 5-minute cache
def get_market_data(ticker: str):
    # Expensive API calls

# Error handling with logging
try:
    result = await trading_operation()
except Exception as e:
    logger.error(f"Trading operation failed: {e}")
    st.error(f"Operation failed: {e}")
```

## Important Files & Locations

### Core Components
- **`app.py`** - Main application entry point (Streamlit tabs: Scanner, My Tickers, Auto-Trader)
- **`services/ai_trading_signals.py`** - AI-powered buy/sell signal generation
- **`services/top_trades_scanner.py`** - Market-wide opportunity scanner
- **`analyzers/comprehensive.py`** - Multi-factor analysis engine
- **`clients/tradier_client.py`** - Broker integration for order execution

### Documentation
- **`documentation/`** - Comprehensive guides for all features
- **`documentation/WARRIOR_SCANNER_QUICK_START.md`** - Scalping strategy guide
- **`documentation/AI_TRADING_GUIDE.md`** - AI signal configuration
- **`documentation/DISCORD_INTEGRATION_GUIDE.md`** - Alert setup

### Data & Logs
- **`data/`** - Static data files (ticker lists, strategy templates)
- **`logs/`** - Application logs and trading history
- **`trading_signals.log`** - Real-time trading signal log

## Integration Points

### External APIs
- **Tradier API** - Live and paper trading execution
- **yfinance** - Free market data (primary data source)
- **Option Alpha** - Webhook integration for automated strategies
- **Discord** - Real-time notifications and alerts
- **News APIs** - Sentiment analysis and catalyst detection

### Data Flow
1. **Market Data** → yfinance/Tradier → **Analysis Engine** → Technical indicators
2. **News/Events** → News APIs → **Sentiment Analysis** → Trading catalysts  
3. **AI Models** → OpenAI/Claude/Gemini → **Signal Generation** → Trading decisions
4. **Execution** → Tradier API → **Order Management** → Position tracking
5. **Notifications** → Discord webhooks → **Real-time alerts** → User notifications

## Common Issues & Solutions

### Windows-Specific
- **Asyncio Policy:** Set `WindowsProactorEventLoopPolicy()` for proper async support
- **Path Handling:** Use `os.path.join()` for cross-platform compatibility
- **Batch Files:** Use `.bat` files for easy mode switching

### API Rate Limits
- **yfinance:** Implement delays between requests, use batch processing
- **Tradier:** Respect 1-2 requests/second limit, implement exponential backoff
- **News APIs:** Cache responses and implement proper rate limiting

### Streamlit Issues
- **Caching:** Use `@st.cache_data()` for expensive operations
- **Session State:** Store user preferences and temporary data
- **Rerun Handling:** Use `st.rerun()` sparingly to avoid infinite loops

### Trading Errors
- **Insufficient Funds:** Always check account balance before orders
- **Market Hours:** Validate trading hours before execution
- **Position Limits:** Implement proper position sizing and risk management

## Performance Considerations

- **Parallel Processing:** Use ThreadPoolExecutor for market scanning (500+ tickers in <60s)
- **Caching Strategy:** Cache market data, analysis results, and expensive computations
- **Async Operations:** Use asyncio for non-blocking I/O operations
- **Memory Management:** Limit dataframe sizes and clear unused data
- **API Optimization:** Batch requests and implement intelligent retry logic

## Testing Strategy

- **Unit Tests:** Test individual components in isolation
- **Integration Tests:** Test API connections and data flows
- **Paper Trading Tests:** Validate strategies without real money
- **Performance Tests:** Ensure scanners complete within time limits
- **Error Handling Tests:** Verify graceful failure handling

Always run the full test suite before deploying any changes to live trading configurations.
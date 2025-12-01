# Cursor AI Assistant Rules - Sentient Trader

## Project Overview
**Sentient Trader** is a sophisticated AI-powered cryptocurrency and stock trading system with advanced DEX hunter capabilities for detecting Solana pump.fun launches and meme coins.

### Key Components:
- **DEX Hunter**: Solana on-chain token verification and scoring
- **Broker Integration**: IBKR (paper/live), Tradier, Kraken
- **Trading Strategies**: Multiple automated trading strategies (scalping, swing, warrior)
- **Monitoring Services**: Crypto monitors, stock monitors, Discord approval bot
- **Control Panel**: Centralized service management

---

## DEX Hunter System (December 2025)

### ✅ Production Status
- **Status**: PRODUCTION READY - All 3 phases complete
- **Running as**: systemd service (`sentient-dex-launch`)
- **Core Services**: 126+ Python services in `services/` directory

### Phase 1: Core Solana RPC Integration ✅ COMPLETE
**Files:**
- `services/solana_mint_inspector.py` - Mint/freeze authority inspection
- `services/solana_lp_analyzer.py` - LP token ownership verification
- `services/token_safety_analyzer.py` - Safety scoring and red flag detection

**Hard Red Flags (Auto-Reject):**
- Mint authority retained → Risk: Can mint more tokens
- Freeze authority retained → Risk: Honeypot (freeze accounts)
- LP tokens in EOA wallet → Risk: Rug pull vulnerability

### Phase 2: Holder Distribution Analysis ✅ COMPLETE
**Files:**
- `services/solana_holder_analyzer.py` - On-chain holder analysis
- Uses `getTokenLargestAccounts` (lightweight, top 20)
- Fallback to `getProgramAccounts` if needed
- 90% reduction in RPC calls vs. original implementation

**Concentration Metrics:**
- Top 1, 5, 10, 20 percentages
- Whale risk detection (top 1 > 20%)
- Centralization flags (top 10 > 60%)

### Phase 3: Enhanced Validation ✅ COMPLETE
**Files:**
- `services/solana_metadata_inspector.py` - On-chain metadata inspection
- `services/price_validator.py` - Cross-source price validation (DexScreener vs Birdeye)
- Enhanced RPC load balancing (3 endpoints)
- Optimized rate limiting with exponential backoff

### Current Features:
✅ Multi-source discovery (DexScreener, Pump.fun API)
✅ Social sentiment integration (X/Twitter via local Ollama LLM)
✅ Risk scoring with detailed breakdowns
✅ Volume spike detection
✅ Token mint inspection
✅ LP token ownership verification
✅ Holder distribution analysis
✅ On-chain metadata inspection
✅ Cross-source price validation
✅ RPC load balancing (3 endpoints)
✅ Comprehensive test suite
✅ Verbose logging system
✅ Systemd service integration

---

## Code Quality Standards

### Python Best Practices:
- Type hints on all function signatures
- Docstrings with parameter and return descriptions
- Error handling with graceful degradation
- Logging with contextual information
- SOLID principles compliance

### File Organization:
```
services/          # Core trading and monitoring services
clients/           # External API clients (DexScreener, Kraken, etc.)
models/            # Data models (dex_models.py, trading models)
utils/             # Utilities (RPC load balancer, helpers)
ui/                # UI components
configs/           # Configuration files for different strategies
tests/             # Test suites (pytest, manual testing)
```

### DEX Hunter Architecture:
```
dex_launch_hunter.py
├── DexScreenerClient
├── LaunchAnnouncementMonitor (Pump.fun)
├── TokenSafetyAnalyzer
│   ├── SolanaMintInspector
│   ├── SolanaLPAnalyzer
│   ├── SolanaHolderAnalyzer
│   ├── SolanaMetadataInspector
│   └── PriceValidator
└── X/Twitter Sentiment Service
```

---

## Development Guidelines

### When Adding Features:

1. **For Solana Token Analysis:**
   - Use `services/solana_rpc_load_balancer.py` for all RPC calls
   - Implement hard red flag checks (auto-reject)
   - Add soft flag warnings (adjusts safety score)
   - Test with real tokens (mint: `65aP2yHMZ6RxZpXn3iHhfBRnzCpwbZeVDTXAoi1gpump`)

2. **For Service Integration:**
   - Register service in `service_control_panel.py`
   - Add `.env` configuration variables
   - Implement logging with context tags (e.g., `[DEX]`, `[WHALE]`, `[X]`)
   - Add graceful error handling

3. **For New RPC Endpoints:**
   - Use `utils/solana_rpc_load_balancer.py`
   - Configure via `.env` variables: `SOLANA_RPC_URL`, `SOLANA_RPC_URL_2`, `SOLANA_RPC_URL_3`
   - Automatic failover on rate limits (429 errors)

### Testing Requirements:

- **Unit Tests**: Individual component testing with pytest
- **Integration Tests**: Full workflow testing
- **Real Token Testing**: Test with actual Solana tokens
- **Rate Limit Testing**: Verify graceful handling of 429 errors
- **Error Scenarios**: Test network failures and API errors

### Logging Standards:

Format: `[TAG] message` where TAG is:
- `[DEX]` - DexScreener scanning
- `[WHALE]` - Smart money tracking
- `[X]` - X/Twitter sentiment
- `[ALERT]` - Alert generation
- `[RPC]` - RPC calls
- `[ERROR]` - Errors

Example:
```python
logger.info("[DEX] ✓ TOKEN: Score=45.5/100, Risk=MEDIUM")
logger.error("[RPC] Rate limit exceeded, switching endpoint")
```

---

## Scoring System

### Composite Score Calculation:
- Pump Potential (0-100): Market metrics
- Velocity Score (0-100): Price momentum
- Safety Score (0-100): Contract safety
- Liquidity Score (0-100): Trading depth
- Social Buzz (0-100): X/Twitter sentiment

### Alert Thresholds:
- ≥ 60: 🔔 HIGH priority alert
- ≥ 70: 🚨 CRITICAL priority alert
- ≥ 30: Gets X/Twitter sentiment check
- < 30: No sentiment check (too low)

### Hard Red Flags:
- Mint authority retained → Score: 0/100 → EXTREME risk → No alert
- Freeze authority retained → Score: 0/100 → EXTREME risk → No alert
- LP in EOA wallet → Score: 0/100 → EXTREME risk → No alert
- Honeypot detected → Blacklisted

---

## Broker Integration

### IBKR Paper Trading (Active)
- **Port**: 7497 (TWS, not Gateway)
- **Config**: `config_paper_trading_ibkr.py`
- **Client ID**: 1
- **Read-only API**: Disabled
- **Test**: `test_ibkr.bat` or manual PYTHONPATH setup

### Supported Brokers:
- IBKR (Interactive Brokers)
- Tradier
- Kraken
- Unified broker adapter in `services/`

---

## Service Management

### Starting Services:
```bash
START_SERVICES.bat           # Start all services
START_DEX_HUNTER.bat        # Start DEX hunter only
START_CRYPTO_AI_TRADER.bat  # Start crypto trader
START_STOCK_MONITOR.bat     # Start stock monitoring
```

### Systemd Services (Linux VPS):
```bash
sudo systemctl start sentient-dex-launch
sudo systemctl status sentient-dex-launch
sudo systemctl restart sentient-dex-launch
tail -f logs/dex_launch_service.log
```

### Environment Variables:
```env
BROKER_TYPE=IBKR
IBKR_PAPER_PORT=7497
IBKR_PAPER_CLIENT_ID=1
SOLANA_RPC_URL=https://solana-mainnet.g.alchemy.com/v2/key1
SOLANA_RPC_URL_2=https://solana-mainnet.g.alchemy.com/v2/key2
SOLANA_RPC_URL_3=https://solana-mainnet.g.alchemy.com/v2/key3
```

---

## Performance Considerations

### RPC Optimization:
- Two-tier holder analysis strategy (getTokenLargestAccounts first)
- 90% reduction in RPC calls
- Multi-endpoint load balancing
- Automatic failover on errors
- Exponential backoff on rate limits (3s, 6s, 12s)

### Caching:
- Seen address tracking (`seen_addresses` set)
- Blacklist enforcement
- Score caching where applicable

### Async Processing:
- Concurrent service runs
- Batch operations (getMultipleAccounts)
- Optimized token discovery polling

---

## Security Focus

### Protected Data:
- Trade data (journaled to `unified_trade_journal.db`)
- API keys (stored in `.env`, never in code)
- Broker credentials (config files, never hardcoded)
- RPC endpoint URLs (configurable)

### Input Validation:
- Address format validation (per-chain)
- Scam name pattern filtering
- Honeypot detection
- Contract safety verification

### Error Handling:
- Graceful degradation on service failures
- Detailed error logging
- No sensitive data in logs

---

## File Structure Reference

### Key Services:
```
services/
├── dex_launch_hunter.py          # Main DEX hunter orchestrator
├── solana_mint_inspector.py      # Phase 1: Mint inspection
├── solana_lp_analyzer.py         # Phase 1: LP analysis
├── solana_holder_analyzer.py     # Phase 2: Holder analysis
├── solana_metadata_inspector.py  # Phase 3: Metadata inspection
├── price_validator.py            # Phase 3: Price validation
├── token_safety_analyzer.py      # Safety scoring orchestrator
├── x_sentiment_service.py        # X/Twitter sentiment
├── launch_announcement_monitor.py # Pump.fun integration
└── [120+ other services]
```

### Configuration:
```
config_*.py                  # Strategy-specific configs
config_paper_trading_ibkr.py # IBKR paper trading setup
```

### Models:
```
models/
└── dex_models.py  # Contains: TokenLaunch, ContractSafety, HolderDistribution, etc.
```

### Utilities:
```
utils/
└── solana_rpc_load_balancer.py  # RPC endpoint management
```

### Tests:
```
tests/
├── test_dex_hunter_complete.py  # Automated pytest suite
├── test_dex_hunter_manual.py    # Interactive testing
└── RUN_DEX_HUNTER_TESTS.bat     # Windows test runner
```

---

## Recent Changes (December 2025)

### ✅ All Phases Complete
- Phase 1: Core Solana RPC integration
- Phase 2: Holder distribution analysis
- Phase 3: Enhanced validation (metadata + price validation)

### ✅ Production Features
- Systemd service integration (running on Linux VPS)
- Verbose logging with context tags
- Comprehensive test suite (all phases passing)
- Real token detection verified (honeypot detection working)

### ✅ Performance Enhancements
- RPC load balancing (3 endpoints)
- Optimized holder analysis (90% reduction in RPC calls)
- Native SOL handling
- Enhanced rate limiting with exponential backoff

---

## Communication

When asking for code help or modifications:
1. **Reference the DEX Hunter Review**: Use `@docs/DEX_HUNTER_REVIEW.md` for context
2. **Specify the service**: Mention which service/file to modify
3. **Include test examples**: Provide token examples for testing
4. **Log examples**: Show expected logging output format
5. **Scope**: Clear about changes (Phase 1/2/3, hard/soft flags, etc.)

---

## Useful Commands

```bash
# Test DEX Hunter
RUN_DEX_HUNTER_TESTS.bat

# View live logs
tail -f logs/dex_launch_service.log

# Check service status
sudo systemctl status sentient-dex-launch

# Stop service
sudo systemctl stop sentient-dex-launch

# View all logs
VIEW_ALL_LOGS.bat

# Launch control panel
python service_control_panel.py
```

---

**Last Updated**: December 1, 2025
**Phases Completed**: 1, 2, 3 ✅
**Status**: Production Ready


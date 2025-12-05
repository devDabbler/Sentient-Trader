"""
Solana Transaction Monitor - Real-time buy/sell order flow tracking

This module provides transaction-level monitoring for Solana tokens, giving you
the same data you see on DexScreener's transactions tab:
- Individual buy/sell transactions
- Buy volume vs sell volume
- Transaction counts (buyers/sellers)
- Whale transaction detection
- Order flow imbalance for entry/exit timing

Data Sources:
- Birdeye API (FREE tier: 100 req/min, txn data included)
- DexScreener (supplement for price/liquidity)
- Helius (PAID: better WebSocket, $49/mo for real-time)

ROI ANALYSIS:
┌─────────────────────────────────────────────────────────────────────┐
│ API Service Comparison for Transaction Monitoring                   │
├─────────────────────┬──────────────┬────────────────────────────────┤
│ Service             │ Cost         │ What You Get                   │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ Birdeye (FREE)      │ $0           │ 100 req/min, txn history,      │
│                     │              │ OHLCV, holder data. SUFFICIENT │
│                     │              │ for 2-5 token monitoring       │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ Birdeye Starter     │ $49/mo       │ 600 req/min, WebSocket,        │
│                     │              │ real-time txns. Worth it if    │
│                     │              │ monitoring 10+ tokens          │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ Helius Free         │ $0           │ 100k credits/mo, basic RPC     │
│                     │              │ NO transaction streaming       │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ Helius Developer    │ $49/mo       │ 2M credits, WebSocket txns,    │
│                     │              │ parsed transactions. BEST for  │
│                     │              │ serious real-time monitoring   │
├─────────────────────┼──────────────┼────────────────────────────────┤
│ DexScreener Paid    │ $49/mo       │ Real-time new pairs, faster    │
│                     │              │ alerts. NOT needed for txns    │
└─────────────────────┴──────────────┴────────────────────────────────┘

RECOMMENDATION: Start with FREE Birdeye (you already have key). 
Upgrade to Helius $49/mo only if you need sub-second alerts for 
many tokens simultaneously.

Author: Sentient Trader
Created: December 2025
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import httpx
from loguru import logger
from collections import deque


class TransactionType(Enum):
    """Transaction type classification"""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class WhaleTier(Enum):
    """Whale classification based on transaction size"""
    MINNOW = "minnow"      # < $100
    FISH = "fish"          # $100 - $1,000
    DOLPHIN = "dolphin"    # $1,000 - $10,000
    WHALE = "whale"        # $10,000 - $100,000
    MEGA_WHALE = "mega"    # > $100,000


@dataclass
class Transaction:
    """Single transaction record"""
    tx_hash: str
    timestamp: datetime
    tx_type: TransactionType
    amount_usd: float
    amount_tokens: float
    price: float
    maker: str  # Wallet address (truncated)
    whale_tier: WhaleTier = WhaleTier.MINNOW
    
    def __post_init__(self):
        # Classify whale tier
        if self.amount_usd >= 100_000:
            self.whale_tier = WhaleTier.MEGA_WHALE
        elif self.amount_usd >= 10_000:
            self.whale_tier = WhaleTier.WHALE
        elif self.amount_usd >= 1_000:
            self.whale_tier = WhaleTier.DOLPHIN
        elif self.amount_usd >= 100:
            self.whale_tier = WhaleTier.FISH
        else:
            self.whale_tier = WhaleTier.MINNOW


@dataclass
class OrderFlowMetrics:
    """
    Real-time order flow metrics for trading decisions.
    
    These are the metrics that help you time entries/exits like a pro.
    """
    # Transaction counts
    total_txns: int = 0
    buy_count: int = 0
    sell_count: int = 0
    
    # Volume
    buy_volume_usd: float = 0.0
    sell_volume_usd: float = 0.0
    net_volume_usd: float = 0.0  # buy - sell (positive = bullish)
    
    # Ratios
    buy_sell_ratio: float = 1.0  # >1 = more buys, <1 = more sells
    volume_imbalance_pct: float = 0.0  # -100 to +100 (bullish/bearish)
    
    # Whale activity
    whale_buys: int = 0
    whale_sells: int = 0
    whale_net_usd: float = 0.0
    
    # Velocity (transactions per minute)
    txn_velocity: float = 0.0
    
    # Time window
    window_seconds: int = 60
    last_update: datetime = field(default_factory=datetime.now)
    
    # Signals
    signal: str = "NEUTRAL"  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    signal_reason: str = ""


@dataclass
class TokenOrderFlow:
    """Complete order flow state for a token"""
    token_address: str
    symbol: str
    
    # Recent transactions (ring buffer, last 100)
    recent_txns: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Metrics for different time windows
    metrics_1m: OrderFlowMetrics = field(default_factory=OrderFlowMetrics)
    metrics_5m: OrderFlowMetrics = field(default_factory=OrderFlowMetrics)
    metrics_15m: OrderFlowMetrics = field(default_factory=OrderFlowMetrics)
    
    # Current price
    current_price: float = 0.0
    price_1m_ago: float = 0.0
    price_5m_ago: float = 0.0
    
    last_update: datetime = field(default_factory=datetime.now)


class SolanaTransactionMonitor:
    """
    Real-time transaction monitoring for Solana tokens.
    
    Provides order flow data similar to DexScreener's transactions tab:
    - Individual buy/sell transactions
    - Buy/sell volume ratios
    - Whale detection
    - Order flow imbalance signals
    
    Usage:
        monitor = SolanaTransactionMonitor()
        flow = await monitor.get_order_flow("token_address")
        print(f"Buy/Sell Ratio: {flow.metrics_1m.buy_sell_ratio}")
        print(f"Signal: {flow.metrics_1m.signal}")
    """
    
    # API endpoints
    BIRDEYE_API_BASE = "https://public-api.birdeye.so"
    DEXSCREENER_API_BASE = "https://api.dexscreener.com"
    
    def __init__(
        self,
        birdeye_api_key: Optional[str] = None,
        helius_api_key: Optional[str] = None,
        cache_ttl_seconds: int = 5  # Cache txn data for 5 seconds
    ):
        """
        Initialize transaction monitor.
        
        Args:
            birdeye_api_key: Birdeye API key (uses env var if not provided)
            helius_api_key: Helius API key for WebSocket (optional, paid feature)
            cache_ttl_seconds: Cache TTL for transaction data
        """
        self.birdeye_api_key = birdeye_api_key or os.getenv("BIRDEYE_API_KEY")
        self.helius_api_key = helius_api_key or os.getenv("HELIUS_API_KEY")
        self.cache_ttl = cache_ttl_seconds
        
        # Token order flow cache
        self._order_flow_cache: Dict[str, TokenOrderFlow] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Rate limiting
        self._last_request_time = 0.0
        self._rate_limit_delay = 0.6  # 100 req/min = 1.67 req/sec, be safe
        
        # Stats
        self.total_requests = 0
        self.successful_requests = 0
        
        if self.birdeye_api_key:
            logger.info("✅ Transaction Monitor initialized with Birdeye API")
        else:
            logger.warning("⚠️ No Birdeye API key - transaction monitoring limited")
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = datetime.now().timestamp() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = datetime.now().timestamp()
    
    async def get_recent_transactions(
        self,
        token_address: str,
        limit: int = 50
    ) -> List[Transaction]:
        """
        Fetch recent transactions for a token from Birdeye.
        
        Args:
            token_address: Solana token mint address
            limit: Number of transactions to fetch (max 100)
            
        Returns:
            List of Transaction objects
        """
        if not self.birdeye_api_key:
            logger.warning("Birdeye API key required for transaction data")
            return []
        
        await self._rate_limit()
        self.total_requests += 1
        
        try:
            url = f"{self.BIRDEYE_API_BASE}/defi/txs/token"
            headers = {
                "X-API-KEY": self.birdeye_api_key,
                "x-chain": "solana"
            }
            params = {
                "address": token_address,
                "tx_type": "swap",  # Only swap transactions
                "limit": min(limit, 100)
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    self.successful_requests += 1
                    
                    transactions = []
                    for tx in data.get("data", {}).get("items", []):
                        try:
                            # Birdeye structure:
                            # - 'side': 'buy' or 'sell' (relative to base token)
                            # - 'from': token being sold
                            # - 'to': token being bought  
                            # - 'quote': the token we're tracking
                            # - 'base': the other token in the pair (usually SOL)
                            
                            side = tx.get("side", "").lower()
                            tx_type = TransactionType.BUY if side == "buy" else TransactionType.SELL
                            
                            # Get the quote token details (the token we're tracking)
                            quote = tx.get("quote", {})
                            base = tx.get("base", {})
                            
                            # Calculate USD value
                            quote_amount = abs(float(quote.get("uiAmount", 0)))
                            quote_price = float(quote.get("price", 0))
                            base_amount = abs(float(base.get("uiAmount", 0)))
                            base_price = float(base.get("price", 0))
                            
                            # USD value is the larger of quote or base value
                            amount_usd = max(quote_amount * quote_price, base_amount * base_price)
                            
                            # Token amount (use quote token)
                            amount_tokens = quote_amount
                            
                            # Price from token
                            price = float(tx.get("tokenPrice", quote_price))
                            
                            txn = Transaction(
                                tx_hash=tx.get("txHash", "")[:16],
                                timestamp=datetime.fromtimestamp(tx.get("blockUnixTime", 0)),
                                tx_type=tx_type,
                                amount_usd=round(amount_usd, 2),
                                amount_tokens=amount_tokens,
                                price=price,
                                maker=tx.get("owner", "")[:8] + "..."
                            )
                            transactions.append(txn)
                        except Exception as e:
                            logger.debug(f"Error parsing transaction: {e}")
                            continue
                    
                    return transactions
                    
                elif response.status_code == 429:
                    logger.warning("Birdeye rate limit hit, backing off...")
                    await asyncio.sleep(5)
                    return []
                else:
                    logger.debug(f"Birdeye returned {response.status_code}: {response.text[:200] if response.text else 'No body'}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return []
    
    def _calculate_metrics(
        self,
        transactions: List[Transaction],
        window_seconds: int
    ) -> OrderFlowMetrics:
        """
        Calculate order flow metrics from transactions.
        
        Args:
            transactions: List of transactions
            window_seconds: Time window in seconds
            
        Returns:
            OrderFlowMetrics with calculated values
        """
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        
        # Filter to time window
        window_txns = [t for t in transactions if t.timestamp >= cutoff]
        
        if not window_txns:
            return OrderFlowMetrics(window_seconds=window_seconds)
        
        # Count buys/sells
        buys = [t for t in window_txns if t.tx_type == TransactionType.BUY]
        sells = [t for t in window_txns if t.tx_type == TransactionType.SELL]
        
        buy_count = len(buys)
        sell_count = len(sells)
        total_txns = len(window_txns)
        
        # Volume
        buy_volume = sum(t.amount_usd for t in buys)
        sell_volume = sum(t.amount_usd for t in sells)
        net_volume = buy_volume - sell_volume
        
        # Ratios
        buy_sell_ratio = buy_volume / max(sell_volume, 0.01)
        total_volume = buy_volume + sell_volume
        volume_imbalance = ((buy_volume - sell_volume) / max(total_volume, 0.01)) * 100
        
        # Whale activity (>$1000)
        whale_buys = len([t for t in buys if t.whale_tier in [WhaleTier.DOLPHIN, WhaleTier.WHALE, WhaleTier.MEGA_WHALE]])
        whale_sells = len([t for t in sells if t.whale_tier in [WhaleTier.DOLPHIN, WhaleTier.WHALE, WhaleTier.MEGA_WHALE]])
        whale_buy_vol = sum(t.amount_usd for t in buys if t.whale_tier in [WhaleTier.DOLPHIN, WhaleTier.WHALE, WhaleTier.MEGA_WHALE])
        whale_sell_vol = sum(t.amount_usd for t in sells if t.whale_tier in [WhaleTier.DOLPHIN, WhaleTier.WHALE, WhaleTier.MEGA_WHALE])
        whale_net = whale_buy_vol - whale_sell_vol
        
        # Velocity
        time_range = max((window_txns[-1].timestamp - window_txns[0].timestamp).total_seconds(), 1)
        txn_velocity = (total_txns / time_range) * 60  # txns per minute
        
        # Generate signal
        signal, reason = self._generate_signal(
            buy_sell_ratio=buy_sell_ratio,
            volume_imbalance=volume_imbalance,
            whale_net=whale_net,
            txn_velocity=txn_velocity,
            buy_count=buy_count,
            sell_count=sell_count
        )
        
        return OrderFlowMetrics(
            total_txns=total_txns,
            buy_count=buy_count,
            sell_count=sell_count,
            buy_volume_usd=round(buy_volume, 2),
            sell_volume_usd=round(sell_volume, 2),
            net_volume_usd=round(net_volume, 2),
            buy_sell_ratio=round(buy_sell_ratio, 2),
            volume_imbalance_pct=round(volume_imbalance, 1),
            whale_buys=whale_buys,
            whale_sells=whale_sells,
            whale_net_usd=round(whale_net, 2),
            txn_velocity=round(txn_velocity, 1),
            window_seconds=window_seconds,
            last_update=datetime.now(),
            signal=signal,
            signal_reason=reason
        )
    
    def _generate_signal(
        self,
        buy_sell_ratio: float,
        volume_imbalance: float,
        whale_net: float,
        txn_velocity: float,
        buy_count: int,
        sell_count: int
    ) -> Tuple[str, str]:
        """
        Generate trading signal based on order flow.
        
        Returns:
            Tuple of (signal, reason)
        """
        reasons = []
        score = 0
        
        # Volume imbalance signal
        if volume_imbalance > 50:
            score += 2
            reasons.append(f"Strong buy pressure ({volume_imbalance:+.0f}%)")
        elif volume_imbalance > 20:
            score += 1
            reasons.append(f"Buy pressure ({volume_imbalance:+.0f}%)")
        elif volume_imbalance < -50:
            score -= 2
            reasons.append(f"Strong sell pressure ({volume_imbalance:+.0f}%)")
        elif volume_imbalance < -20:
            score -= 1
            reasons.append(f"Sell pressure ({volume_imbalance:+.0f}%)")
        
        # Whale activity
        if whale_net > 5000:
            score += 2
            reasons.append(f"Whale accumulation (${whale_net:,.0f} net)")
        elif whale_net > 1000:
            score += 1
            reasons.append(f"Whale buying (${whale_net:,.0f} net)")
        elif whale_net < -5000:
            score -= 2
            reasons.append(f"Whale dumping (${whale_net:,.0f} net)")
        elif whale_net < -1000:
            score -= 1
            reasons.append(f"Whale selling (${whale_net:,.0f} net)")
        
        # Transaction velocity (momentum)
        if txn_velocity > 60:  # >1 txn/sec
            reasons.append(f"High activity ({txn_velocity:.0f} txn/min)")
        
        # Buy/sell count imbalance
        if buy_count > sell_count * 2:
            score += 1
            reasons.append(f"2x more buyers ({buy_count} vs {sell_count})")
        elif sell_count > buy_count * 2:
            score -= 1
            reasons.append(f"2x more sellers ({sell_count} vs {buy_count})")
        
        # Determine signal
        if score >= 3:
            signal = "STRONG_BUY"
        elif score >= 1:
            signal = "BUY"
        elif score <= -3:
            signal = "STRONG_SELL"
        elif score <= -1:
            signal = "SELL"
        else:
            signal = "NEUTRAL"
        
        reason = " | ".join(reasons) if reasons else "No significant signals"
        return signal, reason
    
    async def get_order_flow(
        self,
        token_address: str,
        symbol: str = "TOKEN",
        force_refresh: bool = False
    ) -> TokenOrderFlow:
        """
        Get complete order flow analysis for a token.
        
        This is the main method to call - returns all the data you need
        for timing entries and exits.
        
        Args:
            token_address: Solana token mint address
            symbol: Token symbol (for display)
            force_refresh: Bypass cache
            
        Returns:
            TokenOrderFlow with metrics for 1m, 5m, 15m windows
        """
        # Check cache
        cache_key = token_address.lower()
        if not force_refresh and cache_key in self._order_flow_cache:
            cached = self._order_flow_cache[cache_key]
            cache_time = self._cache_timestamps.get(cache_key, datetime.min)
            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                return cached
        
        # Fetch fresh transactions (Birdeye max is 50 per request)
        transactions = await self.get_recent_transactions(token_address, limit=50)
        
        if not transactions:
            # Return empty flow if no data
            flow = TokenOrderFlow(
                token_address=token_address,
                symbol=symbol
            )
            return flow
        
        # Build order flow
        flow = TokenOrderFlow(
            token_address=token_address,
            symbol=symbol,
            recent_txns=deque(transactions, maxlen=100),
            current_price=transactions[0].price if transactions else 0.0,
            last_update=datetime.now()
        )
        
        # Calculate metrics for different windows
        flow.metrics_1m = self._calculate_metrics(transactions, 60)
        flow.metrics_5m = self._calculate_metrics(transactions, 300)
        flow.metrics_15m = self._calculate_metrics(transactions, 900)
        
        # Cache
        self._order_flow_cache[cache_key] = flow
        self._cache_timestamps[cache_key] = datetime.now()
        
        return flow
    
    def format_order_flow_summary(self, flow: TokenOrderFlow) -> str:
        """
        Format order flow data for Discord/logging.
        
        Returns:
            Formatted string summary
        """
        m1 = flow.metrics_1m
        m5 = flow.metrics_5m
        
        lines = [
            f"📊 **Order Flow: {flow.symbol}**",
            "",
            f"**1-Minute Window:**",
            f"  • Buys: {m1.buy_count} (${m1.buy_volume_usd:,.0f})",
            f"  • Sells: {m1.sell_count} (${m1.sell_volume_usd:,.0f})",
            f"  • Ratio: {m1.buy_sell_ratio:.2f}x",
            f"  • Imbalance: {m1.volume_imbalance_pct:+.1f}%",
            f"  • Signal: **{m1.signal}**",
            f"  • Reason: {m1.signal_reason}",
            "",
            f"**5-Minute Window:**",
            f"  • Buys: {m5.buy_count} (${m5.buy_volume_usd:,.0f})",
            f"  • Sells: {m5.sell_count} (${m5.sell_volume_usd:,.0f})",
            f"  • Whale Net: ${m5.whale_net_usd:+,.0f}",
            f"  • Velocity: {m5.txn_velocity:.0f} txn/min",
            f"  • Signal: **{m5.signal}**",
        ]
        
        return "\n".join(lines)
    
    def get_entry_exit_recommendation(self, flow: TokenOrderFlow) -> Dict:
        """
        Generate entry/exit recommendation based on order flow.
        
        Returns:
            Dict with action, confidence, and reasoning
        """
        m1 = flow.metrics_1m
        m5 = flow.metrics_5m
        
        # Combine signals
        signal_scores = {
            "STRONG_BUY": 2,
            "BUY": 1,
            "NEUTRAL": 0,
            "SELL": -1,
            "STRONG_SELL": -2
        }
        
        combined_score = (
            signal_scores.get(m1.signal, 0) * 2 +  # 1m weighted more
            signal_scores.get(m5.signal, 0)
        )
        
        # Generate recommendation
        if combined_score >= 4:
            action = "ENTER_NOW"
            confidence = 85
            reason = "Strong buying pressure across timeframes"
        elif combined_score >= 2:
            action = "CONSIDER_ENTRY"
            confidence = 65
            reason = "Positive order flow, watch for confirmation"
        elif combined_score <= -4:
            action = "EXIT_NOW"
            confidence = 85
            reason = "Strong selling pressure, protect capital"
        elif combined_score <= -2:
            action = "CONSIDER_EXIT"
            confidence = 65
            reason = "Negative order flow, tighten stops"
        else:
            action = "HOLD"
            confidence = 50
            reason = "Mixed signals, wait for clarity"
        
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "1m_signal": m1.signal,
            "5m_signal": m5.signal,
            "whale_activity": "Accumulating" if m5.whale_net_usd > 0 else "Distributing",
            "velocity": m5.txn_velocity
        }


# Singleton instance
_monitor_instance: Optional[SolanaTransactionMonitor] = None


def get_transaction_monitor() -> SolanaTransactionMonitor:
    """Get or create SolanaTransactionMonitor singleton"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SolanaTransactionMonitor()
    return _monitor_instance


# CLI for testing
async def main():
    """Test the transaction monitor"""
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python solana_transaction_monitor.py <token_address>")
        print("Example: python solana_transaction_monitor.py EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
        return
    
    token_address = sys.argv[1]
    
    # Create fresh instance with loaded env vars
    monitor = SolanaTransactionMonitor()
    
    print(f"\n🔍 Fetching order flow for {token_address[:8]}...")
    print("-" * 60)
    
    flow = await monitor.get_order_flow(token_address, "TEST")
    
    print(monitor.format_order_flow_summary(flow))
    print("-" * 60)
    
    recommendation = monitor.get_entry_exit_recommendation(flow)
    print(f"\n🎯 Recommendation: {recommendation['action']}")
    print(f"   Confidence: {recommendation['confidence']}%")
    print(f"   Reason: {recommendation['reason']}")
    
    print(f"\n📈 Recent Transactions ({len(flow.recent_txns)}):")
    for i, txn in enumerate(list(flow.recent_txns)[:10]):
        emoji = "🟢" if txn.tx_type == TransactionType.BUY else "🔴"
        print(f"   {emoji} {txn.tx_type.value.upper()}: ${txn.amount_usd:.2f} @ ${txn.price:.8f} by {txn.maker}")


if __name__ == "__main__":
    asyncio.run(main())

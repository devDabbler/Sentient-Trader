"""
Signal Memory Service - RAG for trading signals using Supabase pgvector.

Enables: "What happened last time we saw this pattern?"

Supports dual embedding providers:
- OpenAI text-embedding-ada-002 (1536 dims) - Cloud, ~$0.01/1000 signals
- Ollama nomic-embed-text (768 dims) - Local, FREE

Set SIGNAL_MEMORY_EMBEDDING_PROVIDER env var to 'ollama' or 'openai' (default: ollama for cost savings)
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger
import os
import asyncio
import json

# Embedding configuration
EMBEDDING_PROVIDER = os.getenv("SIGNAL_MEMORY_EMBEDDING_PROVIDER", "ollama").lower()

# Embedding dimensions by provider
EMBEDDING_DIMENSIONS = {
    "openai": 1536,  # text-embedding-ada-002
    "ollama": 768,   # nomic-embed-text / mxbai-embed-large
}


@dataclass
class SignalMemory:
    """A remembered trading signal with outcome"""
    id: str
    ticker: str
    strategy: str
    signal_type: str
    outcome: str
    pnl_pct: float
    similarity: float
    context_text: str
    created_at: Optional[str] = None
    confidence: Optional[float] = None


class EmbeddingProvider:
    """Abstract base for embedding providers"""
    
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
    
    def get_embedding(self, text: str) -> List[float]:
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-ada-002 (1536 dimensions)"""
    
    def __init__(self):
        super().__init__(1536)
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return [0.0] * self.dimensions


class OllamaEmbedding(EmbeddingProvider):
    """Ollama local embeddings (768 dimensions with nomic-embed-text)"""
    
    def __init__(self, model: str = "nomic-embed-text"):
        super().__init__(768)
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                if len(embedding) != self.dimensions:
                    logger.warning(f"Unexpected embedding dimensions: {len(embedding)} (expected {self.dimensions})")
                return embedding
            else:
                logger.error(f"Ollama embedding failed: {response.status_code} - {response.text}")
                return [0.0] * self.dimensions
                
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            return [0.0] * self.dimensions


def get_embedding_provider() -> EmbeddingProvider:
    """Factory to get the configured embedding provider"""
    if EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbedding()
    else:
        # Default to Ollama for cost savings
        return OllamaEmbedding()


class SignalMemoryService:
    """
    Store and retrieve trading signals with vector similarity search.
    Enables: "What happened last time we saw this pattern?"
    """
    
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client
        self._embedding_provider = None
        self._initialized = False
        
    @property
    def embedding_provider(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
            logger.info(f"📊 Signal Memory using {EMBEDDING_PROVIDER.upper()} embeddings ({self._embedding_provider.dimensions} dims)")
        return self._embedding_provider
    
    @property
    def dimensions(self) -> int:
        return self.embedding_provider.dimensions
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using configured provider"""
        return self.embedding_provider.get_embedding(text)
    
    def _build_context_text(
        self,
        ticker: str,
        strategy: str,
        signal_type: str,
        price: float,
        rsi: float,
        macd: float,
        vix: float,
        market_regime: str,
        volume: Optional[int] = None,
        confidence: Optional[float] = None
    ) -> str:
        """Build searchable context string for embedding"""
        parts = [
            f"{ticker} {strategy} {signal_type} signal.",
            f"Price ${price:.2f}, RSI {rsi:.1f}, MACD {macd:.4f}.",
            f"VIX {vix:.1f}, Market regime: {market_regime}."
        ]
        
        if volume:
            parts.append(f"Volume: {volume:,}.")
        if confidence:
            parts.append(f"Confidence: {confidence:.1f}%.")
            
        return " ".join(parts)
    
    async def store_signal(
        self,
        ticker: str,
        strategy: str,
        signal_type: str,
        confidence: float,
        price: float,
        volume: int,
        rsi: float,
        macd_histogram: float,
        vix: float,
        market_regime: str,
        trade_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Store a new trading signal with embedding.
        
        Returns:
            Signal ID if successful, None otherwise
        """
        if not self.supabase:
            logger.warning("Supabase client not configured, signal not stored")
            return None
            
        try:
            context_text = self._build_context_text(
                ticker, strategy, signal_type, price, rsi, macd_histogram, vix, market_regime,
                volume=volume, confidence=confidence
            )
            
            embedding = self._get_embedding(context_text)
            
            data = {
                "ticker": ticker,
                "strategy": strategy,
                "signal_type": signal_type,
                "confidence": confidence,
                "price": price,
                "volume": volume,
                "rsi": rsi,
                "macd_histogram": macd_histogram,
                "vix": vix,
                "market_regime": market_regime,
                "outcome": "PENDING",
                "embedding": embedding,
                "context_text": context_text
            }
            
            if trade_id:
                data["trade_id"] = trade_id
            
            result = self.supabase.table("trade_signals_memory").insert(data).execute()
            
            if result.data:
                signal_id = result.data[0]["id"]
                logger.info(f"📝 Stored signal memory: {ticker} {signal_type} (ID: {signal_id[:8]}...)")
                return signal_id
            else:
                logger.error(f"Failed to store signal: No data returned")
                return None
                
        except Exception as e:
            logger.error(f"Error storing signal memory: {e}")
            return None
    
    async def update_outcome(
        self,
        signal_id: str,
        outcome: str,
        pnl_pct: float,
        holding_period_hours: int
    ) -> bool:
        """Update signal with actual outcome after trade closes"""
        if not self.supabase:
            return False
            
        try:
            self.supabase.table("trade_signals_memory").update({
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "holding_period_hours": holding_period_hours
            }).eq("id", signal_id).execute()
            
            emoji = "✅" if outcome == "WIN" else "❌"
            logger.info(f"{emoji} Updated signal outcome: {outcome} ({pnl_pct:+.2f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal outcome: {e}")
            return False
    
    async def update_outcome_by_trade_id(
        self,
        trade_id: str,
        outcome: str,
        pnl_pct: float,
        holding_period_hours: int
    ) -> bool:
        """Update signal outcome using trade_id lookup"""
        if not self.supabase:
            return False
            
        try:
            self.supabase.table("trade_signals_memory").update({
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "holding_period_hours": holding_period_hours
            }).eq("trade_id", trade_id).execute()
            
            emoji = "✅" if outcome == "WIN" else "❌"
            logger.info(f"{emoji} Updated signal by trade_id {trade_id}: {outcome} ({pnl_pct:+.2f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal by trade_id: {e}")
            return False
    
    async def find_similar_signals(
        self,
        ticker: str,
        strategy: str,
        signal_type: str,
        price: float,
        rsi: float,
        macd: float,
        vix: float,
        market_regime: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[SignalMemory]:
        """Find historically similar signals and their outcomes"""
        if not self.supabase:
            return []
            
        try:
            context_text = self._build_context_text(
                ticker, strategy, signal_type, price, rsi, macd, vix, market_regime
            )
            
            embedding = self._get_embedding(context_text)
            
            # Call the match_signals function (RPC)
            result = self.supabase.rpc("match_signals", {
                "query_embedding": embedding,
                "match_threshold": min_similarity,
                "match_count": limit
            }).execute()
            
            signals = []
            for row in result.data:
                signals.append(SignalMemory(
                    id=row["id"],
                    ticker=row["ticker"],
                    strategy=row["strategy"],
                    signal_type=row["signal_type"],
                    outcome=row["outcome"],
                    pnl_pct=row.get("pnl_pct") or 0,
                    similarity=row["similarity"],
                    context_text="",
                    created_at=row.get("created_at"),
                    confidence=row.get("confidence")
                ))
            
            if signals:
                logger.debug(f"🔍 Found {len(signals)} similar signals for {ticker}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error finding similar signals: {e}")
            return []
    
    async def get_historical_win_rate(
        self,
        ticker: str,
        strategy: str,
        signal_type: str,
        price: float,
        rsi: float,
        macd: float,
        vix: float,
        market_regime: str,
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Get win rate for similar historical signals.
        
        Returns:
            {
                "win_rate": float (0-1) or None,
                "avg_pnl": float or None,
                "sample_size": int,
                "avg_similarity": float or None,
                "recommendation": str,  # 'BOOST', 'REDUCE', 'NEUTRAL', 'INSUFFICIENT_DATA'
                "confidence_adjustment": float  # Multiplier for signal confidence
            }
        """
        similar = await self.find_similar_signals(
            ticker, strategy, signal_type, price, rsi, macd, vix, market_regime, 
            limit=20, min_similarity=0.65
        )
        
        if not similar:
            return {
                "win_rate": None, 
                "avg_pnl": None, 
                "sample_size": 0,
                "avg_similarity": None,
                "recommendation": "INSUFFICIENT_DATA",
                "confidence_adjustment": 1.0
            }
        
        completed = [s for s in similar if s.outcome in ("WIN", "LOSS")]
        
        if len(completed) < min_samples:
            return {
                "win_rate": None, 
                "avg_pnl": None, 
                "sample_size": len(completed),
                "avg_similarity": sum(s.similarity for s in similar) / len(similar),
                "recommendation": "INSUFFICIENT_DATA",
                "confidence_adjustment": 1.0
            }
        
        wins = sum(1 for s in completed if s.outcome == "WIN")
        win_rate = wins / len(completed)
        avg_pnl = sum(s.pnl_pct for s in completed) / len(completed)
        avg_similarity = sum(s.similarity for s in completed) / len(completed)
        
        # Determine recommendation and adjustment
        if win_rate > 0.65 and avg_pnl > 0.5:
            recommendation = "BOOST"
            confidence_adjustment = 1.15  # +15% confidence
        elif win_rate < 0.40 or avg_pnl < -0.5:
            recommendation = "REDUCE"
            confidence_adjustment = 0.75  # -25% confidence
        else:
            recommendation = "NEUTRAL"
            confidence_adjustment = 1.0
        
        result = {
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "sample_size": len(completed),
            "avg_similarity": avg_similarity,
            "recommendation": recommendation,
            "confidence_adjustment": confidence_adjustment
        }
        
        if recommendation != "NEUTRAL":
            logger.info(f"📊 Historical analysis for {ticker}: Win rate {win_rate:.1%}, Avg P&L {avg_pnl:+.2f}% → {recommendation}")
        
        return result
    
    async def get_ticker_performance(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """Get overall performance stats for a specific ticker"""
        if not self.supabase:
            return {}
            
        try:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            result = self.supabase.table("trade_signals_memory")\
                .select("outcome, pnl_pct, strategy, signal_type")\
                .eq("ticker", ticker)\
                .gte("created_at", cutoff)\
                .in_("outcome", ["WIN", "LOSS"])\
                .execute()
            
            if not result.data:
                return {"ticker": ticker, "total_trades": 0}
            
            wins = sum(1 for r in result.data if r["outcome"] == "WIN")
            losses = sum(1 for r in result.data if r["outcome"] == "LOSS")
            total_pnl = sum(r.get("pnl_pct", 0) for r in result.data)
            
            return {
                "ticker": ticker,
                "total_trades": len(result.data),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(result.data) if result.data else 0,
                "total_pnl_pct": total_pnl,
                "avg_pnl_pct": total_pnl / len(result.data) if result.data else 0,
                "days": days
            }
            
        except Exception as e:
            logger.error(f"Error getting ticker performance: {e}")
            return {}
    
    async def get_strategy_performance(self, strategy: str, days: int = 30) -> Dict[str, Any]:
        """Get performance stats for a specific strategy"""
        if not self.supabase:
            return {}
            
        try:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            result = self.supabase.table("trade_signals_memory")\
                .select("ticker, outcome, pnl_pct, signal_type")\
                .eq("strategy", strategy)\
                .gte("created_at", cutoff)\
                .in_("outcome", ["WIN", "LOSS"])\
                .execute()
            
            if not result.data:
                return {"strategy": strategy, "total_trades": 0}
            
            wins = sum(1 for r in result.data if r["outcome"] == "WIN")
            total_pnl = sum(r.get("pnl_pct", 0) for r in result.data)
            unique_tickers = set(r["ticker"] for r in result.data)
            
            return {
                "strategy": strategy,
                "total_trades": len(result.data),
                "wins": wins,
                "losses": len(result.data) - wins,
                "win_rate": wins / len(result.data) if result.data else 0,
                "total_pnl_pct": total_pnl,
                "avg_pnl_pct": total_pnl / len(result.data) if result.data else 0,
                "unique_tickers": len(unique_tickers),
                "days": days
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return {}


# Singleton instance
_service: Optional[SignalMemoryService] = None


def get_signal_memory_service(supabase_client=None) -> Optional[SignalMemoryService]:
    """
    Get or create the SignalMemoryService singleton.
    
    Args:
        supabase_client: Supabase client instance (required on first call)
        
    Returns:
        SignalMemoryService instance or None if not configured
    """
    global _service
    
    if _service is None:
        if supabase_client is None:
            # Try to get supabase client
            try:
                from clients.supabase_client import get_supabase_client
                supabase_client = get_supabase_client()
            except Exception as e:
                logger.debug(f"Could not auto-initialize Supabase client: {e}")
                return None
        
        if supabase_client:
            _service = SignalMemoryService(supabase_client)
            logger.info("🧠 Signal Memory Service initialized")
        
    return _service


def reset_signal_memory_service():
    """Reset the singleton (for testing)"""
    global _service
    _service = None


# Async helper for sync contexts
def store_signal_sync(**kwargs) -> Optional[str]:
    """Synchronous wrapper for store_signal"""
    service = get_signal_memory_service()
    if service:
        try:
            # Try to get existing loop
            try:
                loop = asyncio.get_running_loop()
                # Loop is running, use thread executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, service.store_signal(**kwargs))
                    return future.result()
            except RuntimeError:
                # No running loop, create one
                return asyncio.run(service.store_signal(**kwargs))
        except Exception as e:
            logger.debug(f"store_signal_sync failed: {e}")
            return None
    return None


def get_historical_win_rate_sync(**kwargs) -> Dict[str, Any]:
    """Synchronous wrapper for get_historical_win_rate"""
    service = get_signal_memory_service()
    if service:
        try:
            # Try to get existing loop
            try:
                loop = asyncio.get_running_loop()
                # Loop is running, use thread executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, service.get_historical_win_rate(**kwargs))
                    return future.result()
            except RuntimeError:
                # No running loop, create one
                return asyncio.run(service.get_historical_win_rate(**kwargs))
        except Exception as e:
            logger.debug(f"get_historical_win_rate_sync failed: {e}")
    return {"recommendation": "INSUFFICIENT_DATA", "confidence_adjustment": 1.0, "sample_size": 0}


# ============================================================
# CRYPTO-SPECIFIC HELPERS
# ============================================================

def get_crypto_historical_performance(
    symbol: str,
    strategy: str,
    signal_type: str,
    price: float,
    rsi: float = 50.0,
    volume_ratio: float = 1.0,
    change_24h: float = 0.0
) -> Dict[str, Any]:
    """
    Get historical performance for crypto signals - simplified interface.
    
    Args:
        symbol: Crypto pair (e.g., 'BTC/USD')
        strategy: Strategy name (e.g., 'BREAKOUT', 'BUZZING')
        signal_type: 'BUY' or 'SELL'
        price: Current price
        rsi: RSI value (default 50)
        volume_ratio: Volume vs average (default 1.0)
        change_24h: 24h price change percentage
        
    Returns:
        {
            "win_rate": float or None,
            "avg_pnl": float or None,
            "sample_size": int,
            "recommendation": str,  # 'BOOST', 'REDUCE', 'NEUTRAL', 'INSUFFICIENT_DATA'
            "confidence_adjustment": float,
            "similar_patterns": int
        }
    """
    # Map 24h change to approximate MACD histogram
    macd_estimate = change_24h / 10.0  # Rough approximation
    
    # Crypto markets are 24/7, use a default "moderate" VIX equivalent
    crypto_volatility = min(40, max(10, abs(change_24h) * 2 + 15))
    
    # Determine market regime from price action
    if change_24h > 5:
        market_regime = 'BULLISH'
    elif change_24h < -5:
        market_regime = 'BEARISH'
    elif volume_ratio > 2:
        market_regime = 'VOLATILE'
    else:
        market_regime = 'NEUTRAL'
    
    result = get_historical_win_rate_sync(
        ticker=symbol,
        strategy=f"CRYPTO_{strategy}",  # Prefix to separate from stock strategies
        signal_type=signal_type,
        price=price,
        rsi=rsi,
        macd=macd_estimate,
        vix=crypto_volatility,
        market_regime=market_regime
    )
    
    result['similar_patterns'] = result.get('sample_size', 0)
    return result


def store_crypto_signal(
    symbol: str,
    strategy: str,
    signal_type: str,
    confidence: float,
    price: float,
    rsi: float = 50.0,
    volume_ratio: float = 1.0,
    change_24h: float = 0.0,
    trade_id: Optional[str] = None
) -> Optional[str]:
    """
    Store a crypto trading signal for future pattern matching.
    
    Args:
        symbol: Crypto pair (e.g., 'BTC/USD')
        strategy: Strategy name (e.g., 'BREAKOUT', 'BUZZING')
        signal_type: 'BUY' or 'SELL'
        confidence: Signal confidence (0-100)
        price: Entry price
        rsi: RSI value
        volume_ratio: Volume vs average
        change_24h: 24h price change percentage
        trade_id: Optional trade ID for outcome tracking
        
    Returns:
        Signal ID if stored successfully, None otherwise
    """
    service = get_signal_memory_service()
    if not service:
        return None
    
    try:
        # Map crypto metrics to signal memory format
        macd_estimate = change_24h / 10.0
        crypto_volatility = min(40, max(10, abs(change_24h) * 2 + 15))
        
        if change_24h > 5:
            market_regime = 'BULLISH'
        elif change_24h < -5:
            market_regime = 'BEARISH'
        elif volume_ratio > 2:
            market_regime = 'VOLATILE'
        else:
            market_regime = 'NEUTRAL'
        
        # Use sync wrapper pattern
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    service.store_signal(
                        ticker=symbol,
                        strategy=f"CRYPTO_{strategy}",
                        signal_type=signal_type,
                        confidence=confidence,
                        price=price,
                        volume=int(volume_ratio * 1000),  # Store as relative volume
                        rsi=rsi,
                        macd_histogram=macd_estimate,
                        vix=crypto_volatility,
                        market_regime=market_regime,
                        trade_id=trade_id
                    )
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                service.store_signal(
                    ticker=symbol,
                    strategy=f"CRYPTO_{strategy}",
                    signal_type=signal_type,
                    confidence=confidence,
                    price=price,
                    volume=int(volume_ratio * 1000),
                    rsi=rsi,
                    macd_histogram=macd_estimate,
                    vix=crypto_volatility,
                    market_regime=market_regime,
                    trade_id=trade_id
                )
            )
    except Exception as e:
        logger.debug(f"store_crypto_signal failed: {e}")
        return None


def update_crypto_signal_outcome(
    trade_id: str,
    outcome: str,
    pnl_pct: float,
    holding_hours: int = 0
) -> bool:
    """
    Update a crypto signal with its outcome after trade closes.
    
    Args:
        trade_id: Trade ID used when storing the signal
        outcome: 'WIN' or 'LOSS'
        pnl_pct: Profit/loss percentage
        holding_hours: Hours position was held
        
    Returns:
        True if updated successfully
    """
    service = get_signal_memory_service()
    if not service:
        return False
    
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    service.update_outcome_by_trade_id(
                        trade_id=trade_id,
                        outcome=outcome,
                        pnl_pct=pnl_pct,
                        holding_period_hours=holding_hours
                    )
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                service.update_outcome_by_trade_id(
                    trade_id=trade_id,
                    outcome=outcome,
                    pnl_pct=pnl_pct,
                    holding_period_hours=holding_hours
                )
            )
    except Exception as e:
        logger.debug(f"update_crypto_signal_outcome failed: {e}")
        return False

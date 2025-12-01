"""
Jupiter Aggregator API Client
Provides real-time routing and pricing for Solana DEX swaps
"""

import httpx
import asyncio
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum


class JupiterPricingError(Exception):
    """Jupiter pricing error"""
    pass


@dataclass
class JupiterQuote:
    """Jupiter swap quote data"""
    input_token: str
    output_token: str
    input_amount: float
    output_amount: float
    route_count: int
    price_impact_pct: float
    execution_price: float
    fetched_at: datetime
    
    def __repr__(self):
        return (f"JupiterQuote(input={self.input_token[:4]}...{self.input_token[-4:]}, "
                f"output={self.output_token[:4]}...{self.output_token[-4:]}, "
                f"output_amount={self.output_amount:.2f}, impact={self.price_impact_pct:.2f}%)")


class JupiterClient:
    """
    Jupiter Aggregator v6 API client for Solana DEX routing and pricing
    
    Features:
    - Real-time swap quotes across all Solana DEXs
    - Price impact calculation
    - Liquidity depth analysis
    - Token routing optimization
    """
    
    BASE_URL = "https://api.jup.ag/price"
    QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
    
    def __init__(self, cache_ttl_seconds: int = 60):
        """
        Initialize Jupiter client
        
        Args:
            cache_ttl_seconds: Quote cache TTL (avoid rate limits)
        """
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.quote_cache: Dict[str, Tuple[JupiterQuote, datetime]] = {}
        self.rate_limit_delay = 0.2  # 200ms between requests
        self.last_request_time = 0.0
        
    async def get_price(
        self,
        mint_id: str,
        vs_token: str = "EPjFWaJPg5w7zJ7Y5aQUNjpWN9BABdZc8m5DMXBPfHXo"  # Default: USDC
    ) -> Optional[float]:
        """
        Get current token price in USDC (or specified token)
        
        Args:
            mint_id: Token mint address on Solana
            vs_token: Reference token for pricing (default USDC)
            
        Returns:
            Price as float or None if failed
        """
        try:
            # Rate limiting
            await self._apply_rate_limit()
            
            params = {
                "ids": mint_id,
                "vsToken": vs_token
            }
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if "data" in data and mint_id in data["data"]:
                    price_info = data["data"][mint_id]
                    return price_info.get("price", None)
                    
                return None
                
        except Exception as e:
            logger.warning(f"[JUPITER] Price fetch failed for {mint_id[:8]}...: {e}")
            return None
    
    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount_in: int,  # In smallest unit (lamports for SOL, etc.)
        slippage_bps: int = 50  # 0.5% default
    ) -> Optional[JupiterQuote]:
        """
        Get swap quote from Jupiter (input amount → output amount)
        
        Args:
            input_mint: Input token mint
            output_mint: Output token mint
            amount_in: Amount in smallest unit (lamports)
            slippage_bps: Slippage in basis points (50 = 0.5%)
            
        Returns:
            JupiterQuote or None if failed
        """
        cache_key = f"{input_mint}_{output_mint}_{amount_in}"
        
        # Check cache
        if cache_key in self.quote_cache:
            quote, cached_at = self.quote_cache[cache_key]
            if datetime.now() - cached_at < self.cache_ttl:
                logger.debug(f"[JUPITER] Using cached quote for {input_mint[:8]}→{output_mint[:8]}")
                return quote
        
        try:
            await self._apply_rate_limit()
            
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount_in,
                "slippageBps": slippage_bps
            }
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(self.QUOTE_URL, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                quote = JupiterQuote(
                    input_token=input_mint,
                    output_token=output_mint,
                    input_amount=amount_in,
                    output_amount=float(data.get("outAmount", 0)),
                    route_count=len(data.get("routePlan", [])),
                    price_impact_pct=float(data.get("priceImpactPct", 0)) * 100,
                    execution_price=float(data.get("outAmount", 1)) / float(amount_in) if amount_in > 0 else 0,
                    fetched_at=datetime.now()
                )
                
                # Cache it
                self.quote_cache[cache_key] = (quote, datetime.now())
                
                logger.debug(f"[JUPITER] Quote: {quote}")
                return quote
                
        except Exception as e:
            logger.warning(f"[JUPITER] Quote failed for {input_mint[:8]}→{output_mint[:8]}: {e}")
            return None
    
    async def check_price_spread(
        self,
        token_mint: str,
        reference_price: float,
        reference_source: str = "Kraken"
    ) -> Optional[Dict]:
        """
        Check if Jupiter prices differ from reference (validates liquidity & arbitrage potential)
        
        Args:
            token_mint: Token mint address
            reference_price: Reference price (e.g., from Kraken)
            reference_source: Source of reference price (for logging)
            
        Returns:
            Dict with spread info:
            {
                'jupiter_price': float,
                'reference_price': float,
                'spread_pct': float,
                'spread_direction': 'HIGHER'|'LOWER'|'EQUAL',
                'arbitrage_opportunity': bool,
                'validation_status': 'CONFIRMED'|'MISMATCH'|'ERROR'
            }
        """
        try:
            # Get Jupiter price (vs USDC)
            jupiter_price = await self.get_price(token_mint)
            
            if jupiter_price is None:
                logger.warning(f"[JUPITER] Could not fetch price for {token_mint[:8]}...")
                return None
            
            # Calculate spread
            spread_pct = ((jupiter_price - reference_price) / reference_price) * 100
            
            # Determine direction
            if abs(spread_pct) < 0.5:
                direction = "EQUAL"
            elif spread_pct > 0:
                direction = "HIGHER"
            else:
                direction = "LOWER"
            
            # Flag arbitrage opportunities (>1% spread)
            arbitrage_opportunity = abs(spread_pct) > 1.0
            
            result = {
                'jupiter_price': jupiter_price,
                'reference_price': reference_price,
                'spread_pct': spread_pct,
                'spread_direction': direction,
                'arbitrage_opportunity': arbitrage_opportunity,
                'validation_status': 'CONFIRMED' if abs(spread_pct) < 5.0 else 'MISMATCH'
            }
            
            if arbitrage_opportunity:
                logger.info(
                    f"[JUPITER] ⚡ Arbitrage opportunity detected: "
                    f"{token_mint[:8]}... spread={spread_pct:+.2f}% "
                    f"(Jupiter={jupiter_price:.6f} vs {reference_source}={reference_price:.6f})"
                )
            
            return result
            
        except Exception as e:
            logger.warning(f"[JUPITER] Spread check failed: {e}")
            return None
    
    async def get_liquidity_depth(
        self,
        token_mint: str,
        depth_levels: List[float] = [0.1, 0.5, 1.0]  # USD amounts
    ) -> Optional[Dict]:
        """
        Estimate liquidity depth at various price levels
        
        Args:
            token_mint: Token mint address
            depth_levels: USD amounts to check liquidity for
            
        Returns:
            Dict with liquidity depth info or None
        """
        try:
            results = {}
            
            for depth_usd in depth_levels:
                # Use USDC as reference, convert USD to USDC base units (1 USDC = 1,000,000 lamports)
                amount_in_base = int(depth_usd * 1_000_000)
                
                quote = await self.get_quote(
                    input_mint="EPjFWaJPg5w7zJ7Y5aQUNjpWN9BABdZc8m5DMXBPfHXo",  # USDC
                    output_mint=token_mint,
                    amount_in=amount_in_base,
                    slippage_bps=50
                )
                
                if quote:
                    results[depth_usd] = {
                        'usd_in': depth_usd,
                        'tokens_out': quote.output_amount,
                        'price_impact': quote.price_impact_pct,
                        'routes_available': quote.route_count
                    }
            
            if results:
                logger.debug(f"[JUPITER] Liquidity depth for {token_mint[:8]}...: {results}")
            
            return results if results else None
            
        except Exception as e:
            logger.warning(f"[JUPITER] Liquidity depth check failed: {e}")
            return None
    
    async def _apply_rate_limit(self):
        """Respect Jupiter API rate limits"""
        elapsed = datetime.now().timestamp() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = datetime.now().timestamp()
    
    def clear_cache(self):
        """Clear quote cache"""
        self.quote_cache.clear()
        logger.debug("[JUPITER] Quote cache cleared")


# Singleton instance
_jupiter_client: Optional[JupiterClient] = None


def get_jupiter_client() -> JupiterClient:
    """Get or create Jupiter client singleton"""
    global _jupiter_client
    if _jupiter_client is None:
        _jupiter_client = JupiterClient()
    return _jupiter_client


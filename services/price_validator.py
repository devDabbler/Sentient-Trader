"""
Price Validator

Validates token price data across multiple sources to detect:
1. Price inconsistencies (potential manipulation or API errors)
2. Liquidity discrepancies
3. Volume discrepancies
4. Cross-source verification

This helps catch:
- API indexing delays
- Manipulated price data
- Inconsistent market data
- Potential scams with fake liquidity
"""

import os
import asyncio
import httpx
from typing import Dict, Optional, Tuple, List
from loguru import logger
from dotenv import load_dotenv
from models.dex_models import Chain

load_dotenv()

# API endpoints
BIRDEYE_API_BASE = "https://public-api.birdeye.so"
DEXSCREENER_API_BASE = "https://api.dexscreener.com/latest"


class PriceValidator:
    """Validate price data across multiple sources"""
    
    def __init__(self):
        self.birdeye_api_key = os.getenv("BIRDEYE_API_KEY")
        self.last_request_time = {}
        self.rate_limit_delays = {
            "birdeye": 1.0,  # 1 second between requests
            "dexscreener": 0.5
        }
        
        if self.birdeye_api_key:
            logger.info("✅ Price Validator initialized with Birdeye API key")
        else:
            logger.warning("⚠️ Birdeye API key not found - cross-source validation limited")
    
    async def validate_price_data(
        self,
        contract_address: str,
        chain: Chain,
        dexscreener_price: float,
        dexscreener_liquidity: float,
        dexscreener_volume_24h: float
    ) -> Dict:
        """
        Validate price data by comparing DexScreener with other sources
        
        Args:
            contract_address: Token contract address
            chain: Blockchain network
            dexscreener_price: Price from DexScreener (USD)
            dexscreener_liquidity: Liquidity from DexScreener (USD)
            dexscreener_volume_24h: 24h volume from DexScreener (USD)
            
        Returns:
            {
                'is_consistent': bool,
                'consistency_score': float,  # 0-100, higher = more consistent
                'price_diff_pct': float,  # % difference in price
                'liquidity_diff_pct': float,  # % difference in liquidity
                'volume_diff_pct': float,  # % difference in volume
                'warnings': List[str],
                'errors': List[str],
                'source_data': Dict  # Data from other sources
            }
        """
        warnings = []
        errors = []
        source_data = {}
        
        # For Solana tokens, compare with Birdeye
        if chain == Chain.SOLANA and self.birdeye_api_key:
            birdeye_data = await self._get_birdeye_data(contract_address)
            
            if birdeye_data and not birdeye_data.get('error'):
                source_data['birdeye'] = birdeye_data
                
                # Compare prices
                birdeye_price = birdeye_data.get('price_usd', 0)
                if birdeye_price > 0 and dexscreener_price > 0:
                    price_diff_pct = abs(birdeye_price - dexscreener_price) / max(birdeye_price, dexscreener_price) * 100
                    
                    if price_diff_pct > 20:  # >20% difference
                        warnings.append(
                            f"⚠️ Price inconsistency: DexScreener=${dexscreener_price:.8f} vs "
                            f"Birdeye=${birdeye_price:.8f} ({price_diff_pct:.1f}% diff)"
                        )
                    elif price_diff_pct > 10:  # 10-20% difference
                        warnings.append(
                            f"⚠️ Moderate price difference: {price_diff_pct:.1f}% between sources"
                        )
                else:
                    price_diff_pct = 0.0
                
                # Compare liquidity
                birdeye_liquidity = birdeye_data.get('liquidity_usd', 0)
                if birdeye_liquidity > 0 and dexscreener_liquidity > 0:
                    liquidity_diff_pct = abs(birdeye_liquidity - dexscreener_liquidity) / max(birdeye_liquidity, dexscreener_liquidity) * 100
                    
                    if liquidity_diff_pct > 50:  # >50% difference
                        warnings.append(
                            f"🚨 Major liquidity inconsistency: DexScreener=${dexscreener_liquidity:,.0f} vs "
                            f"Birdeye=${birdeye_liquidity:,.0f} ({liquidity_diff_pct:.0f}% diff) - Potential data manipulation!"
                        )
                    elif liquidity_diff_pct > 30:  # 30-50% difference
                        warnings.append(
                            f"⚠️ Liquidity difference: {liquidity_diff_pct:.0f}% between sources"
                        )
                else:
                    liquidity_diff_pct = 0.0
                
                # Compare volume
                birdeye_volume = birdeye_data.get('volume_24h', 0)
                if birdeye_volume > 0 and dexscreener_volume_24h > 0:
                    volume_diff_pct = abs(birdeye_volume - dexscreener_volume_24h) / max(birdeye_volume, dexscreener_volume_24h) * 100
                    
                    if volume_diff_pct > 50:  # >50% difference
                        warnings.append(
                            f"⚠️ Volume inconsistency: DexScreener=${dexscreener_volume_24h:,.0f} vs "
                            f"Birdeye=${birdeye_volume:,.0f} ({volume_diff_pct:.0f}% diff)"
                        )
                else:
                    volume_diff_pct = 0.0
                
                # Calculate consistency score (0-100)
                consistency_score = 100.0
                if price_diff_pct > 0:
                    consistency_score -= min(price_diff_pct * 2, 40)  # Max 40 point penalty
                if liquidity_diff_pct > 0:
                    consistency_score -= min(liquidity_diff_pct, 40)  # Max 40 point penalty
                if volume_diff_pct > 0:
                    consistency_score -= min(volume_diff_pct / 2, 20)  # Max 20 point penalty
                
                consistency_score = max(0, consistency_score)
                
                # Determine if consistent (score >= 70 and no major warnings)
                is_consistent = consistency_score >= 70 and liquidity_diff_pct < 50
                
                return {
                    'is_consistent': is_consistent,
                    'consistency_score': consistency_score,
                    'price_diff_pct': price_diff_pct if birdeye_price > 0 else 0.0,
                    'liquidity_diff_pct': liquidity_diff_pct if birdeye_liquidity > 0 else 0.0,
                    'volume_diff_pct': volume_diff_pct if birdeye_volume > 0 else 0.0,
                    'warnings': warnings,
                    'errors': errors,
                    'source_data': source_data
                }
            else:
                errors.append(f"Birdeye API error: {birdeye_data.get('error', 'Unknown error') if birdeye_data else 'No response'}")
        
        # For non-Solana or if Birdeye unavailable, return neutral result
        return {
            'is_consistent': True,  # Assume consistent if we can't verify
            'consistency_score': 100.0,
            'price_diff_pct': 0.0,
            'liquidity_diff_pct': 0.0,
            'volume_diff_pct': 0.0,
            'warnings': warnings,
            'errors': errors,
            'source_data': source_data
        }
    
    async def _get_birdeye_data(self, contract_address: str) -> Dict:
        """Get token data from Birdeye API"""
        if not self.birdeye_api_key:
            return {'error': 'Birdeye API key not configured'}
        
        try:
            await self._rate_limit("birdeye")
            
            headers = {
                "X-API-KEY": self.birdeye_api_key
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get token overview (price, liquidity, volume)
                response = await client.get(
                    f"{BIRDEYE_API_BASE}/defi/token_overview",
                    headers=headers,
                    params={
                        "address": contract_address
                    }
                )
                
                if response.status_code != 200:
                    logger.debug(f"Birdeye API error: {response.status_code}")
                    return {'error': f'Birdeye API returned {response.status_code}'}
                
                data = response.json()
                
                if data.get('success') is False:
                    error_msg = data.get('message', 'Unknown error')
                    logger.debug(f"Birdeye API error: {error_msg}")
                    return {'error': error_msg}
                
                result = data.get('data', {})
                
                # Extract relevant fields
                price_usd = result.get('price', 0)
                liquidity_usd = result.get('liquidity', 0)
                volume_24h = result.get('volume24hUSD', 0)
                
                return {
                    'price_usd': price_usd,
                    'liquidity_usd': liquidity_usd,
                    'volume_24h': volume_24h,
                    'market_cap': result.get('mc', 0),
                    'price_change_24h': result.get('priceChange24h', 0),
                    'error': None
                }
                
        except Exception as e:
            logger.error(f"Error getting Birdeye data: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def _rate_limit(self, api_name: str):
        """Enforce rate limiting per API"""
        if api_name not in self.last_request_time:
            self.last_request_time[api_name] = 0
        
        elapsed = asyncio.get_event_loop().time() - self.last_request_time[api_name]
        delay = self.rate_limit_delays.get(api_name, 1.0)
        
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        
        self.last_request_time[api_name] = asyncio.get_event_loop().time()


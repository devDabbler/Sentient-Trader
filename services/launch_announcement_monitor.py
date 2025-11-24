"""
Launch Announcement Monitor - Detects new token launches from multiple sources

Monitors:
1. Pump.fun API (Solana new tokens)
2. Twitter/X mentions (requires API)
3. Telegram channels (via webhook/bot)
4. DexScreener boosted tokens
5. Community announcements

Usage:
    monitor = LaunchAnnouncementMonitor()
    await monitor.start_monitoring()
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from loguru import logger
import os
import json

from models.dex_models import Chain


@dataclass
class LaunchAnnouncement:
    """Detected launch announcement"""
    source: str  # "pumpfun", "twitter", "telegram", "dexscreener"
    token_address: str
    token_symbol: str
    token_name: str
    chain: Chain
    announced_at: datetime
    announcement_url: Optional[str] = None
    announcement_text: Optional[str] = None
    confidence_score: float = 0.0  # How confident this is a real launch
    metadata: Dict = field(default_factory=dict)


class LaunchAnnouncementMonitor:
    """Monitor multiple sources for new token launch announcements"""
    
    # Pump.fun API (Solana - NO AUTH NEEDED!)
    PUMPFUN_API = "https://frontend-api.pump.fun"
    
    # DexScreener boosted tokens (trending/promoted)
    DEXSCREENER_BOOSTED = "https://api.dexscreener.com/token-boosts/latest/v1"
    
    def __init__(
        self,
        twitter_bearer_token: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        scan_interval_seconds: int = 300  # 5 minutes default
    ):
        """
        Initialize monitor
        
        Args:
            twitter_bearer_token: Twitter API v2 bearer token (optional)
            telegram_bot_token: Telegram bot token (optional)
            scan_interval_seconds: How often to check sources (default 5 min)
        """
        self.twitter_token = twitter_bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        self.telegram_token = telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.scan_interval = scan_interval_seconds
        
        self.seen_announcements: Set[str] = set()  # Track to avoid duplicates
        self.recent_announcements: List[LaunchAnnouncement] = []
        self.is_running = False
        
        # Rate limiting
        self.last_pumpfun_call = 0
        self.last_twitter_call = 0
        self.last_dexscreener_call = 0
        
        logger.info(f"LaunchAnnouncementMonitor initialized (scan every {scan_interval_seconds}s)")
    
    async def start_monitoring(self):
        """Start continuous monitoring loop"""
        self.is_running = True
        logger.info("🔍 Starting launch announcement monitoring...")
        
        while self.is_running:
            try:
                # Check all sources
                announcements = await self._check_all_sources()
                
                if announcements:
                    logger.info(f"📢 Found {len(announcements)} new launch announcements!")
                    
                    for announcement in announcements:
                        # Add to recent list
                        self.recent_announcements.append(announcement)
                        
                        # Log it
                        logger.info(
                            f"🚀 NEW LAUNCH: {announcement.token_symbol} "
                            f"({announcement.chain.value}) from {announcement.source}"
                        )
                
                # Keep recent list manageable (last 100)
                if len(self.recent_announcements) > 100:
                    self.recent_announcements = self.recent_announcements[-100:]
                
                # Wait before next check
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 min on error
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        logger.info("Stopped launch announcement monitoring")
    
    async def _check_all_sources(self) -> List[LaunchAnnouncement]:
        """Check all configured sources for new announcements"""
        announcements = []
        
        # 1. Pump.fun (Solana) - Always available, no auth!
        try:
            pumpfun_launches = await self._check_pumpfun()
            announcements.extend(pumpfun_launches)
        except Exception as e:
            logger.debug(f"Pump.fun check failed: {e}")
        
        # 2. DexScreener boosted tokens
        try:
            boosted_tokens = await self._check_dexscreener_boosted()
            announcements.extend(boosted_tokens)
        except Exception as e:
            logger.debug(f"DexScreener boosted check failed: {e}")
        
        # 3. Twitter (if configured)
        if self.twitter_token:
            try:
                twitter_launches = await self._check_twitter()
                announcements.extend(twitter_launches)
            except Exception as e:
                logger.debug(f"Twitter check failed: {e}")
        
        # 4. Telegram (if configured)
        if self.telegram_token:
            try:
                telegram_launches = await self._check_telegram()
                announcements.extend(telegram_launches)
            except Exception as e:
                logger.debug(f"Telegram check failed: {e}")
        
        # Filter out duplicates
        unique_announcements = []
        for announcement in announcements:
            key = f"{announcement.token_address}_{announcement.chain.value}"
            if key not in self.seen_announcements:
                self.seen_announcements.add(key)
                unique_announcements.append(announcement)
        
        return unique_announcements
    
    async def _check_pumpfun(self) -> List[LaunchAnnouncement]:
        """
        Check Pump.fun for new Solana token launches
        
        Pump.fun is THE source for Solana meme coins - most go viral here first!
        API is FREE and public, no auth needed!
        """
        # Rate limit: 1 call per 10 seconds
        now = datetime.now().timestamp()
        if now - self.last_pumpfun_call < 10:
            return []
        
        self.last_pumpfun_call = now
        
        announcements = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get latest token launches
                async with session.get(
                    f"{self.PUMPFUN_API}/coins/latest",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process tokens created in last 5 minutes
                        five_min_ago = datetime.now() - timedelta(minutes=5)
                        
                        for token in data[:20]:  # Check top 20 latest
                            created_timestamp = token.get('created_timestamp', 0)
                            created_at = datetime.fromtimestamp(created_timestamp / 1000)
                            
                            if created_at > five_min_ago:
                                announcement = LaunchAnnouncement(
                                    source="pumpfun",
                                    token_address=token.get('mint', ''),
                                    token_symbol=token.get('symbol', 'UNKNOWN'),
                                    token_name=token.get('name', 'Unknown'),
                                    chain=Chain.SOLANA,
                                    announced_at=created_at,
                                    announcement_url=f"https://pump.fun/coin/{token.get('mint', '')}",
                                    confidence_score=80.0,  # Pump.fun is reliable
                                    metadata={
                                        'market_cap': token.get('market_cap', 0),
                                        'description': token.get('description', ''),
                                        'twitter': token.get('twitter', ''),
                                        'telegram': token.get('telegram', ''),
                                        'website': token.get('website', '')
                                    }
                                )
                                announcements.append(announcement)
                        
                        if announcements:
                            logger.info(f"📢 Pump.fun: Found {len(announcements)} new launches!")
                    
        except Exception as e:
            logger.debug(f"Pump.fun API error: {e}")
        
        return announcements
    
    async def _check_dexscreener_boosted(self) -> List[LaunchAnnouncement]:
        """
        Check DexScreener for boosted/promoted tokens
        
        Boosted tokens = projects paying for visibility = serious launches
        """
        # Rate limit: 1 call per 20 seconds
        now = datetime.now().timestamp()
        if now - self.last_dexscreener_call < 20:
            return []
        
        self.last_dexscreener_call = now
        
        announcements = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.DEXSCREENER_BOOSTED,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check boosts from last 30 minutes
                        thirty_min_ago = datetime.now() - timedelta(minutes=30)
                        
                        for boost in data[:10]:  # Top 10 boosts
                            # Parse boost data
                            chain_id = boost.get('chainId', '')
                            chain = self._parse_chain_id(chain_id)
                            
                            if not chain:
                                continue
                            
                            token_address = boost.get('tokenAddress', '')
                            
                            announcement = LaunchAnnouncement(
                                source="dexscreener_boost",
                                token_address=token_address,
                                token_symbol=boost.get('name', 'UNKNOWN'),
                                token_name=boost.get('description', 'Unknown'),
                                chain=chain,
                                announced_at=datetime.now(),
                                announcement_url=boost.get('url', ''),
                                confidence_score=70.0,  # Boosted = paid promotion
                                metadata={
                                    'boost_amount': boost.get('amount', 0),
                                    'links': boost.get('links', [])
                                }
                            )
                            announcements.append(announcement)
                        
                        if announcements:
                            logger.info(f"📢 DexScreener: Found {len(announcements)} boosted tokens!")
                    
        except Exception as e:
            logger.debug(f"DexScreener boosted API error: {e}")
        
        return announcements
    
    async def _check_twitter(self) -> List[LaunchAnnouncement]:
        """
        Check Twitter for launch announcements
        
        Requires Twitter API v2 bearer token (free tier: 500k tweets/month)
        """
        if not self.twitter_token:
            return []
        
        # Rate limit: 1 call per 15 seconds (free tier limit)
        now = datetime.now().timestamp()
        if now - self.last_twitter_call < 15:
            return []
        
        self.last_twitter_call = now
        
        announcements = []
        
        try:
            # Search for launch tweets in last 15 minutes
            search_query = "(new launch OR just launched) crypto -is:retweet"
            
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.twitter_token}"}
                
                async with session.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    params={
                        "query": search_query,
                        "max_results": 10,
                        "tweet.fields": "created_at,text"
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for tweet in data.get('data', []):
                            # Try to extract contract address from tweet
                            text = tweet.get('text', '')
                            address = self._extract_contract_address(text)
                            
                            if address:
                                # Determine chain from address format
                                chain = self._guess_chain_from_address(address)
                                
                                announcement = LaunchAnnouncement(
                                    source="twitter",
                                    token_address=address,
                                    token_symbol="UNKNOWN",  # Parse from text if possible
                                    token_name="Unknown",
                                    chain=chain,
                                    announced_at=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                                    announcement_url=f"https://twitter.com/i/web/status/{tweet['id']}",
                                    announcement_text=text,
                                    confidence_score=50.0,  # Lower confidence (needs verification)
                                    metadata={'tweet_id': tweet['id']}
                                )
                                announcements.append(announcement)
                        
                        if announcements:
                            logger.info(f"📢 Twitter: Found {len(announcements)} launch tweets!")
                    
        except Exception as e:
            logger.debug(f"Twitter API error: {e}")
        
        return announcements
    
    async def _check_telegram(self) -> List[LaunchAnnouncement]:
        """
        Check Telegram channels for launch announcements
        
        Requires Telegram bot token and channel access
        """
        # TODO: Implement Telegram bot monitoring
        # Would require setting up webhook or polling updates
        logger.debug("Telegram monitoring not yet implemented")
        return []
    
    def _parse_chain_id(self, chain_id: str) -> Optional[Chain]:
        """Parse DexScreener chain ID to Chain enum"""
        chain_map = {
            'ethereum': Chain.ETH,
            'bsc': Chain.BSC,
            'solana': Chain.SOLANA,
            'base': Chain.BASE,
            'arbitrum': Chain.ARBITRUM,
            'polygon': Chain.POLYGON
        }
        return chain_map.get(chain_id.lower())
    
    def _extract_contract_address(self, text: str) -> Optional[str]:
        """Extract contract address from text"""
        import re
        
        # EVM address pattern (0x...)
        evm_pattern = r'0x[a-fA-F0-9]{40}'
        evm_match = re.search(evm_pattern, text)
        if evm_match:
            return evm_match.group(0)
        
        # Solana address pattern (base58, 32-44 chars)
        sol_pattern = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
        sol_match = re.search(sol_pattern, text)
        if sol_match:
            address = sol_match.group(0)
            # Verify it doesn't start with 0x
            if not address.startswith('0x'):
                return address
        
        return None
    
    def _guess_chain_from_address(self, address: str) -> Chain:
        """Guess chain from address format"""
        if address.startswith('0x'):
            return Chain.ETH  # Default to ETH for EVM addresses
        else:
            return Chain.SOLANA  # Assume Solana for base58
    
    def get_recent_announcements(self, minutes: int = 30) -> List[LaunchAnnouncement]:
        """Get announcements from last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            a for a in self.recent_announcements
            if a.announced_at > cutoff
        ]
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'total_announcements': len(self.recent_announcements),
            'unique_tokens': len(self.seen_announcements),
            'last_30_min': len(self.get_recent_announcements(30)),
            'is_running': self.is_running,
            'sources_configured': {
                'pumpfun': True,  # Always available
                'dexscreener': True,  # Always available
                'twitter': bool(self.twitter_token),
                'telegram': bool(self.telegram_token)
            }
        }


# Singleton instance
_monitor_instance: Optional[LaunchAnnouncementMonitor] = None


def get_announcement_monitor(
    twitter_token: Optional[str] = None,
    telegram_token: Optional[str] = None,
    scan_interval: int = 300
) -> LaunchAnnouncementMonitor:
    """Get or create announcement monitor singleton"""
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = LaunchAnnouncementMonitor(
            twitter_bearer_token=twitter_token,
            telegram_bot_token=telegram_token,
            scan_interval_seconds=scan_interval
        )
    
    return _monitor_instance

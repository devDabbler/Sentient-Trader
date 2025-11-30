"""
Solana RPC Load Balancer

Distributes RPC requests across multiple endpoints to avoid rate limits.
Supports:
- Multiple Alchemy endpoints
- Public RPC endpoints
- Round-robin or failover routing
- Automatic retry on failure
"""

import os
import random
from typing import List, Optional, Dict
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class SolanaRPCLoadBalancer:
    """Load balance Solana RPC requests across multiple endpoints"""
    
    def __init__(self, rpc_urls: Optional[List[str]] = None):
        """
        Initialize RPC load balancer
        
        Args:
            rpc_urls: List of RPC URLs to use. If None, reads from env vars.
        
        Environment variables:
            SOLANA_RPC_URL: Primary RPC endpoint
            SOLANA_RPC_URL_2, SOLANA_RPC_URL_3, etc.: Additional endpoints
            SOLANA_RPC_URLS: Comma-separated list of endpoints
        """
        self.rpc_urls: List[str] = []
        self.current_index = 0
        self.failed_endpoints: Dict[str, int] = {}  # endpoint -> failure count
        self.max_failures = 3  # Remove endpoint after 3 failures
        
        # Load RPC URLs from environment or use provided
        if rpc_urls:
            self.rpc_urls = rpc_urls
        else:
            self._load_rpc_urls_from_env()
        
        if not self.rpc_urls:
            # Fallback to multiple public endpoints for better rate limit handling
            self.rpc_urls = [
                "https://api.mainnet-beta.solana.com",
                "https://solana-mainnet.g.alchemy.com/v2/demo",  # Alchemy free tier
                "https://rpc.ankr.com/solana",  # Ankr free tier
                "https://solana-api.projectserum.com",  # Project Serum
            ]
            logger.warning("No RPC URLs configured, using public endpoints with load balancing")
        
        logger.info(f"RPC Load Balancer initialized with {len(self.rpc_urls)} endpoint(s)")
        for i, url in enumerate(self.rpc_urls, 1):
            logger.info(f"  {i}. {url[:60]}...")
    
    def _load_rpc_urls_from_env(self):
        """Load RPC URLs from environment variables"""
        # Primary endpoint
        primary = os.getenv("SOLANA_RPC_URL")
        if primary:
            self.rpc_urls.append(primary)
        
        # Check for comma-separated list
        urls_str = os.getenv("SOLANA_RPC_URLS")
        if urls_str:
            urls = [url.strip() for url in urls_str.split(",") if url.strip()]
            for url in urls:
                if url not in self.rpc_urls:
                    self.rpc_urls.append(url)
        
        # Check for numbered endpoints (SOLANA_RPC_URL_2, SOLANA_RPC_URL_3, etc.)
        i = 2
        while True:
            endpoint = os.getenv(f"SOLANA_RPC_URL_{i}")
            if not endpoint:
                break
            if endpoint not in self.rpc_urls:
                self.rpc_urls.append(endpoint)
            i += 1
            if i > 10:  # Limit to 10 endpoints
                break
    
    def get_next_rpc_url(self, strategy: str = "round_robin") -> str:
        """
        Get next RPC URL based on strategy
        
        Args:
            strategy: "round_robin" or "random"
            
        Returns:
            RPC URL string
        """
        # Filter out failed endpoints temporarily
        available = [
            url for url in self.rpc_urls
            if self.failed_endpoints.get(url, 0) < self.max_failures
        ]
        
        if not available:
            # Reset failures if all endpoints failed
            logger.warning("All RPC endpoints failed, resetting failure counts")
            self.failed_endpoints.clear()
            available = self.rpc_urls
        
        if strategy == "round_robin":
            url = available[self.current_index % len(available)]
            self.current_index += 1
            return url
        elif strategy == "random":
            return random.choice(available)
        else:
            return available[0]
    
    def mark_failure(self, rpc_url: str):
        """Mark an RPC endpoint as failed"""
        self.failed_endpoints[rpc_url] = self.failed_endpoints.get(rpc_url, 0) + 1
        logger.warning(f"RPC endpoint failed: {rpc_url} (failures: {self.failed_endpoints[rpc_url]})")
    
    def mark_success(self, rpc_url: str):
        """Mark an RPC endpoint as successful (reset failure count)"""
        if rpc_url in self.failed_endpoints:
            del self.failed_endpoints[rpc_url]
            logger.debug(f"RPC endpoint recovered: {rpc_url}")
    
    def get_primary_rpc_url(self) -> str:
        """Get the primary (first) RPC URL"""
        return self.rpc_urls[0] if self.rpc_urls else "https://api.mainnet-beta.solana.com"
    
    def get_all_rpc_urls(self) -> List[str]:
        """Get all configured RPC URLs"""
        return self.rpc_urls.copy()


# Global instance
_global_load_balancer: Optional[SolanaRPCLoadBalancer] = None


def get_rpc_load_balancer() -> SolanaRPCLoadBalancer:
    """Get or create global RPC load balancer instance"""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = SolanaRPCLoadBalancer()
    return _global_load_balancer


def get_rpc_url(strategy: str = "round_robin") -> str:
    """Convenience function to get next RPC URL"""
    return get_rpc_load_balancer().get_next_rpc_url(strategy)


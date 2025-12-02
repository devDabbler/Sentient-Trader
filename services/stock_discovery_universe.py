"""
Stock Discovery Universe Service

Uses TieredStockScanner categories to discover trading opportunities outside the watchlist.
Leverages existing scanner infrastructure from the main app.
"""

import os
import sys
import time
from typing import List, Optional, Dict, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from loguru import logger

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.stock_tiered_scanner import get_tiered_stock_scanner


@dataclass
class DiscoveryMode:
    """Configuration for discovery mode - maps to TieredStockScanner categories"""
    name: str
    description: str
    enabled: bool = False
    max_universe_size: int = 50
    priority: int = 1  # 1=high priority, higher=lower


class StockDiscoveryUniverse:
    """
    Discovers stocks outside watchlist using TieredStockScanner categories
    Uses the same scanner infrastructure as the main app
    """
    
    def __init__(self, detector=None):
        """
        Initialize discovery universe
        
        Args:
            detector: Enhanced stock opportunity detector instance (not used, kept for compatibility)
        """
        # Initialize TieredStockScanner
        self.scanner = get_tiered_stock_scanner(use_ai=False)  # No AI for discovery, just filtering
        
        # Map scanner categories to discovery modes
        # These match the categories available in the main app's Daily Stock Scanner
        category_map = {
            'mega_cap': ('Mega Caps', 'Options-friendly large caps (AAPL, MSFT, etc.)'),
            'high_beta_tech': ('High Beta Tech', 'Volatile tech stocks (PLTR, SOFI, etc.)'),
            'momentum': ('Momentum/Meme', 'High momentum and meme stocks'),
            'ev_energy': ('EV/Clean Energy', 'Electric vehicle and clean energy stocks'),
            'crypto_related': ('Crypto-Related', 'Stocks tied to crypto (MARA, RIOT, COIN)'),
            'ai_stocks': ('AI Stocks', 'Artificial intelligence related stocks'),
            'biotech': ('Biotech', 'Biotechnology and pharma stocks'),
            'financial': ('Financial', 'Banks and financial services'),
            'energy': ('Energy', 'Oil and gas stocks'),
            'high_iv': ('High IV Options', 'High implied volatility for options trading'),
            'penny_stocks': ('Penny Stocks', 'Low-priced stocks under $5'),
        }
        
        # Discovery modes mapped to scanner categories
        self.modes = {}
        for category_key, (display_name, description) in category_map.items():
            self.modes[category_key] = DiscoveryMode(
                category_key,
                description,
                enabled=False,
                max_universe_size=30 if category_key in ['mega_cap', 'high_beta_tech'] else 20,
                priority=1 if category_key in ['mega_cap', 'high_beta_tech', 'momentum'] else 2
            )
        
        # Cache for discovered tickers
        self.discovered_tickers: Dict[str, List[str]] = {}
        self.discovery_cache_time: Dict[str, datetime] = {}
        self.cache_ttl_minutes = 30  # Cache for 30 minutes (shorter than before for fresher results)
        
        logger.info("✅ Stock Discovery Universe initialized (using TieredStockScanner)")
        logger.info(f"   Available categories: {len(self.modes)}")
    
    def set_discovery_enabled(self, mode_name: str, enabled: bool):
        """
        Enable or disable a discovery mode
        
        Args:
            mode_name: Name of mode ('top_gainers', 'most_active', etc.)
            enabled: True to enable, False to disable
        """
        if mode_name in self.modes:
            self.modes[mode_name].enabled = enabled
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            logger.info(f"Discovery mode '{mode_name}': {status}")
        else:
            logger.warning(f"Unknown discovery mode: {mode_name}")
    
    def get_discovery_config(self) -> Dict:
        """Get current discovery configuration"""
        config = {}
        for name, mode in self.modes.items():
            config[name] = {
                'enabled': mode.enabled,
                'description': mode.description,
                'max_universe_size': mode.max_universe_size,
                'priority': mode.priority,
            }
        return config
    
    def set_discovery_config(self, config: Dict):
        """
        Set discovery configuration from control panel
        
        Args:
            config: Dict with mode names and settings
        """
        for mode_name, settings in config.items():
            if mode_name in self.modes:
                if 'enabled' in settings:
                    self.modes[mode_name].enabled = settings['enabled']
                if 'max_universe_size' in settings:
                    self.modes[mode_name].max_universe_size = settings['max_universe_size']
                if 'priority' in settings:
                    self.modes[mode_name].priority = settings['priority']
                logger.debug(f"Updated mode {mode_name}: {settings}")
    
    def discover_stocks(self, exclude_watchlist: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Discover stocks from enabled categories using TieredStockScanner
        
        Args:
            exclude_watchlist: List of tickers to exclude from discovery
            
        Returns:
            Dict of mode_name -> list of tickers
        """
        if exclude_watchlist is None:
            exclude_watchlist = []
        
        exclude_set = set(t.upper() for t in exclude_watchlist)
        discovered = {}
        
        logger.info("🔍 Starting stock discovery using TieredStockScanner...")
        
        for mode_name, mode in self.modes.items():
            if not mode.enabled:
                continue
            
            try:
                logger.info(f"  📊 Scanning {mode.description}...")
                
                # Check cache
                if self._is_cache_valid(mode_name):
                    tickers = self.discovered_tickers.get(mode_name, [])
                    logger.debug(f"    (using cache: {len(tickers)} tickers)")
                else:
                    # Get tickers from scanner category
                    category_tickers = self.scanner.get_tickers_by_category(mode_name)
                    
                    if not category_tickers:
                        logger.warning(f"    No tickers found for category: {mode_name}")
                        discovered[mode_name] = []
                        continue
                    
                    # Use Tier 1 quick filter to find promising stocks
                    logger.debug(f"    Running Tier 1 filter on {len(category_tickers)} tickers...")
                    tier1_results = self.scanner.tier1_quick_filter(
                        category_tickers,
                        max_results=mode.max_universe_size
                    )
                    
                    # Extract tickers from results, excluding watchlist
                    tickers = [
                        r['ticker'] for r in tier1_results
                        if r['ticker'].upper() not in exclude_set
                    ]
                    
                    # Cache results
                    self.discovered_tickers[mode_name] = tickers
                    self.discovery_cache_time[mode_name] = datetime.now()
                    
                    logger.info(f"    ✅ Discovered {len(tickers)} tickers from {mode.description}")
                
                discovered[mode_name] = tickers
            
            except Exception as e:
                logger.error(f"Error discovering {mode_name}: {e}", exc_info=True)
                discovered[mode_name] = []
        
        # Combine all discovered tickers (deduplicated)
        all_discovered = set()
        for tickers in discovered.values():
            all_discovered.update(t.upper() for t in tickers)
        
        logger.info(f"🎯 Total discovered: {len(all_discovered)} unique tickers")
        
        return discovered
    
    def analyze_discovered_stocks(
        self, 
        discovered_dict: Dict[str, List[str]], 
        min_score: float = 70.0
    ) -> Dict[str, List[Dict]]:
        """
        Analyze discovered stocks for opportunities
        
        Args:
            discovered_dict: Dict from discover_stocks()
            min_score: Minimum composite score threshold
            
        Returns:
            Dict of mode_name -> list of opportunities
        """
        opportunities_by_mode = {}
        total_opportunities = 0
        
        for mode_name, tickers in discovered_dict.items():
            if not tickers:
                continue
            
            mode_opportunities = []
            logger.info(f"📈 Analyzing {len(tickers)} stocks from {mode_name}...")
            
            for ticker in tickers:
                try:
                    opp = self.detector.analyze_ticker(ticker)
                    if opp and opp.composite_score >= min_score:
                        mode_opportunities.append({
                            'symbol': ticker,
                            'composite_score': opp.composite_score,
                            'confidence': opp.confidence,
                            'price': opp.price,
                            'trend': opp.technical_signals.trend,
                            'volume_ratio': opp.technical_signals.volume_ratio,
                            'reasoning': opp.composite_reasoning,
                            'mode': mode_name,
                        })
                        logger.debug(f"  ✅ {ticker}: {opp.composite_score:.0f} ({opp.confidence})")
                except Exception as e:
                    logger.debug(f"  Error analyzing {ticker}: {e}")
            
            if mode_opportunities:
                # Sort by score
                mode_opportunities.sort(key=lambda x: x['composite_score'], reverse=True)
                opportunities_by_mode[mode_name] = mode_opportunities
                total_opportunities += len(mode_opportunities)
                logger.info(f"  Found {len(mode_opportunities)} opportunities from {mode_name}")
        
        logger.info(f"🎯 Total discovery opportunities: {total_opportunities}")
        
        return opportunities_by_mode
    
    # ============================================================
    # DISCOVERY METHODS
    # ============================================================
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def _is_valid_stock_ticker(self, ticker: str) -> bool:
        """
        Validate that a ticker is a valid stock ticker (not crypto, not invalid format)
        
        Args:
            ticker: The ticker symbol to validate
            
        Returns:
            True if valid stock ticker, False otherwise
        """
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.upper().strip()
        
        # Skip crypto pairs (e.g., BTC/USD, ETH/USD)
        if '/' in ticker:
            return False
        
        # Skip if too short or too long
        if len(ticker) < 1 or len(ticker) > 6:
            return False
        
        # Allow alphanumeric with dots (for BRK.B, BRK.A style tickers)
        if not ticker.replace('.', '').replace('-', '').isalnum():
            return False
        
        # Skip common crypto tickers that might slip through
        crypto_tickers = {'BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'AVAX', 'MATIC', 'DOT', 'LINK'}
        if ticker in crypto_tickers:
            return False
        
        return True
    
    def _is_cache_valid(self, mode_name: str) -> bool:
        """Check if cache is still valid"""
        if mode_name not in self.discovery_cache_time:
            return False
        age_minutes = (datetime.now() - self.discovery_cache_time[mode_name]).total_seconds() / 60
        return age_minutes < self.cache_ttl_minutes


# Singleton instance
_discovery_instance = None


def get_stock_discovery_universe(detector=None) -> StockDiscoveryUniverse:
    """Get singleton instance of discovery universe"""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = StockDiscoveryUniverse(detector=detector)
    return _discovery_instance


if __name__ == "__main__":
    discovery = get_stock_discovery_universe()
    
    # Enable all modes
    for mode_name in discovery.modes.keys():
        discovery.set_discovery_enabled(mode_name, True)
    
    # Discover stocks
    discovered = discovery.discover_stocks(exclude_watchlist=['AAPL', 'SPY'])
    
    # Analyze them
    opportunities = discovery.analyze_discovered_stocks(discovered, min_score=70)
    
    for mode, opps in opportunities.items():
        logger.info(f"\n{mode}:")
        for opp in opps[:5]:
            logger.info(f"  {opp['symbol']}: {opp['composite_score']:.0f} ({opp['confidence']})")


"""
Stock Discovery Universe Service

Automatically discovers trading opportunities outside the watchlist:
- Top gainers/losers
- Most active by volume
- Technical breakouts (new highs, momentum plays)
- Sector rotations
- Trend-following strategies
- Screener-based discovery

Configurable via Service Control Panel to enable/disable discovery modes
and set universe size.
"""

import os
import sys
import time
from typing import List, Optional, Dict, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from loguru import logger
import yfinance as yf
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.enhanced_stock_opportunity_detector import get_enhanced_stock_detector


@dataclass
class DiscoveryMode:
    """Configuration for discovery mode"""
    name: str
    description: str
    enabled: bool = False
    max_universe_size: int = 50
    priority: int = 1  # 1=high priority, higher=lower


class StockDiscoveryUniverse:
    """
    Discovers stocks outside watchlist using multiple sources
    Can be toggled on/off via control panel
    """
    
    def __init__(self, detector=None):
        """
        Initialize discovery universe
        
        Args:
            detector: Enhanced stock opportunity detector instance
        """
        self.detector = detector or get_enhanced_stock_detector()
        
        # Discovery modes (all can be toggled)
        self.modes = {
            'top_gainers': DiscoveryMode(
                'top_gainers',
                'Top percentage gainers (momentum plays)',
                enabled=False,
                max_universe_size=30
            ),
            'top_losers': DiscoveryMode(
                'top_losers',
                'Top percentage losers (reversal plays)',
                enabled=False,
                max_universe_size=20
            ),
            'most_active': DiscoveryMode(
                'most_active',
                'Most active by volume (breakout candidates)',
                enabled=False,
                max_universe_size=40
            ),
            'new_highs': DiscoveryMode(
                'new_highs',
                '52-week new highs (trend breakouts)',
                enabled=False,
                max_universe_size=30
            ),
            'high_volume_breakouts': DiscoveryMode(
                'high_volume_breakouts',
                'High volume with price movement (technical breakouts)',
                enabled=False,
                max_universe_size=25
            ),
            'sector_leaders': DiscoveryMode(
                'sector_leaders',
                'Top performers in each sector',
                enabled=False,
                max_universe_size=20
            ),
        }
        
        # Cache for discovered tickers
        self.discovered_tickers: Dict[str, List[str]] = {}
        self.discovery_cache_time: Dict[str, datetime] = {}
        self.cache_ttl_minutes = 60  # Update every hour
        
        # Known universe pools
        self.sp500_tickers = self._get_sp500_tickers()
        self.russell_2000 = self._get_russell_2000_tickers()
        
        logger.info("✅ Stock Discovery Universe initialized")
        logger.info(f"   Available modes: {len(self.modes)}")
        logger.info(f"   SP500 universe: {len(self.sp500_tickers) if self.sp500_tickers else 'N/A'}")
    
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
        Discover stocks from enabled modes
        
        Args:
            exclude_watchlist: List of tickers to exclude from discovery
            
        Returns:
            Dict of mode_name -> list of tickers
        """
        if exclude_watchlist is None:
            exclude_watchlist = []
        
        exclude_set = set(t.upper() for t in exclude_watchlist)
        discovered = {}
        
        logger.info("🔍 Starting stock discovery...")
        
        for mode_name, mode in self.modes.items():
            if not mode.enabled:
                continue
            
            try:
                logger.debug(f"  📊 {mode.name}: {mode.description}...")
                
                # Check cache
                if self._is_cache_valid(mode_name):
                    tickers = self.discovered_tickers.get(mode_name, [])
                    logger.debug(f"    (using cache)")
                else:
                    # Discover new tickers
                    tickers = self._discover_mode_tickers(mode_name, mode)
                    
                    # Exclude watchlist
                    tickers = [t for t in tickers if t.upper() not in exclude_set]
                    
                    # Cache results
                    self.discovered_tickers[mode_name] = tickers
                    self.discovery_cache_time[mode_name] = datetime.now()
                    
                    logger.info(f"    ✅ Discovered {len(tickers)} tickers from {mode.name}")
                
                discovered[mode_name] = tickers
            
            except Exception as e:
                logger.error(f"Error discovering {mode_name}: {e}")
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
    
    def _discover_mode_tickers(self, mode_name: str, mode: DiscoveryMode) -> List[str]:
        """Discover tickers for specific mode"""
        
        if mode_name == 'top_gainers':
            return self._get_top_gainers(mode.max_universe_size)
        elif mode_name == 'top_losers':
            return self._get_top_losers(mode.max_universe_size)
        elif mode_name == 'most_active':
            return self._get_most_active(mode.max_universe_size)
        elif mode_name == 'new_highs':
            return self._get_new_highs(mode.max_universe_size)
        elif mode_name == 'high_volume_breakouts':
            return self._get_volume_breakouts(mode.max_universe_size)
        elif mode_name == 'sector_leaders':
            return self._get_sector_leaders(mode.max_universe_size)
        else:
            return []
    
    def _get_top_gainers(self, limit: int = 30) -> List[str]:
        """Get top percentage gainers"""
        try:
            # Using a subset of liquid stocks
            universe = self.sp500_tickers[:200] if self.sp500_tickers else []
            if not universe:
                logger.debug("    SP500 universe not available, using defaults")
                universe = ['SPY', 'QQQ', 'IWM', 'EEM', 'FXI', 'EWJ', 'EWG', 'EWU']
            
            gainers = []
            for ticker in universe[:limit * 2]:  # Check more to get top
                try:
                    hist = yf.download(ticker, period='5d', progress=False, threads=False)
                    if len(hist) > 0:
                        pct_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                        if pct_change > 2:  # At least 2% gain
                            gainers.append({'ticker': ticker, 'change': pct_change})
                except:
                    pass
            
            # Sort by change and return top
            gainers.sort(key=lambda x: x['change'], reverse=True)
            return [g['ticker'] for g in gainers[:limit]]
        
        except Exception as e:
            logger.debug(f"Error getting top gainers: {e}")
            return []
    
    def _get_top_losers(self, limit: int = 20) -> List[str]:
        """Get top percentage losers (reversal candidates)"""
        try:
            universe = self.sp500_tickers[:200] if self.sp500_tickers else []
            if not universe:
                universe = ['SPY', 'QQQ', 'IWM', 'EEM']
            
            losers = []
            for ticker in universe[:limit * 3]:
                try:
                    hist = yf.download(ticker, period='5d', progress=False, threads=False)
                    if len(hist) > 0:
                        pct_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                        if -5 < pct_change < -1:  # -5% to -1% loss (reversal play)
                            losers.append({'ticker': ticker, 'change': pct_change})
                except:
                    pass
            
            losers.sort(key=lambda x: x['change'])
            return [l['ticker'] for l in losers[:limit]]
        
        except Exception as e:
            logger.debug(f"Error getting top losers: {e}")
            return []
    
    def _get_most_active(self, limit: int = 40) -> List[str]:
        """Get most active stocks by volume"""
        try:
            universe = self.sp500_tickers[:300] if self.sp500_tickers else []
            if not universe:
                universe = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'TLT']
            
            active = []
            for ticker in universe[:limit * 2]:
                try:
                    hist = yf.download(ticker, period='5d', progress=False, threads=False)
                    if len(hist) > 1:
                        recent_vol = hist['Volume'].iloc[-1]
                        avg_vol = hist['Volume'].iloc[-5:].mean()
                        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
                        if vol_ratio > 1.3:  # 30% above average
                            active.append({'ticker': ticker, 'vol_ratio': vol_ratio})
                except:
                    pass
            
            active.sort(key=lambda x: x['vol_ratio'], reverse=True)
            return [a['ticker'] for a in active[:limit]]
        
        except Exception as e:
            logger.debug(f"Error getting most active: {e}")
            return []
    
    def _get_new_highs(self, limit: int = 30) -> List[str]:
        """Get 52-week new highs (trend followers)"""
        try:
            universe = self.sp500_tickers[:200] if self.sp500_tickers else []
            if not universe:
                universe = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA']
            
            new_highs = []
            for ticker in universe[:limit * 2]:
                try:
                    hist = yf.download(ticker, period='1y', progress=False, threads=False)
                    if len(hist) > 50:
                        current = hist['Close'].iloc[-1]
                        high_52w = hist['Close'].max()
                        if current >= high_52w * 0.99:  # Within 1% of 52-week high
                            new_highs.append({'ticker': ticker, 'price': current})
                except:
                    pass
            
            return [n['ticker'] for n in new_highs[:limit]]
        
        except Exception as e:
            logger.debug(f"Error getting new highs: {e}")
            return []
    
    def _get_volume_breakouts(self, limit: int = 25) -> List[str]:
        """Get high volume with price movement breakouts"""
        try:
            universe = self.sp500_tickers[:250] if self.sp500_tickers else []
            if not universe:
                universe = ['SPY', 'QQQ', 'IWM']
            
            breakouts = []
            for ticker in universe[:limit * 3]:
                try:
                    hist = yf.download(ticker, period='10d', progress=False, threads=False)
                    if len(hist) > 2:
                        # High volume
                        recent_vol = hist['Volume'].iloc[-1]
                        avg_vol = hist['Volume'].iloc[:-1].mean()
                        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
                        
                        # Price movement
                        price_move = abs((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2])
                        
                        if vol_ratio > 1.5 and price_move > 0.02:  # 50% vol spike + 2% price move
                            breakouts.append({'ticker': ticker, 'vol_ratio': vol_ratio})
                except:
                    pass
            
            breakouts.sort(key=lambda x: x['vol_ratio'], reverse=True)
            return [b['ticker'] for b in breakouts[:limit]]
        
        except Exception as e:
            logger.debug(f"Error getting volume breakouts: {e}")
            return []
    
    def _get_sector_leaders(self, limit: int = 20) -> List[str]:
        """Get top performers in each major sector"""
        # Simplified - use sector ETFs as proxy
        sector_etfs = ['XLK', 'XLV', 'XLE', 'XLI', 'XLY', 'XLRE', 'XLF']
        return sector_etfs[:limit]
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def _is_cache_valid(self, mode_name: str) -> bool:
        """Check if cache is still valid"""
        if mode_name not in self.discovery_cache_time:
            return False
        age_minutes = (datetime.now() - self.discovery_cache_time[mode_name]).total_seconds() / 60
        return age_minutes < self.cache_ttl_minutes
    
    def _get_sp500_tickers(self) -> Optional[List[str]]:
        """Get SP500 tickers"""
        try:
            # Try to get from yfinance
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return sp500['Symbol'].tolist()
        except:
            # Fallback to common stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JNJ', 'V']
    
    def _get_russell_2000_tickers(self) -> Optional[List[str]]:
        """Get Russell 2000 tickers"""
        # Simplified fallback - in production would fetch real list
        return []


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


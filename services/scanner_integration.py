"""
Scanner Integration Helper

Provides easy integration of performance optimizations into existing scanners.
Allows gradual adoption without breaking existing functionality.
"""

from loguru import logger
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import numpy as np

# Import optimization modules
from .parallel_scanner import ParallelScannerIntegration
from .enhanced_cache import get_cache_instance, CacheConfig, smart_cache
from .enhanced_api_client import get_enhanced_yfinance_client
from .optimized_technical import OptimizedTechnicalAnalyzer, TechnicalAnalyzerOptimized



class OptimizedScanner:
    """
    Enhanced scanner that integrates all performance optimizations.
    
    Features:
    - Parallel processing with ThreadPoolExecutor
    - Multi-tier caching (memory + Supabase + disk)
    - Connection pooling for API clients
    - TA-Lib optimized technical calculations
    - Performance monitoring and metrics
    """
    
    def __init__(self, 
                 max_workers: int = 10,
                 enable_caching: bool = True,
                 enable_parallel: bool = True,
                 enable_talib: bool = True):
        """
        Initialize optimized scanner
        
        Args:
            max_workers: Maximum concurrent workers
            enable_caching: Enable multi-tier caching
            enable_parallel: Enable parallel processing
            enable_talib: Use TA-Lib optimizations
        """
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.enable_talib = enable_talib
        
        # Initialize components
        if enable_parallel:
            self.parallel_scanner = ParallelScannerIntegration(max_workers)
        
        self.yfinance_client = get_enhanced_yfinance_client(max_workers)
        
        # Performance tracking
        self.performance_stats = {
            'scans_executed': 0,
            'total_execution_time': 0.0,
            'cache_hits': 0,
            'parallel_scans': 0,
            'talib_calculations': 0
        }
    
    @smart_cache(ttl=CacheConfig.PRICE_DATA_EOD)
    def _get_ticker_data_cached(self, ticker: str) -> Optional[Dict]:
        """Get ticker data with caching"""
        try:
            result = self.yfinance_client._fetch_single_ticker(ticker)
            if 'error' not in result:
                return result
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
        
        return None
    
    def analyze_options_opportunities_optimized(self, tickers: List[str], top_n: int = 20) -> List[Dict]:
        """
        Optimized options opportunities scanner
        
        Args:
            tickers: List of ticker symbols
            top_n: Number of top results to return
            
        Returns:
            List of opportunity dictionaries sorted by score
        """
        start_time = time.perf_counter()
        
        if self.enable_parallel and len(tickers) > 10:
            # Use parallel processing for large lists
            results = self.parallel_scanner.parallel_options_scan(tickers)
            self.performance_stats['parallel_scans'] += 1
        else:
            # Sequential processing for small lists
            results = []
            for ticker in tickers:
                try:
                    data = self._get_ticker_data_cached(ticker)
                    if data:
                        analysis = self._analyze_options_opportunity_optimized(data)
                        if analysis and analysis.get('score', 0) > 0:
                            results.append(analysis)
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Update performance stats
        execution_time = time.perf_counter() - start_time
        self.performance_stats['scans_executed'] += 1
        self.performance_stats['total_execution_time'] += execution_time
        
        logger.info(f"Options scan completed: {len(results)} opportunities in {execution_time:.2f}s")
        
        return results[:top_n]
    
    def _analyze_options_opportunity_optimized(self, ticker_data: Dict) -> Optional[Dict]:
        """
        Optimized analysis of a single options opportunity
        
        Args:
            ticker_data: Dictionary with ticker information
            
        Returns:
            Analysis result dictionary
        """
        try:
            ticker = ticker_data['ticker']
            price = ticker_data['price']
            change_pct = ticker_data['change_pct']
            volume_ratio = ticker_data['volume_ratio']
            hist_data = ticker_data.get('hist_data')
            
            # Skip high-priced stocks for options
            if price > 5.0:
                return None
            
            # Calculate technical indicators using optimized methods
            tech_indicators = {}
            if self.enable_talib and hist_data is not None and not hist_data.empty:
                try:
                    # Use optimized technical analysis
                    tech_indicators = OptimizedTechnicalAnalyzer.batch_calculate_indicators(
                        hist_data, 
                        ['rsi', 'macd', 'ema_8', 'ema_21', 'atr']
                    )
                    self.performance_stats['talib_calculations'] += 1
                except Exception as e:
                    logger.warning(f"TA-Lib calculation failed for {ticker}: {e}")
            
            # Calculate opportunity score
            score = 50.0  # Base score
            reasons = []
            
            # Volume analysis (30 points)
            if volume_ratio > 2.0:
                score += 30
                reasons.append(f"Volume spike ({volume_ratio:.1f}x)")
            elif volume_ratio > 1.5:
                score += 20
                reasons.append(f"High volume ({volume_ratio:.1f}x)")
            elif volume_ratio > 1.0:
                score += 10
            
            # Price movement (25 points)
            if abs(change_pct) > 5:
                score += 25
                reasons.append(f"Big move ({change_pct:+.1f}%)")
            elif abs(change_pct) > 3:
                score += 15
                reasons.append(f"Strong move ({change_pct:+.1f}%)")
            elif abs(change_pct) > 1:
                score += 5
            
            # Technical indicator bonuses
            if tech_indicators:
                # RSI analysis (10 points)
                rsi = tech_indicators.get('rsi', 50)
                if 30 <= rsi <= 70:  # Good range for options
                    score += 10
                    reasons.append(f"Good RSI ({rsi:.1f})")
                
                # MACD analysis (10 points)
                macd_signal = tech_indicators.get('macd_signal', 'NEUTRAL')
                if macd_signal == 'BULLISH':
                    score += 10
                    reasons.append("MACD bullish")
                elif macd_signal == 'BEARISH':
                    score += 5
                    reasons.append("MACD bearish")
                
                # EMA analysis (10 points)
                ema8 = tech_indicators.get('ema_8')
                ema21 = tech_indicators.get('ema_21')
                if ema8 and ema21 and price > ema8 > ema21:
                    score += 10
                    reasons.append("Above EMAs")
            
            # Ultra-low price bonus
            if price < 1.0:
                score += 15
                reasons.append(f"Ultra-low price (${price:.3f})")
            elif price < 2.0:
                score += 10
            elif price < 3.0:
                score += 5
            
            # Determine confidence and risk
            if score >= 75:
                confidence = "VERY HIGH"
                risk = "M"
            elif score >= 60:
                confidence = "HIGH"  
                risk = "M"
            elif score >= 45:
                confidence = "MEDIUM"
                risk = "M-H"
            else:
                confidence = "LOW"
                risk = "H"
            
            return {
                'ticker': ticker,
                'score': round(score, 1),
                'price': price,
                'change_pct': change_pct,
                'volume_ratio': volume_ratio,
                'volume': ticker_data.get('volume', 0),
                'reason': " | ".join(reasons) if reasons else "Standard opportunity",
                'confidence': confidence,
                'risk_level': risk,
                'trade_type': 'options',
                'technical_indicators': tech_indicators
            }
            
        except Exception as e:
            logger.error(f"Error in optimized options analysis: {e}")
            return None
    
    def analyze_penny_stocks_optimized(self, tickers: List[str], top_n: int = 20) -> List[Dict]:
        """
        Optimized penny stock scanner
        
        Args:
            tickers: List of ticker symbols
            top_n: Number of top results to return
            
        Returns:
            List of opportunity dictionaries sorted by score
        """
        start_time = time.perf_counter()
        
        if self.enable_parallel and len(tickers) > 10:
            results = self.parallel_scanner.parallel_penny_stock_scan(tickers)
            self.performance_stats['parallel_scans'] += 1
        else:
            results = []
            for ticker in tickers:
                try:
                    data = self._get_ticker_data_cached(ticker)
                    if data:
                        analysis = self._analyze_penny_stock_optimized(data)
                        if analysis and analysis.get('score', 0) > 0:
                            results.append(analysis)
                except Exception as e:
                    logger.error(f"Error analyzing {ticker}: {e}")
        
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        execution_time = time.perf_counter() - start_time
        self.performance_stats['scans_executed'] += 1
        self.performance_stats['total_execution_time'] += execution_time
        
        logger.info(f"Penny stock scan completed: {len(results)} opportunities in {execution_time:.2f}s")
        
        return results[:top_n]
    
    def _analyze_penny_stock_optimized(self, ticker_data: Dict) -> Optional[Dict]:
        """Optimized penny stock analysis"""
        try:
            ticker = ticker_data['ticker']
            price = ticker_data['price']
            change_pct = ticker_data['change_pct']
            volume_ratio = ticker_data['volume_ratio']
            
            # Filter penny stocks only
            if price > 5.0:
                return None
            
            score = 40.0  # Base score
            reasons = []
            
            # Ultra-low price bonus
            if price < 1.0:
                score += 15
                reasons.append(f"Ultra-low price (${price:.3f})")
            elif price < 2.0:
                score += 10
            
            # Volume spike analysis
            if volume_ratio > 3.0:
                score += 35
                reasons.append(f"Massive volume ({volume_ratio:.1f}x)")
            elif volume_ratio > 2.0:
                score += 25
                reasons.append(f"Volume spike ({volume_ratio:.1f}x)")
            elif volume_ratio > 1.5:
                score += 15
            
            # Price movement analysis
            if abs(change_pct) > 10:
                score += 30
                reasons.append(f"Major move ({change_pct:+.1f}%)")
            elif abs(change_pct) > 5:
                score += 20
                reasons.append(f"Big move ({change_pct:+.1f}%)")
            elif abs(change_pct) > 2:
                score += 10
            
            # Sector bonus (if available)
            sector = ticker_data.get('sector')
            if sector in ['Healthcare', 'Biotechnology', 'Energy']:
                score += 5
                reasons.append(f"{sector} sector")
            
            return {
                'ticker': ticker,
                'score': round(score, 1),
                'price': price,
                'change_pct': change_pct,
                'volume_ratio': volume_ratio,
                'volume': ticker_data.get('volume', 0),
                'reason': " | ".join(reasons) if reasons else "Penny stock opportunity",
                'confidence': "MEDIUM" if score >= 60 else "LOW",
                'risk_level': 'HIGH',
                'trade_type': 'penny_stock',
                'sector': sector
            }
            
        except Exception as e:
            logger.error(f"Error in penny stock analysis: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        cache_instance = get_cache_instance()
        cache_stats = cache_instance.get_stats() if self.enable_caching else {}
        
        # Calculate average execution time
        avg_execution_time = 0
        if self.performance_stats['scans_executed'] > 0:
            avg_execution_time = self.performance_stats['total_execution_time'] / self.performance_stats['scans_executed']
        
        scanner_stats = {}
        if hasattr(self, 'parallel_scanner'):
            scanner_stats = self.parallel_scanner.get_stats()
        
        yfinance_stats = self.yfinance_client.get_stats()
        
        return {
            'scanner_performance': {
                'scans_executed': self.performance_stats['scans_executed'],
                'avg_execution_time_s': round(avg_execution_time, 2),
                'parallel_scans': self.performance_stats['parallel_scans'],
                'talib_calculations': self.performance_stats['talib_calculations'],
                'optimizations_enabled': {
                    'caching': self.enable_caching,
                    'parallel': self.enable_parallel, 
                    'talib': self.enable_talib
                }
            },
            'cache_performance': cache_stats,
            'parallel_scanner_performance': scanner_stats,
            'yfinance_performance': yfinance_stats
        }
    
    def run_performance_test(self, test_tickers: List[str] = None) -> Dict[str, Any]:
        """
        Run performance test comparing optimized vs standard scanning
        
        Args:
            test_tickers: Optional list of tickers for testing
            
        Returns:
            Performance comparison results
        """
        if test_tickers is None:
            test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'] * 4  # 20 tickers
        
        logger.info(f"Running performance test with {len(test_tickers)} tickers...")
        
        # Test optimized scanning
        start_time = time.perf_counter()
        optimized_results = self.analyze_options_opportunities_optimized(test_tickers, top_n=10)
        optimized_time = time.perf_counter() - start_time
        
        # Estimate sequential time (0.5s per ticker)
        estimated_sequential_time = len(test_tickers) * 0.5
        speedup = estimated_sequential_time / optimized_time
        
        return {
            'test_config': {
                'tickers_tested': len(test_tickers),
                'optimizations_enabled': {
                    'parallel': self.enable_parallel,
                    'caching': self.enable_caching,
                    'talib': self.enable_talib
                }
            },
            'performance_results': {
                'optimized_time_s': round(optimized_time, 2),
                'estimated_sequential_time_s': round(estimated_sequential_time, 2),
                'speedup_factor': round(speedup, 2),
                'opportunities_found': len(optimized_results),
                'time_per_ticker_ms': round((optimized_time / len(test_tickers)) * 1000, 2)
            },
            'detailed_metrics': self.get_performance_metrics()
        }


# Convenience function for easy integration
def create_optimized_scanner(config: Dict[str, Any] = None) -> OptimizedScanner:
    """
    Create optimized scanner with configuration
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured OptimizedScanner instance
    """
    if config is None:
        config = {
            'max_workers': 10,
            'enable_caching': True,
            'enable_parallel': True,
            'enable_talib': True
        }
    
    return OptimizedScanner(**config)

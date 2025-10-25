"""Preset scan filters for rapid opportunity identification."""

import logging
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from analyzers.comprehensive import ComprehensiveAnalyzer
from services.alert_system import get_alert_system, SetupDetector

logger = logging.getLogger(__name__)


class ScanPreset(Enum):
    """Predefined scan presets"""
    TRIPLE_THREAT = "TRIPLE_THREAT"  # Reclaim + Aligned + Strong RS
    EMA_RECLAIM = "EMA_RECLAIM"  # EMA reclaim only
    TIMEFRAME_ALIGNED = "TIMEFRAME_ALIGNED"  # Multi-timeframe alignment
    SECTOR_LEADERS = "SECTOR_LEADERS"  # Strong relative strength
    DEMARKER_PULLBACK = "DEMARKER_PULLBACK"  # DeMarker oversold in uptrend
    FIBONACCI_SETUP = "FIBONACCI_SETUP"  # Fibonacci targets detected
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"  # Confidence score >= 85
    POWER_ZONE = "POWER_ZONE"  # EMA power zone active
    OPTIONS_PREMIUM_SELL = "OPTIONS_PREMIUM_SELL"  # High IV + power zone
    OPTIONS_DIRECTIONAL = "OPTIONS_DIRECTIONAL"  # Low IV + reclaim/alignment


@dataclass
class ScanResult:
    """Result from a scan"""
    ticker: str
    preset: ScanPreset
    confidence_score: float
    analysis: any  # StockAnalysis object
    match_reasons: List[str]
    priority_score: float  # 0-100, higher = better
    
    def __str__(self) -> str:
        return f"{self.ticker} [{self.preset.value}] Confidence: {self.confidence_score:.0f} | {', '.join(self.match_reasons)}"


class PresetScanner:
    """Scanner with preset filters for different trading setups"""
    
    def __init__(self, alert_system=None, my_tickers_only=False, ticker_manager=None):
        """Initialize scanner
        
        Args:
            alert_system: Alert system to use
            my_tickers_only: If True, only generate alerts for tickers in 'My Tickers'
            ticker_manager: TickerManager instance for filtering
        """
        self.alert_system = alert_system or get_alert_system()
        self.setup_detector = SetupDetector(self.alert_system, my_tickers_only, ticker_manager)
    
    def scan(self, tickers: List[str], preset: ScanPreset, 
             trading_style: str = "SWING_TRADE",
             generate_alerts: bool = True) -> List[ScanResult]:
        """
        Scan tickers using a preset filter.
        
        Args:
            tickers: List of ticker symbols to scan
            preset: Scan preset to use
            trading_style: Trading style for analysis
            generate_alerts: Whether to generate alerts for matches
        
        Returns:
            List of ScanResult objects, sorted by priority
        """
        logger.info(f"Scanning {len(tickers)} tickers with {preset.value} preset")
        
        results = []
        
        for ticker in tickers:
            try:
                # Analyze stock
                analysis = ComprehensiveAnalyzer.analyze_stock(ticker, trading_style)
                
                if not analysis:
                    continue
                
                # Check if matches preset criteria
                match, reasons, priority = self._check_preset(analysis, preset)
                
                if match:
                    result = ScanResult(
                        ticker=ticker,
                        preset=preset,
                        confidence_score=analysis.confidence_score,
                        analysis=analysis,
                        match_reasons=reasons,
                        priority_score=priority
                    )
                    results.append(result)
                    
                    # Generate alerts if enabled
                    if generate_alerts:
                        self.setup_detector.analyze_for_alerts(analysis)
                    
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        # Sort by priority score (descending)
        results.sort(key=lambda x: x.priority_score, reverse=True)
        
        logger.info(f"Found {len(results)} matches for {preset.value}")
        return results
    
    def _check_preset(self, analysis, preset: ScanPreset) -> tuple[bool, List[str], float]:
        """
        Check if analysis matches preset criteria.
        Returns (match, reasons, priority_score)
        """
        reasons = []
        priority = analysis.confidence_score  # Base priority on confidence
        
        if preset == ScanPreset.TRIPLE_THREAT:
            # Reclaim + Aligned + Strong RS
            has_reclaim = analysis.ema_reclaim
            has_aligned = analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned', False)
            has_strong_rs = analysis.sector_rs and analysis.sector_rs.get('rs_score', 0) > 60
            
            match = has_reclaim and has_aligned and has_strong_rs
            
            if has_reclaim:
                reasons.append("EMA Reclaim")
                priority += 10
            if has_aligned:
                reasons.append(f"Aligned {analysis.timeframe_alignment.get('alignment_score', 0):.0f}%")
                priority += 10
            if has_strong_rs:
                reasons.append(f"RS {analysis.sector_rs.get('rs_score', 0):.1f}")
                priority += 10
            
            return match, reasons, priority
        
        elif preset == ScanPreset.EMA_RECLAIM:
            # EMA reclaim only
            match = analysis.ema_reclaim
            if match:
                reasons.append("EMA Reclaim Confirmed")
                priority += 20
                if analysis.demarker and analysis.demarker <= 0.30:
                    reasons.append(f"DeMarker {analysis.demarker:.2f}")
                    priority += 5
            return match, reasons, priority
        
        elif preset == ScanPreset.TIMEFRAME_ALIGNED:
            # Multi-timeframe alignment
            match = analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned', False)
            if match:
                alignment_score = analysis.timeframe_alignment.get('alignment_score', 0)
                reasons.append(f"Timeframes Aligned {alignment_score:.0f}%")
                priority += alignment_score / 10
                
                # Bonus for power zone
                if analysis.ema_power_zone:
                    reasons.append("Power Zone")
                    priority += 10
            return match, reasons, priority
        
        elif preset == ScanPreset.SECTOR_LEADERS:
            # Strong relative strength
            match = analysis.sector_rs and analysis.sector_rs.get('rs_score', 0) > 70
            if match:
                rs_score = analysis.sector_rs.get('rs_score', 0)
                reasons.append(f"Sector Leader RS {rs_score:.1f}")
                priority += (rs_score - 50) / 2  # Add bonus for strong RS
                
                sector = analysis.sector_rs.get('sector', '')
                if sector:
                    reasons.append(f"Sector: {sector}")
            return match, reasons, priority
        
        elif preset == ScanPreset.DEMARKER_PULLBACK:
            # DeMarker oversold in uptrend
            match = (analysis.demarker is not None and 
                    analysis.demarker <= 0.30 and 
                    "UPTREND" in analysis.trend)
            if match:
                reasons.append(f"DeMarker {analysis.demarker:.2f} in {analysis.trend}")
                priority += 15
                
                if analysis.ema_power_zone:
                    reasons.append("Power Zone")
                    priority += 10
                if analysis.fib_targets:
                    reasons.append("Fib Targets")
                    priority += 5
            return match, reasons, priority
        
        elif preset == ScanPreset.FIBONACCI_SETUP:
            # Fibonacci targets detected
            match = analysis.fib_targets is not None
            if match:
                reasons.append("Fibonacci A-B-C Detected")
                priority += 10
                
                if analysis.demarker and analysis.demarker <= 0.35:
                    reasons.append(f"Entry Zone (DM {analysis.demarker:.2f})")
                    priority += 10
                if analysis.ema_power_zone:
                    reasons.append("Power Zone")
                    priority += 5
            return match, reasons, priority
        
        elif preset == ScanPreset.HIGH_CONFIDENCE:
            # High confidence score
            match = analysis.confidence_score >= 85
            if match:
                reasons.append(f"High Confidence {analysis.confidence_score:.0f}")
                
                if analysis.ema_reclaim:
                    reasons.append("Reclaim")
                if analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned'):
                    reasons.append("Aligned")
                if analysis.sector_rs and analysis.sector_rs.get('rs_score', 0) > 60:
                    reasons.append("Strong RS")
            return match, reasons, priority
        
        elif preset == ScanPreset.POWER_ZONE:
            # EMA power zone active
            match = analysis.ema_power_zone
            if match:
                reasons.append("Power Zone Active (8>21)")
                priority += 15
                
                if "UPTREND" in analysis.trend:
                    reasons.append(analysis.trend)
                    priority += 5
                if analysis.demarker and analysis.demarker <= 0.35:
                    reasons.append(f"Pullback (DM {analysis.demarker:.2f})")
                    priority += 10
            return match, reasons, priority
        
        elif preset == ScanPreset.OPTIONS_PREMIUM_SELL:
            # High IV + power zone (for selling premium)
            match = analysis.iv_rank > 60 and analysis.ema_power_zone
            if match:
                reasons.append(f"High IV {analysis.iv_rank:.1f}")
                reasons.append("Power Zone")
                priority += 20
                
                if analysis.ema_reclaim:
                    reasons.append("Reclaim → Sell Puts")
                    priority += 15
            return match, reasons, priority
        
        elif preset == ScanPreset.OPTIONS_DIRECTIONAL:
            # Low IV + reclaim/alignment (for buying options)
            match = (analysis.iv_rank < 40 and 
                    (analysis.ema_reclaim or 
                     (analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned'))))
            if match:
                reasons.append(f"Low IV {analysis.iv_rank:.1f}")
                
                if analysis.ema_reclaim:
                    reasons.append("Reclaim → Buy Calls")
                    priority += 15
                if analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned'):
                    reasons.append("Aligned")
                    priority += 10
                if analysis.fib_targets:
                    reasons.append("Fib Targets")
                    priority += 5
            return match, reasons, priority
        
        return False, [], 0
    
    def scan_all_presets(self, tickers: List[str], trading_style: str = "SWING_TRADE") -> Dict[ScanPreset, List[ScanResult]]:
        """
        Scan with all presets and return results grouped by preset.
        
        Args:
            tickers: List of ticker symbols
            trading_style: Trading style for analysis
        
        Returns:
            Dictionary mapping preset to scan results
        """
        results = {}
        
        for preset in ScanPreset:
            preset_results = self.scan(tickers, preset, trading_style, generate_alerts=False)
            if preset_results:
                results[preset] = preset_results
        
        return results
    
    def get_top_opportunities(self, tickers: List[str], top_n: int = 10, 
                             trading_style: str = "SWING_TRADE") -> List[ScanResult]:
        """
        Get top N opportunities across all presets.
        
        Args:
            tickers: List of ticker symbols
            top_n: Number of top results to return
            trading_style: Trading style for analysis
        
        Returns:
            List of top ScanResult objects
        """
        all_results = []
        
        # Scan with all presets
        preset_results = self.scan_all_presets(tickers, trading_style)
        
        # Collect all unique results (deduplicate by ticker)
        seen_tickers = set()
        for preset, results in preset_results.items():
            for result in results:
                if result.ticker not in seen_tickers:
                    all_results.append(result)
                    seen_tickers.add(result.ticker)
        
        # Sort by priority and return top N
        all_results.sort(key=lambda x: x.priority_score, reverse=True)
        return all_results[:top_n]
    
    def print_results(self, results: List[ScanResult], show_details: bool = False):
        """Print scan results in a formatted table"""
        if not results:
            print("No matches found.")
            return
        
        print(f"\n{'='*100}")
        print(f"SCAN RESULTS: {results[0].preset.value if results else ''}")
        print(f"Found {len(results)} matches")
        print(f"{'='*100}\n")
        
        print(f"{'Rank':<6} {'Ticker':<8} {'Conf':<6} {'Priority':<9} {'Reasons'}")
        print(f"{'-'*6} {'-'*8} {'-'*6} {'-'*9} {'-'*50}")
        
        for i, result in enumerate(results, 1):
            reasons_str = ', '.join(result.match_reasons)
            print(f"{i:<6} {result.ticker:<8} {result.confidence_score:<6.0f} {result.priority_score:<9.1f} {reasons_str}")
            
            if show_details and result.analysis:
                a = result.analysis
                print(f"       Price: ${a.price:.2f} | Trend: {a.trend} | RSI: {a.rsi:.1f} | IV Rank: {a.iv_rank:.1f}")
                if a.fib_targets:
                    t1 = a.fib_targets.get('T1_1272')
                    t2 = a.fib_targets.get('T2_1618')
                    print(f"       Fib Targets: T1=${t1:.2f} T2=${t2:.2f}")
                print()
        
        print(f"{'='*100}\n")


# Convenience functions for common watchlists

def get_sp500_tickers() -> List[str]:
    """Get S&P 500 tickers (simplified list)"""
    # In production, fetch from Wikipedia or use a package like 'yfinance'
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
        "UNH", "JNJ", "V", "XOM", "WMT", "JPM", "MA", "PG", "CVX", "HD",
        "LLY", "MRK", "ABBV", "KO", "PEP", "AVGO", "COST", "TMO", "MCD",
        "CSCO", "ABT", "ACN", "DHR", "WFC", "VZ", "DIS", "ADBE", "CMCSA",
        "PFE", "NKE", "CRM", "NFLX", "NEE", "BMY", "TXN", "UPS", "PM",
        "INTC", "AMD", "QCOM", "T", "LOW", "ORCL"
    ]


def get_nasdaq100_tickers() -> List[str]:
    """Get NASDAQ 100 tickers (simplified list)"""
    return [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
        "AVGO", "COST", "ASML", "PEP", "AZN", "CSCO", "TMUS", "ADBE",
        "NFLX", "AMD", "CMCSA", "INTC", "TXN", "QCOM", "INTU", "AMGN",
        "HON", "AMAT", "SBUX", "ISRG", "BKNG", "ADI", "MDLZ", "VRTX",
        "GILD", "REGN", "PANW", "ADP", "LRCX", "MU", "PYPL", "SNPS",
        "KLAC", "MELI", "CDNS", "MNST", "ORLY", "MAR", "FTNT", "ABNB",
        "CTAS", "MRVL"
    ]


def get_high_volume_tech() -> List[str]:
    """Get high volume tech stocks for day trading"""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
        "NFLX", "INTC", "QCOM", "ADBE", "CRM", "AVGO", "TXN", "ORCL",
        "CSCO", "PYPL", "COIN", "SQ", "SNOW", "PLTR", "RBLX", "U"
    ]

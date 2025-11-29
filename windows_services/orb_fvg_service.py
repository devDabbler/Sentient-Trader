"""
ORB FVG Scanner - Windows Service Wrapper

Runs the 15-Minute Opening Range Breakout + Fair Value Gap strategy scanner
as a Windows service with:
- Auto-start on system boot
- Windows Event Log integration
- Service management via Services panel
- Automatic restart on failure

Strategy Overview:
- 15-minute Opening Range Breakout (9:30-9:45 AM ET)
- Fair Value Gap confirmation for entry
- Clean structure, tight risk, once-per-day trading
- Discord alerts for instant notification

Installation:
    python windows_services\\orb_fvg_service.py install

Start/Stop:
    python windows_services\\orb_fvg_service.py start
    python windows_services\\orb_fvg_service.py stop

Removal:
    python windows_services\\orb_fvg_service.py remove
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.windows_service_base import WindowsServiceBase, install_service, check_pywin32_installed
from windows_services.runners.service_config_loader import save_analysis_results, get_pending_analysis_requests, mark_analysis_complete
from loguru import logger


class ORBFVGService(WindowsServiceBase):
    """Windows service wrapper for ORB FVG Scanner"""
    
    _svc_name_ = "SentientORBFVG"
    _svc_display_name_ = "Sentient ORB FVG Scanner"
    _svc_description_ = (
        "Scans stocks for 15-Minute Opening Range Breakout + Fair Value Gap setups. "
        "Monitors during optimal trading window (9:30 AM - 12:30 PM ET). "
        "Sends Discord alerts for high-confidence setups."
    )
    
    def run_service(self):
        """Main service loop - runs the ORB FVG scanner continuously"""
        logger.info("=" * 70)
        logger.info("📊 ORB FVG SCANNER SERVICE STARTED")
        logger.info("=" * 70)
        
        try:
            from services.orb_fvg_strategy import ORBFVGStrategy
            from services.orb_fvg_alerts import ORBFVGAlertManager
            from services.ticker_manager import TickerManager
            
            # Initialize components
            strategy = ORBFVGStrategy(
                orb_minutes=15,
                min_gap_pct=0.2,
                min_volume_ratio=1.5,
                target_rr=2.0
            )
            
            alert_manager = ORBFVGAlertManager()
            ticker_manager = TickerManager()
            
            # Get scan interval from environment or default
            scan_interval = int(os.getenv('ORB_FVG_SCAN_INTERVAL', '60'))  # Default 1 minute
            
            logger.info("ORB FVG Scanner initialized")
            logger.info(f"Scan interval: {scan_interval} seconds")
            logger.info("Trading window: 9:30 AM - 12:30 PM ET")
            
            eastern = pytz.timezone('US/Eastern')
            
            while self.running and not self.is_stop_requested():
                try:
                    current_time = datetime.now(eastern)
                    current_hour = current_time.hour
                    current_minute = current_time.minute
                    
                    # Check for on-demand analysis requests
                    try:
                        pending_requests = get_pending_analysis_requests()
                        # Filter for ORB FVG requests or custom requests with stock tickers (heuristic: no '/')
                        my_requests = []
                        for r in pending_requests:
                            preset = r.get('preset', '')
                            tickers = r.get('tickers', [])
                            is_orb = preset == 'orb_fvg_scan'
                            is_custom_stock = preset == 'custom' and tickers and not any('/' in t for t in tickers)
                            
                            if (is_orb or is_custom_stock) and tickers:
                                my_requests.append(r)
                        
                        for req in my_requests:
                            logger.info(f"📋 Processing on-demand request {req['id']}...")
                            req_tickers = req.get('tickers', [])
                            req_results = []
                            
                            for ticker in req_tickers:
                                try:
                                    # Force analyze even if outside hours? Maybe better to try.
                                    # strategy.analyze_ticker checks hours.
                                    signal = strategy.analyze_ticker(ticker, current_time)
                                    if signal:
                                        req_results.append({
                                            'ticker': signal.symbol,
                                            'signal': f"{signal.signal_type} (ORB+FVG)",
                                            'confidence': int(signal.confidence),
                                            'price': signal.entry_price,
                                            'change_24h': 0.0,
                                            'volume_24h': signal.volume_ratio,
                                            'timestamp': datetime.now().isoformat()
                                        })
                                except Exception as e:
                                    logger.error(f"Error analyzing {ticker}: {e}")
                            
                            save_analysis_results('ORB FVG (On-Demand)', req_results)
                            mark_analysis_complete(req['id'], {'count': len(req_results)})
                            logger.info(f"   ✅ Request {req['id']} complete")
                            
                    except Exception as e:
                        logger.error(f"Error processing analysis queue: {e}")
                    
                    # Only scan during market hours (9:30 AM - 4:00 PM ET)
                    market_open = (current_hour == 9 and current_minute >= 30) or (current_hour > 9)
                    market_close = current_hour < 16
                    
                    if market_open and market_close:
                        # Get watchlist tickers
                        tickers = ticker_manager.get_my_tickers()
                        
                        if not tickers:
                            # Use default ORB FVG tickers if no watchlist
                            tickers = [
                                'SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT',
                                'META', 'AMZN', 'GOOGL', 'NFLX', 'COIN', 'PLTR', 'SOFI'
                            ]
                        
                        logger.info(f"Scanning {len(tickers)} tickers for ORB+FVG setups...")
                        
                        signals_found = 0
                        results_for_panel = []
                        
                        for ticker in tickers:
                            try:
                                signal = strategy.analyze_ticker(ticker, current_time)
                                
                                if signal:
                                    # Add to results for panel
                                    results_for_panel.append({
                                        'ticker': signal.symbol,
                                        'signal': f"{signal.signal_type} (ORB+FVG)",
                                        'confidence': int(signal.confidence),
                                        'price': signal.entry_price,
                                        'change_24h': 0.0,
                                        'volume_24h': signal.volume_ratio,
                                        'timestamp': datetime.now().isoformat()
                                    })
                                    
                                    if signal.confidence >= 70:
                                        signals_found += 1
                                        logger.info(
                                            f"🎯 SIGNAL: {signal.symbol} {signal.signal_type} "
                                            f"@ ${signal.entry_price:.2f} (Conf: {signal.confidence:.1f}%)"
                                        )
                                        
                                        # Send alert
                                        alert_manager.send_signal_alert(signal)
                                    
                            except Exception as e:
                                logger.error(f"Error analyzing {ticker}: {e}")
                                continue
                        
                        # Save results to control panel
                        if results_for_panel:
                            try:
                                save_analysis_results('ORB FVG Scanner', results_for_panel)
                                logger.debug(f"💾 Saved {len(results_for_panel)} results to control panel")
                            except Exception as e:
                                logger.warning(f"Failed to save results to control panel: {e}")
                        
                        if signals_found > 0:
                            logger.info(f"✅ Found {signals_found} ORB+FVG signals this scan")
                        else:
                            logger.debug("No signals this scan")
                    else:
                        # Outside market hours
                        if current_hour < 9 or (current_hour == 9 and current_minute < 30):
                            logger.info(f"Pre-market ({current_time.strftime('%H:%M')} ET) - waiting for market open...")
                        else:
                            logger.info(f"After hours ({current_time.strftime('%H:%M')} ET) - market closed")
                    
                    # Wait for next scan
                    time.sleep(scan_interval)
                    
                except Exception as e:
                    logger.error(f"Error in scan cycle: {e}", exc_info=True)
                    time.sleep(60)  # Wait 1 min on error
            
            logger.info("Service stop requested, shutting down...")
            
        except Exception as e:
            logger.error(f"Fatal error in service: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    if not check_pywin32_installed():
        sys.exit(1)
    
    install_service(ORBFVGService)

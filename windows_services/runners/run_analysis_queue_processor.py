"""
Analysis Queue Processor Service

Polls analysis_requests.json for pending requests and runs AI analysis.
Results are saved to analysis_results.json for display in Control Panel.

This enables mobile workflow:
1. User queues analysis from Control Panel or Discord
2. This service picks it up and runs AI analysis
3. Results appear in Control Panel for review
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from windows_services.runners.service_config_loader import (
    get_pending_analysis_requests,
    mark_analysis_complete,
    save_analysis_results,
    ANALYSIS_PRESETS
)

# Configure logging
LOG_FILE = PROJECT_ROOT / "logs" / "analysis_queue_processor.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(str(LOG_FILE), rotation="10 MB", retention="7 days", level="DEBUG")

# Files
ANALYSIS_REQUESTS_FILE = PROJECT_ROOT / "data" / "analysis_requests.json"
ANALYSIS_RESULTS_FILE = PROJECT_ROOT / "data" / "analysis_results.json"

# Check interval
CHECK_INTERVAL = 10  # seconds


def run_crypto_analysis(ticker: str, mode: str = "standard") -> dict:
    """Run crypto AI analysis and return results"""
    try:
        import os
        from clients.kraken_client import KrakenClient
        from services.ai_entry_assistant import AIEntryAssistant
        
        # Normalize ticker format
        if "/" not in ticker:
            ticker = f"{ticker}/USD"
        
        # Get Kraken credentials from environment
        api_key = os.getenv("KRAKEN_API_KEY", "")
        api_secret = os.getenv("KRAKEN_API_SECRET", "")
        
        kraken = KrakenClient(api_key=api_key, api_secret=api_secret)
        assistant = AIEntryAssistant(kraken_client=kraken)
        
        # Run analysis
        analysis = assistant.analyze_entry(
            pair=ticker,
            side="BUY",  # Default to BUY for discovery
            position_size=1000,
            risk_pct=2.0,
            take_profit_pct=6.0
        )
        
        if analysis:
            return {
                "ticker": ticker,
                "action": analysis.action,
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning,
                "urgency": analysis.urgency,
                "current_price": analysis.current_price,
                "suggested_entry": analysis.suggested_entry,
                "suggested_stop": analysis.suggested_stop,
                "suggested_target": analysis.suggested_target,
                "risk_reward": analysis.risk_reward_ratio,
                "technical_score": analysis.technical_score,
                "trend_score": analysis.trend_score,
                "timing_score": analysis.timing_score,
                "analysis_time": datetime.now().isoformat(),
                "mode": mode,
                "asset_type": "crypto"
            }
        else:
            return {
                "ticker": ticker,
                "action": "ERROR",
                "confidence": 0,
                "reasoning": "Analysis returned no result",
                "analysis_time": datetime.now().isoformat(),
                "mode": mode,
                "asset_type": "crypto"
            }
            
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        return {
            "ticker": ticker,
            "action": "ERROR",
            "confidence": 0,
            "reasoning": str(e),
            "analysis_time": datetime.now().isoformat(),
            "mode": mode,
            "asset_type": "crypto"
        }


def run_stock_analysis(ticker: str, mode: str = "standard") -> dict:
    """Run stock AI analysis and return results"""
    try:
        import os
        from services.ai_stock_entry_assistant import AIStockEntryAssistant
        
        # Try to get a broker client - Tradier is simplest
        broker_client = None
        try:
            tradier_token = os.getenv("TRADIER_ACCESS_TOKEN")
            if tradier_token:
                # Check if Tradier client exists before importing
                try:
                    from src.integrations.tradier_client import TradierClient
                    broker_client = TradierClient(access_token=tradier_token)
                except ImportError:
                    logger.warning("Tradier client not available - using fallback")
        except Exception as e:
            logger.warning(f"Could not initialize Tradier client: {e}")
        
        # If no broker, use a minimal mock for analysis-only mode
        if not broker_client:
            class MinimalBrokerMock:
                def get_quote(self, symbol):
                    # Use yfinance as fallback for price data
                    try:
                        import yfinance as yf
                        ticker_obj = yf.Ticker(symbol)
                        hist = ticker_obj.history(period="1d")
                        if not hist.empty:
                            return {"last": hist['Close'].iloc[-1]}
                    except:
                        pass
                    return {"last": 0}
            broker_client = MinimalBrokerMock()
        
        assistant = AIStockEntryAssistant(broker_client=broker_client)
        
        analysis = assistant.analyze_entry(
            symbol=ticker.upper(),
            side="BUY",
            position_size=1000,
            risk_pct=2.0,
            take_profit_pct=6.0
        )
        
        if analysis:
            return {
                "ticker": ticker.upper(),
                "action": analysis.action,
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning,
                "urgency": getattr(analysis, 'urgency', 'MEDIUM'),
                "current_price": analysis.current_price,
                "suggested_entry": analysis.suggested_entry,
                "suggested_stop": analysis.suggested_stop,
                "suggested_target": analysis.suggested_target,
                "risk_reward": getattr(analysis, 'risk_reward_ratio', 0),
                "technical_score": getattr(analysis, 'technical_score', 0),
                "analysis_time": datetime.now().isoformat(),
                "mode": mode,
                "asset_type": "stock"
            }
        else:
            return {
                "ticker": ticker.upper(),
                "action": "ERROR",
                "confidence": 0,
                "reasoning": "Analysis returned no result",
                "analysis_time": datetime.now().isoformat(),
                "mode": mode,
                "asset_type": "stock"
            }
            
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        return {
            "ticker": ticker.upper(),
            "action": "ERROR",
            "confidence": 0,
            "reasoning": str(e),
            "analysis_time": datetime.now().isoformat(),
            "mode": mode,
            "asset_type": "stock"
        }


def process_request(request: dict) -> list:
    """Process a single analysis request and return results"""
    request_id = request.get("id")
    tickers = request.get("tickers", [])
    asset_type = request.get("asset_type", "crypto")
    mode = request.get("analysis_mode", "standard")
    
    logger.info(f"Processing request {request_id}: {len(tickers)} {asset_type} tickers, mode={mode}")
    
    results = []
    
    for ticker in tickers:
        logger.info(f"  Analyzing {ticker}...")
        
        if asset_type == "crypto":
            result = run_crypto_analysis(ticker, mode)
        else:
            result = run_stock_analysis(ticker, mode)
        
        results.append(result)
        
        # Log result
        action = result.get("action", "UNKNOWN")
        conf = result.get("confidence", 0)
        logger.info(f"  {ticker}: {action} ({conf:.0f}% confidence)")
    
    return results


def save_results_to_file(results: list, request: dict):
    """Save analysis results to file for Control Panel display"""
    try:
        ANALYSIS_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        if ANALYSIS_RESULTS_FILE.exists():
            with open(ANALYSIS_RESULTS_FILE, 'r') as f:
                all_results = json.load(f)
        
        # Add new results under a key
        result_key = f"queue_{request.get('id', 'unknown')}"
        all_results[result_key] = {
            "results": results,
            "request": request,
            "completed": datetime.now().isoformat(),
            "count": len(results)
        }
        
        # Also update "latest" for quick access
        all_results["latest"] = {
            "results": results,
            "request": request,
            "completed": datetime.now().isoformat(),
            "count": len(results)
        }
        
        # Keep only last 10 result sets
        keys = [k for k in all_results.keys() if k.startswith("queue_")]
        if len(keys) > 10:
            for old_key in sorted(keys)[:-10]:
                del all_results[old_key]
        
        with open(ANALYSIS_RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved {len(results)} results to {ANALYSIS_RESULTS_FILE}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def mark_request_complete(request_id: str):
    """Mark a request as complete in the queue file"""
    try:
        if not ANALYSIS_REQUESTS_FILE.exists():
            return
        
        with open(ANALYSIS_REQUESTS_FILE, 'r') as f:
            requests = json.load(f)
        
        for req in requests:
            if req.get("id") == request_id:
                req["status"] = "complete"
                req["completed"] = datetime.now().isoformat()
                break
        
        with open(ANALYSIS_REQUESTS_FILE, 'w') as f:
            json.dump(requests, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error marking request complete: {e}")


def main_loop():
    """Main processing loop"""
    logger.info("=" * 50)
    logger.info("Analysis Queue Processor Started")
    logger.info(f"Checking every {CHECK_INTERVAL} seconds")
    logger.info("=" * 50)
    
    while True:
        try:
            # Check for pending requests
            if not ANALYSIS_REQUESTS_FILE.exists():
                time.sleep(CHECK_INTERVAL)
                continue
            
            with open(ANALYSIS_REQUESTS_FILE, 'r') as f:
                requests = json.load(f)
            
            # Find pending requests
            pending = [r for r in requests if r.get("status") == "pending"]
            
            if pending:
                logger.info(f"Found {len(pending)} pending request(s)")
                
                for request in pending:
                    request_id = request.get("id")
                    
                    # Process the request
                    results = process_request(request)
                    
                    # Save results
                    save_results_to_file(results, request)
                    
                    # Mark complete
                    mark_request_complete(request_id)
                    
                    logger.info(f"✅ Completed request {request_id}")
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down...")

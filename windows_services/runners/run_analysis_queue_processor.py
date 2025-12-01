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

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from windows_services.runners.service_config_loader import (
    get_pending_analysis_requests,
    mark_analysis_complete,
    save_analysis_results,
    ANALYSIS_PRESETS
)

# Discord notifications (optional)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
ENABLE_DISCORD_ALERTS = os.getenv("ENABLE_DISCORD_ALERTS", "false").lower() == "true"

# Configure logging
LOG_FILE = PROJECT_ROOT / "logs" / "analysis_queue_processor.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger.remove()
# Force unbuffered/immediate output for systemd compatibility
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(str(LOG_FILE), rotation="10 MB", retention="7 days", level="DEBUG")

# Force stdout to be unbuffered
import sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# Files
ANALYSIS_REQUESTS_FILE = PROJECT_ROOT / "data" / "analysis_requests.json"
ANALYSIS_RESULTS_FILE = PROJECT_ROOT / "data" / "analysis_results.json"

# Check interval
CHECK_INTERVAL = 10  # seconds

# LLM Analysis Mode:
# - "primary" = use AI_ANALYZER_MODEL only (default)
# - "compare" = run both Ollama and OpenRouter, return both results  
# - "fallback" = use primary, fall back to secondary if primary times out
ANALYSIS_LLM_MODE = os.getenv("ANALYSIS_LLM_MODE", "primary")
ANALYSIS_TIMEOUT = int(os.getenv("ANALYSIS_TIMEOUT", "120"))  # seconds


def create_llm_analyzer(provider: str = None, model: str = None):
    """Create LLM analyzer with specific provider/model"""
    try:
        from services.llm_strategy_analyzer import LLMStrategyAnalyzer
        
        if provider and model:
            return LLMStrategyAnalyzer(provider=provider, model=model)
        else:
            return LLMStrategyAnalyzer()  # Uses AI_ANALYZER_MODEL from env
    except Exception as e:
        logger.warning(f"Failed to create LLM analyzer ({provider}/{model}): {e}")
        return None


def run_single_analysis(ticker: str, kraken, llm_analyzer, mode: str, llm_name: str = "primary") -> dict:
    """Run analysis with a specific LLM analyzer"""
    from services.ai_entry_assistant import AIEntryAssistant
    
    try:
        logger.info(f"   [{llm_name}] Creating AIEntryAssistant...")
        assistant = AIEntryAssistant(kraken_client=kraken, llm_analyzer=llm_analyzer)
        
        logger.info(f"   [{llm_name}] Running analyze_entry...")
        analysis = assistant.analyze_entry(
            pair=ticker,
            side="BUY",
            position_size=1000,
            risk_pct=2.0,
            take_profit_pct=6.0
        )
        logger.info(f"   [{llm_name}] Analysis complete")
        
        if analysis:
            return {
                "ticker": ticker,
                "llm": llm_name,
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
                "llm": llm_name,
                "action": "NO_RESULT",
                "confidence": 0,
                "reasoning": "Analysis returned no result",
                "analysis_time": datetime.now().isoformat(),
                "mode": mode,
                "asset_type": "crypto"
            }
    except Exception as e:
        logger.error(f"Error in {llm_name} analysis for {ticker}: {e}")
        return {
            "ticker": ticker,
            "llm": llm_name,
            "action": "ERROR",
            "confidence": 0,
            "reasoning": str(e),
            "analysis_time": datetime.now().isoformat(),
            "mode": mode,
            "asset_type": "crypto"
        }


def run_crypto_analysis(ticker: str, mode: str = "standard") -> dict:
    """Run crypto AI analysis and return results"""
    logger.info(f"   [1] Starting analysis for {ticker}")  # INFO so it always shows
    try:
        import os
        logger.info(f"   [2] os imported")
        from clients.kraken_client import KrakenClient
        logger.info(f"   [3] KrakenClient imported")
        
        # Normalize ticker format
        if "/" not in ticker:
            ticker = f"{ticker}/USD"
        logger.info(f"   [4] Ticker normalized: {ticker}")
        
        # Get Kraken credentials from environment
        api_key = os.getenv("KRAKEN_API_KEY", "")
        api_secret = os.getenv("KRAKEN_API_SECRET", "")
        logger.info(f"   [5] Kraken credentials loaded (key present: {bool(api_key)})")
        
        kraken = KrakenClient(api_key=api_key, api_secret=api_secret)
        logger.info(f"   [6] Kraken client created")
        
        # Check analysis mode
        llm_mode = ANALYSIS_LLM_MODE
        logger.info(f"   LLM mode: {llm_mode}, timeout: {ANALYSIS_TIMEOUT}s")
        
        if llm_mode == "compare":
            # Run BOTH Ollama and OpenRouter, return comparison
            results = {"ticker": ticker, "mode": mode, "comparison": []}
            
            # Ollama analysis
            logger.info(f"   Running Ollama analysis...")
            ollama_llm = create_llm_analyzer(provider="ollama", model="qwen2.5:7b")
            if ollama_llm:
                ollama_result = run_single_analysis(ticker, kraken, ollama_llm, mode, "ollama")
                results["comparison"].append(ollama_result)
                logger.info(f"   Ollama: {ollama_result.get('action')} ({ollama_result.get('confidence', 0):.0f}%)")
            
            # OpenRouter analysis  
            logger.info(f"   Running OpenRouter analysis...")
            openrouter_llm = create_llm_analyzer(provider="openrouter", model="google/gemini-2.0-flash-exp:free")
            if openrouter_llm:
                openrouter_result = run_single_analysis(ticker, kraken, openrouter_llm, mode, "openrouter")
                results["comparison"].append(openrouter_result)
                logger.info(f"   OpenRouter: {openrouter_result.get('action')} ({openrouter_result.get('confidence', 0):.0f}%)")
            
            # Use the best result as primary
            valid_results = [r for r in results["comparison"] if r.get("action") not in ["ERROR", "TIMEOUT", "NO_RESULT"]]
            if valid_results:
                best = max(valid_results, key=lambda x: x.get("confidence", 0))
                results.update(best)
                results["comparison_note"] = f"Best of {len(valid_results)} LLMs"
            else:
                results["action"] = "ERROR"
                results["confidence"] = 0
                results["reasoning"] = "All LLMs failed or timed out"
            
            results["analysis_time"] = datetime.now().isoformat()
            results["asset_type"] = "crypto"
            return results
            
        elif llm_mode == "fallback":
            # Try primary, fall back to secondary on failure
            primary_llm = create_llm_analyzer()  # Uses AI_ANALYZER_MODEL
            if primary_llm:
                result = run_single_analysis(ticker, kraken, primary_llm, mode, "primary")
                if result.get("action") not in ["ERROR", "TIMEOUT", "NO_RESULT"]:
                    return result
                logger.warning(f"   Primary LLM failed, trying fallback...")
            
            # Fallback to OpenRouter
            fallback_llm = create_llm_analyzer(provider="openrouter", model="google/gemini-2.0-flash-exp:free")
            if fallback_llm:
                result = run_single_analysis(ticker, kraken, fallback_llm, mode, "fallback")
                result["fallback_used"] = True
                return result
            
            return {
                "ticker": ticker,
                "action": "ERROR",
                "confidence": 0,
                "reasoning": "All LLMs failed",
                "analysis_time": datetime.now().isoformat(),
                "mode": mode,
                "asset_type": "crypto"
            }
        
        else:
            # Primary mode - use AI_ANALYZER_MODEL only
            llm_analyzer = create_llm_analyzer()
            result = run_single_analysis(ticker, kraken, llm_analyzer, mode, "primary")
            return result
            
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
        from services.llm_strategy_analyzer import LLMStrategyAnalyzer
        
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
        
        # Create LLM analyzer for AI decisions
        llm_analyzer = None
        try:
            llm_analyzer = LLMStrategyAnalyzer()
        except Exception as e:
            logger.warning(f"Could not initialize LLM analyzer: {e}")
        
        assistant = AIStockEntryAssistant(broker_client=broker_client, llm_analyzer=llm_analyzer)
        
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
            try:
                with open(ANALYSIS_RESULTS_FILE, 'r') as f:
                    all_results = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing results, starting fresh: {e}")
                all_results = {}
        
        # Add new results under a key
        result_key = f"queue_{request.get('id', 'unknown')}"
        now_iso = datetime.now().isoformat()
        
        all_results[result_key] = {
            "results": results,
            "request": request,
            "completed": now_iso,
            "updated": now_iso,  # Also use "updated" for consistency
            "count": len(results)
        }
        
        # Also update "queue_latest" for quick access to latest queue result
        all_results["queue_latest"] = {
            "results": results,
            "request": request,
            "completed": now_iso,
            "updated": now_iso,
            "count": len(results)
        }
        
        # Keep only last 20 result sets to avoid file bloat
        keys = [k for k in all_results.keys() if k.startswith("queue_") and k != "queue_latest"]
        if len(keys) > 20:
            for old_key in sorted(keys)[:-20]:
                logger.info(f"Removing old result set: {old_key}")
                del all_results[old_key]
        
        with open(ANALYSIS_RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"✅ Saved {len(results)} results from request {result_key}")
        logger.info(f"   Tickers: {', '.join([r.get('ticker', '?') for r in results[:5]])}" + 
                   (f"... and {len(results)-5} more" if len(results) > 5 else ""))
        
    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)


def send_discord_notification(results: list, request: dict):
    """Send analysis results to Discord webhook"""
    if not ENABLE_DISCORD_ALERTS or not DISCORD_WEBHOOK_URL:
        return
    
    try:
        import requests as req_lib
        
        # Build summary
        asset_type = request.get("asset_type", "crypto").upper()
        mode = request.get("analysis_mode", "standard")
        completed_time = datetime.now().strftime("%H:%M:%S")
        
        # Count results by action
        actions = {}
        for r in results:
            action = r.get("action", "UNKNOWN")
            actions[action] = actions.get(action, 0) + 1
        
        # Build message
        summary = f"**Analysis Complete** ({completed_time})\n"
        summary += f"Asset Type: {asset_type} | Mode: {mode}\n"
        summary += f"Analyzed: {len(results)} tickers\n\n"
        summary += "**Results:**\n"
        for action, count in sorted(actions.items()):
            summary += f"• {action}: {count}\n"
        
        # Get top result by confidence if available
        top_result = None
        for r in results:
            if r.get("confidence", 0) > 0:
                if not top_result or r.get("confidence", 0) > top_result.get("confidence", 0):
                    top_result = r
        
        if top_result:
            summary += f"\n**Top Opportunity:**\n"
            summary += f"{top_result.get('ticker', '?')}: {top_result.get('action', '?')} ({top_result.get('confidence', 0):.0f}%)\n"
            if top_result.get('reasoning'):
                summary += f"*{top_result.get('reasoning')[:100]}...*"
        
        payload = {
            "content": summary,
            "username": "Analysis Queue Processor"
        }
        
        response = req_lib.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code in [200, 204]:
            logger.info(f"✅ Discord notification sent")
        else:
            logger.warning(f"Discord notification failed: {response.status_code}")
            
    except Exception as e:
        logger.warning(f"Could not send Discord notification: {e}")


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
    logger.info("=" * 60)
    logger.info("🚀 ANALYSIS QUEUE PROCESSOR SERVICE STARTED")
    logger.info(f"📅 Started at: {datetime.now().isoformat()}")
    logger.info(f"⏱️  Checking for new requests every {CHECK_INTERVAL} seconds")
    logger.info(f"📊 LLM Mode: {ANALYSIS_LLM_MODE}")
    logger.info(f"⏰ Analysis Timeout: {ANALYSIS_TIMEOUT}s")
    logger.info("=" * 60)
    
    loop_count = 0
    processed_count = 0
    error_count = 0
    
    while True:
        loop_count += 1
        try:
            # Check for pending requests
            if not ANALYSIS_REQUESTS_FILE.exists():
                # Periodic logging every 60 loops (10 min at 10s intervals)
                if loop_count % 60 == 0:
                    logger.info(f"⏳ Waiting for requests... (checked {loop_count} times, processed {processed_count})")
                time.sleep(CHECK_INTERVAL)
                continue
            
            with open(ANALYSIS_REQUESTS_FILE, 'r') as f:
                requests = json.load(f)
            
            # Find pending requests
            pending = [r for r in requests if r.get("status") == "pending"]
            
            if pending:
                logger.info(f"📋 Found {len(pending)} pending request(s)")
                
                for request in pending:
                    request_id = request.get("id")
                    tickers = request.get("tickers", [])
                    asset_type = request.get("asset_type", "crypto")
                    
                    logger.info(f"{'='*60}")
                    logger.info(f"▶️  PROCESSING REQUEST: {request_id}")
                    logger.info(f"    Asset Type: {asset_type.upper()}")
                    logger.info(f"    Tickers: {', '.join(tickers[:5])}" + 
                               (f" ... and {len(tickers)-5} more" if len(tickers) > 5 else ""))
                    logger.info(f"{'='*60}")
                    
                    # Process the request
                    results = process_request(request)
                    
                    # Save results
                    save_results_to_file(results, request)
                    
                    # Send Discord notification
                    send_discord_notification(results, request)
                    
                    # Mark complete
                    mark_request_complete(request_id)
                    
                    logger.info(f"✅ COMPLETED request {request_id}")
                    logger.info(f"   📊 Results: {len(results)} tickers analyzed")
                    processed_count += 1
            
            else:
                # Periodic logging every 30 loops (5 min at 10s intervals)
                if loop_count % 30 == 0:
                    logger.info(f"⏳ No pending requests... (checked {loop_count} times, processed {processed_count}, errors {error_count})")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            error_count += 1
            time.sleep(CHECK_INTERVAL * 2)  # Wait longer on error
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            error_count += 1
            time.sleep(CHECK_INTERVAL * 2)  # Wait longer on error
        
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Analysis Queue Processor initializing...")
    logger.info("=" * 60)
    
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("🛑 Received keyboard interrupt signal")
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"💥 FATAL ERROR: {e}", exc_info=True)
        exit(1)

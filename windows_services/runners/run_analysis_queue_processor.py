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

# ============================================================
# SINGLETON CHECK - Prevent multiple instances
# ============================================================
from utils.process_lock import ensure_single_instance

force_restart = '--force' in sys.argv or '-f' in sys.argv
process_lock = ensure_single_instance("analysis_queue_processor", force=force_restart)
# ============================================================

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from windows_services.runners.service_config_loader import (
    get_pending_analysis_requests,
    mark_analysis_complete,
    save_analysis_results,
    ANALYSIS_PRESETS
)

# Discord notifications - ENABLED by default for analysis results
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
# Enable Discord alerts for analysis results - users want to see results in Discord
ENABLE_DISCORD_ALERTS = os.getenv("ANALYSIS_QUEUE_DISCORD_ALERTS", "true").lower() in ("true", "1", "yes")

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
                "trend_score": getattr(analysis, 'trend_score', 0),
                "timing_score": getattr(analysis, 'timing_score', 0),
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


def format_action_name(action: str) -> str:
    """Convert action codes to human-readable format (WAIT_FOR_PULLBACK → Wait for Pullback)"""
    action_labels = {
        'ENTER_NOW': 'Enter Now',
        'WAIT_FOR_PULLBACK': 'Wait for Pullback',
        'DO_NOT_ENTER': 'Do Not Enter',
        'WAIT_FOR_BREAKOUT': 'Wait for Breakout',
        'WAIT_FOR_CONFIRMATION': 'Wait for Confirmation',
        'LONG': 'Long',
        'SHORT': 'Short',
        'BUY': 'Buy',
        'SELL': 'Sell',
        'BULLISH': 'Bullish',
        'BEARISH': 'Bearish',
        'WAIT': 'Wait',
        'HOLD': 'Hold',
        'NO_RESULT': 'No Result',
        'ERROR': 'Error',
        'UNKNOWN': 'Unknown'
    }
    return action_labels.get(action, action.replace('_', ' ').title())


def get_strategy_context(mode: str, action: str, confidence: float) -> str:
    """Generate strategy context explaining why this signal was chosen"""
    
    # Mode descriptions with strategy details
    mode_strategies = {
        "standard": {
            "name": "Standard Analysis",
            "strategies_tested": ["Trend Following"],
            "timeframes": ["Primary timeframe"],
            "description": "Single strategy evaluation on primary timeframe"
        },
        "multi": {
            "name": "Multi-Strategy Analysis", 
            "strategies_tested": ["Trend Following", "Mean Reversion", "Momentum", "Breakout"],
            "timeframes": ["Short-term", "Medium-term"],
            "directions": ["Long", "Short"],
            "description": "Tests multiple strategies across directions"
        },
        "multi_config": {
            "name": "Multi-Strategy Analysis",
            "strategies_tested": ["Trend Following", "Mean Reversion", "Momentum", "Breakout"],
            "timeframes": ["Short-term", "Medium-term"],
            "directions": ["Long", "Short"],
            "description": "Tests multiple strategies across directions"
        },
        "ultimate": {
            "name": "Ultimate Analysis",
            "strategies_tested": ["Trend Following", "Mean Reversion", "Momentum", "Breakout", "Scalping", "Swing"],
            "timeframes": ["1H", "4H", "Daily", "Weekly"],
            "directions": ["Long", "Short", "Neutral"],
            "description": "Exhaustive analysis of ALL strategy combinations"
        }
    }
    
    mode_info = mode_strategies.get(mode, mode_strategies["standard"])
    
    # Build context based on action
    if action in ['ENTER_NOW', 'LONG', 'BUY', 'BULLISH']:
        signal_reason = f"**Why {format_action_name(action)}?** This signal emerged as the strongest across {len(mode_info.get('strategies_tested', []))} strategies tested."
        if confidence >= 85:
            signal_reason += " High confidence indicates strong alignment across multiple indicators."
        elif confidence >= 70:
            signal_reason += " Good confirmation from technical and trend analysis."
    elif action in ['SHORT', 'SELL', 'BEARISH']:
        signal_reason = f"**Why {format_action_name(action)}?** Bearish signals dominated across strategy tests."
    elif action in ['WAIT_FOR_PULLBACK']:
        signal_reason = "**Why Wait for Pullback?** Current entry is suboptimal. A pullback to support would improve risk/reward ratio significantly."
    elif action in ['WAIT_FOR_BREAKOUT']:
        signal_reason = "**Why Wait for Breakout?** Price is consolidating. Entry on confirmed breakout offers better probability."
    elif action in ['DO_NOT_ENTER', 'WAIT']:
        signal_reason = "**Why Wait/Skip?** No clear edge detected. Conflicting signals or unfavorable market conditions."
    else:
        signal_reason = ""
    
    return signal_reason


def send_discord_notification(results: list, request: dict):
    """Send analysis results to Discord webhook using rich embeds"""
    if not DISCORD_WEBHOOK_URL:
        logger.debug("Discord webhook not configured, skipping notification")
        return
    
    if not ENABLE_DISCORD_ALERTS:
        logger.debug("Discord alerts disabled, skipping notification")
        return
    
    try:
        import requests as req_lib
        
        # Build metadata
        asset_type = request.get("asset_type", "crypto").upper()
        mode = request.get("analysis_mode", "standard")
        completed_time = datetime.now().strftime("%H:%M:%S")
        
        # Mode labels for display
        mode_display = {
            "standard": "🔬 Standard (Single Strategy)",
            "multi": "🎯 Multi-Strategy (Long/Short + Timeframes)",
            "multi_config": "🎯 Multi-Strategy (Long/Short + Timeframes)",
            "ultimate": "🚀 Ultimate (ALL Strategies + Timeframes)"
        }.get(mode, mode)
        
        # Count results by action (formatted)
        actions = {}
        for r in results:
            action = format_action_name(r.get("action", "UNKNOWN"))
            actions[action] = actions.get(action, 0) + 1
        
        # Create embeds for each result (Discord allows multiple embeds)
        embeds = []
        
        # Header embed with summary
        summary_text = f"**Asset Type:** {asset_type}\n"
        summary_text += f"**Mode:** {mode_display}\n"
        summary_text += f"**Analyzed:** {len(results)} ticker(s)\n\n"
        summary_text += "**Signal Summary:**\n"
        for action, count in sorted(actions.items()):
            summary_text += f"• {action}: {count}\n"
        
        header_embed = {
            "title": "📊 Analysis Complete",
            "description": summary_text,
            "color": 3447003,  # Blue
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": f"Completed at {completed_time}"}
        }
        embeds.append(header_embed)
        
        # Create embed for each result with full details
        for result in results:
            ticker = result.get('ticker', '?')
            action = result.get('action', 'UNKNOWN')
            action_display = format_action_name(action)
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', 'No analysis provided')
            result_mode = result.get('mode', mode)
            
            # Determine trade direction and color based on action
            if action in ['LONG', 'BUY', 'BULLISH', 'ENTER_NOW']:
                color = 65280  # Green
                emoji = "🟢"
                trade_direction = "📈 Long / Buy"
            elif action in ['SHORT', 'SELL', 'BEARISH']:
                color = 16711680  # Red
                emoji = "🔴"
                trade_direction = "📉 Short / Sell"
            elif action in ['DO_NOT_ENTER', 'WAIT', 'WAIT_FOR_PULLBACK', 'WAIT_FOR_BREAKOUT', 'WAIT_FOR_CONFIRMATION']:
                color = 16776960  # Yellow
                emoji = "🟡"
                trade_direction = "⏸️ Wait / No Trade"
            else:
                color = 8421504  # Gray
                emoji = "⚪"
                trade_direction = "❓ Undetermined"
            
            # Mode descriptions
            mode_labels = {
                "standard": "🔬 Standard",
                "multi": "🎯 Multi-Strategy",
                "multi_config": "🎯 Multi-Strategy",
                "ultimate": "🚀 Ultimate"
            }
            mode_label = mode_labels.get(result_mode, f"📊 {result_mode}")
            
            # Build fields with available data
            fields = [
                {"name": "Signal", "value": f"{emoji} **{action_display}**", "inline": True},
                {"name": "Confidence", "value": f"**{confidence:.0f}%**" if isinstance(confidence, (int, float)) else str(confidence), "inline": True},
                {"name": "Trade Direction", "value": trade_direction, "inline": True}
            ]
            
            # Add urgency
            if result.get('urgency'):
                urgency = result.get('urgency')
                urgency_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}.get(urgency, "")
                fields.append({"name": "Urgency", "value": f"{urgency_emoji} {urgency}", "inline": True})
            
            # Add analysis mode
            fields.append({"name": "Analysis Mode", "value": mode_label, "inline": True})
            
            # Add entry/stop/target for actionable signals
            if result.get('suggested_entry') is not None:
                entry = result.get('suggested_entry')
                entry_str = f"${entry:.4f}" if isinstance(entry, (int, float)) else str(entry)
                # Show entry as "suggested" for wait signals
                entry_label = "Entry Point" if action in ['ENTER_NOW', 'LONG', 'SHORT', 'BUY', 'SELL'] else "Target Entry"
                fields.append({"name": entry_label, "value": entry_str, "inline": True})
            
            if result.get('suggested_stop') is not None:
                stop = result.get('suggested_stop')
                stop_str = f"${stop:.4f}" if isinstance(stop, (int, float)) else str(stop)
                fields.append({"name": "Stop Loss", "value": stop_str, "inline": True})
            
            if result.get('suggested_target') is not None:
                target = result.get('suggested_target')
                target_str = f"${target:.4f}" if isinstance(target, (int, float)) else str(target)
                fields.append({"name": "Take Profit", "value": target_str, "inline": True})
            
            if result.get('risk_reward') is not None:
                rr = result.get('risk_reward')
                rr_str = f"**{rr:.2f}:1**" if isinstance(rr, (int, float)) else str(rr)
                fields.append({"name": "Risk/Reward", "value": rr_str, "inline": True})
            
            # Add technical scores if available
            scores = []
            if result.get('technical_score') is not None:
                scores.append(f"Tech: {result.get('technical_score'):.0f}")
            if result.get('trend_score') is not None:
                scores.append(f"Trend: {result.get('trend_score'):.0f}")
            if result.get('timing_score') is not None:
                scores.append(f"Timing: {result.get('timing_score'):.0f}")
            
            if scores:
                fields.append({"name": "Scores", "value": " | ".join(scores), "inline": True})
            
            # Add strategy context explaining WHY this signal
            strategy_context = get_strategy_context(result_mode, action, confidence)
            if strategy_context:
                fields.append({"name": "Strategy Insight", "value": strategy_context, "inline": False})
            
            # Add reasoning as main field (Discord truncates at 2048 chars per field)
            reasoning_text = reasoning[:900] if len(reasoning) > 900 else reasoning
            fields.append({"name": "📝 Analysis", "value": reasoning_text, "inline": False})
            
            result_embed = {
                "title": f"{emoji} {ticker}",
                "color": color,
                "fields": fields
            }
            embeds.append(result_embed)
        
        # Send via webhook (Discord allows up to 10 embeds per message)
        payload = {
            "username": "🤖 Analysis Queue Processor",
            "embeds": embeds[:10]  # Limit to 10 embeds per message
        }
        
        response = req_lib.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code in [200, 204]:
            logger.info(f"✅ Discord notification sent ({len(embeds)} embeds)")
        else:
            logger.warning(f"Discord notification failed: {response.status_code}")
            
    except Exception as e:
        logger.warning(f"Could not send Discord notification: {e}")


def mark_request_complete(request_id: str):
    """Mark a request as complete in the queue file with proper file flushing"""
    try:
        if not ANALYSIS_REQUESTS_FILE.exists():
            logger.warning(f"Analysis requests file doesn't exist, can't mark {request_id} complete")
            return False
        
        with open(ANALYSIS_REQUESTS_FILE, 'r') as f:
            requests = json.load(f)
        
        found = False
        for req in requests:
            if req.get("id") == request_id:
                req["status"] = "complete"
                req["completed"] = datetime.now().isoformat()
                found = True
                break
        
        if not found:
            logger.warning(f"Request {request_id} not found in analysis queue")
            return False
        
        # Write back to file with explicit flush to ensure Windows file system persists
        import os
        with open(ANALYSIS_REQUESTS_FILE, 'w') as f:
            json.dump(requests, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Force OS to write to disk
        
        logger.info(f"✅ Marked request {request_id} as complete (file synced)")
        
        # Verify the write by re-reading (debug for Windows file system issues)
        with open(ANALYSIS_REQUESTS_FILE, 'r') as f:
            verify = json.load(f)
        for req in verify:
            if req.get("id") == request_id:
                if req.get("status") == "complete":
                    logger.debug(f"✅ Verified: request {request_id} is complete in file")
                else:
                    logger.error(f"❌ VERIFICATION FAILED: request {request_id} still shows status={req.get('status')}")
                break
        
        return True
            
    except Exception as e:
        logger.error(f"Error marking request complete: {e}", exc_info=True)
        return False


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
            
            # Find pending requests (skip completed, failed, etc)
            pending = [r for r in requests if r.get("status", "pending") == "pending"]
            
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
    finally:
        process_lock.release()

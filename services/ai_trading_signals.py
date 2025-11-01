"""
AI Trading Signals Generator
Uses LLM to analyze multiple data sources and generate buy/sell signals
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """AI-generated trading signal"""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size: int
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH
    time_horizon: str  # SCALP, DAY_TRADE, SWING
    technical_score: float
    sentiment_score: float
    news_score: float
    social_score: float
    discord_score: float = 0.0
    timestamp: str = ""


class AITradingSignalGenerator:
    """Generate trading signals using AI analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI signal generator
        
        Args:
            api_key: API key (optional if in env)
        """
        self.api_key = api_key
        model = os.getenv('AI_TRADING_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')
        logger.info(f"AI Trading Signal Generator initialized with OpenRouter using model: {model}")
    
    def generate_signal(
        self,
        symbol: str,
        technical_data: Dict,
        news_data: List[Dict],
        sentiment_data: Dict,
        social_data: Optional[Dict] = None,
        discord_data: Optional[Dict] = None,
        account_balance: float = 10000.0,
        risk_tolerance: str = "MEDIUM",
        current_positions: List[str] = None
    ) -> Optional[TradingSignal]:
        """
        Generate AI trading signal
        
        Args:
            symbol: Stock symbol
            technical_data: Technical indicators (RSI, MACD, volume, etc.)
            news_data: Recent news articles
            sentiment_data: News sentiment scores
            social_data: Reddit/social media sentiment
            discord_data: Discord trading alerts from monitored channels
            account_balance: Available capital
            risk_tolerance: LOW, MEDIUM, HIGH
            current_positions: List of symbols currently owned
            
        Returns:
            TradingSignal or None
        """
        try:
            # Build comprehensive analysis prompt
            prompt = self._build_analysis_prompt(
                symbol=symbol,
                technical_data=technical_data,
                news_data=news_data,
                sentiment_data=sentiment_data,
                social_data=social_data,
                discord_data=discord_data,
                account_balance=account_balance,
                risk_tolerance=risk_tolerance,
                current_positions=current_positions or []
            )
            
            # Get AI analysis
            response = self._call_openrouter(prompt)
            
            if response:
                # Parse AI response into trading signal
                signal = self._parse_signal(response, symbol)
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating AI signal for {symbol}: {e}")
            return None
    
    def _build_analysis_prompt(
        self,
        symbol: str,
        technical_data: Dict,
        news_data: List[Dict],
        sentiment_data: Dict,
        social_data: Optional[Dict],
        discord_data: Optional[Dict],
        account_balance: float,
        risk_tolerance: str,
        current_positions: List[str]
    ) -> str:
        """Build comprehensive analysis prompt for LLM"""
        
        # Format technical data
        tech_summary = f"""
**Technical Analysis for {symbol}:**
- Current Price: ${technical_data.get('price', 0):.2f}
- Price Change: {technical_data.get('change_pct', 0):+.2f}%
- RSI: {technical_data.get('rsi', 50):.1f} (Oversold<30, Overbought>70)
- MACD Signal: {technical_data.get('macd_signal', 'NEUTRAL')}
- Trend: {technical_data.get('trend', 'SIDEWAYS')}
- Volume: {technical_data.get('volume', 0):,} (Avg: {technical_data.get('avg_volume', 0):,})
- Support: ${technical_data.get('support', 0):.2f}
- Resistance: ${technical_data.get('resistance', 0):.2f}
- IV Rank: {technical_data.get('iv_rank', 50):.1f}%
"""
        
        # Format news sentiment
        news_summary = "**Recent News:**\n"
        for idx, article in enumerate(news_data[:5], 1):
            news_summary += f"{idx}. {article.get('title', 'N/A')[:80]}...\n"
        
        sentiment_summary = f"""
**Sentiment Analysis:**
- Overall Sentiment: {sentiment_data.get('score', 0):.2f} (-1=Bearish, +1=Bullish)
- Positive Signals: {len([s for s in sentiment_data.get('signals', []) if '✅' in s])}
- Negative Signals: {len([s for s in sentiment_data.get('signals', []) if '⚠️' in s])}
"""
        
        # Format social sentiment
        social_summary = ""
        if social_data:
            social_summary = f"""
**Social Media Sentiment (Reddit/Twitter):**
- Reddit Mentions: {social_data.get('reddit_mentions', 0)}
- Bullish Posts: {social_data.get('bullish_count', 0)}
- Bearish Posts: {social_data.get('bearish_count', 0)}
- Trending Score: {social_data.get('trending_score', 0):.1f}/10
- Overall Social Sentiment: {social_data.get('sentiment', 'NEUTRAL')}
"""
        
        # Format Discord alerts
        discord_summary = ""
        if discord_data:
            alerts = discord_data.get('alerts', [])
            if alerts:
                discord_summary = f"""
**Discord Trading Alerts (Professional Traders):**
- Total Alerts (24h): {len(alerts)}
- Recent Alerts:"""
                for idx, alert in enumerate(alerts[:5], 1):
                    alert_type = alert.get('alert_type', 'ALERT')
                    price = alert.get('price')
                    target = alert.get('target')
                    premium = '🔒 Premium' if alert.get('premium_channel') else '📢 Free'
                    
                    price_str = f"@ ${price:.2f}" if price else ""
                    target_str = f"→ ${target:.2f}" if target else ""
                    
                    discord_summary += f"\n  {idx}. {alert_type} {price_str} {target_str} ({premium})"
                
                # Summary stats
                entry_alerts = len([a for a in alerts if a.get('alert_type') == 'ENTRY'])
                runner_alerts = len([a for a in alerts if a.get('alert_type') == 'RUNNER'])
                exit_alerts = len([a for a in alerts if a.get('alert_type') == 'EXIT'])
                
                discord_summary += f"""
- Entry Signals: {entry_alerts}
- Runner Alerts: {runner_alerts}
- Exit Signals: {exit_alerts}
- Discord Sentiment: {'BULLISH' if entry_alerts > exit_alerts else 'BEARISH' if exit_alerts > entry_alerts else 'NEUTRAL'}
"""
        
        # Check if we currently own this stock
        own_this_stock = symbol in current_positions
        position_status = f"✅ YOU CURRENTLY OWN {symbol}" if own_this_stock else f"❌ YOU DO NOT OWN {symbol}"
        
        # Build complete prompt
        prompt = f"""You are an expert day trader and scalper analyzing {symbol} for a potential trade.

{tech_summary}

{news_summary}

{sentiment_summary}

{social_summary}

{discord_summary}

**Trading Parameters:**
- Account Balance: ${account_balance:,.2f}
- Risk Tolerance: {risk_tolerance}
- Trading Style: Day Trading / Scalping (intraday exits)

**CRITICAL - Current Position Status:**
{position_status}
- Current Holdings: {', '.join(current_positions) if current_positions else 'None'}

**Your Task:**
Analyze all the data above and provide a trading recommendation. Consider:
1. Technical momentum and trend
2. News catalysts and sentiment
3. Social media buzz and retail sentiment
4. Discord alerts from professional traders (if available)
5. Risk/reward ratio
6. Entry/exit timing for day trading

**Respond in this EXACT JSON format:**
{{
    "signal": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "entry_price": price to enter (current price or better),
    "target_price": price target for day trade,
    "stop_loss": stop loss price,
    "position_size": number of shares (based on account size and risk),
    "reasoning": "2-3 sentence explanation of your decision",
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "time_horizon": "SCALP" or "DAY_TRADE" or "SWING",
    "technical_score": 0-100,
    "sentiment_score": 0-100,
    "news_score": 0-100,
    "social_score": 0-100,
    "discord_score": 0-100,
    "key_factors": ["factor 1", "factor 2", "factor 3"]
}}

**Important:**
- Only recommend BUY/SELL if confidence > 60%
- **CRITICAL: Only recommend SELL if you CURRENTLY OWN the stock ({own_this_stock})**
- If you DON'T own it and it looks weak, recommend HOLD (avoid it), NOT SELL
- If you DON'T own it and it looks strong, recommend BUY
- If you DO own it and it looks weak or hit target, recommend SELL
- For day trading, recommend closing position same day
- Position size should not exceed 20% of account balance
- Stop loss should be 1-3% below entry for day trades
- Return ONLY valid JSON, no other text
- **CRITICAL JSON FORMATTING**: Keep all string values on a SINGLE LINE. Do NOT use newlines inside strings. Keep reasoning brief and on one line.
"""
        
        return prompt
    
    
    def _call_openrouter(self, prompt: str) -> Optional[str]:
        """Call OpenRouter API"""
        try:
            import os
            import requests
            
            api_key = self.api_key or os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                logger.error("No OpenRouter API key found")
                return None
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": os.getenv('AI_TRADING_MODEL', 'meta-llama/llama-3.1-8b-instruct:free'),
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert day trader. Always respond with valid JSON only. CRITICAL: All string values must be on a single line with no newline characters. Keep text concise."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                logger.error(f"OpenRouter API error: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")
            return None
    
    
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to handle malformed responses from LLM"""
        import re
        
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        
        # Fix common LLM issues: Replace actual newlines inside strings with \n
        # This is tricky - we need to find strings and replace newlines only inside them
        def replace_newlines_in_strings(match):
            """Replace actual newlines with \n inside JSON string values"""
            string_content = match.group(1)
            # Replace newlines with space (safer than \n which could also break)
            cleaned = string_content.replace('\n', ' ').replace('\r', ' ')
            # Also remove other control characters
            cleaned = re.sub(r'[\x00-\x1F\x7F]', ' ', cleaned)
            # Collapse multiple spaces
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return f'"{cleaned}"'
        
        # Match string values in JSON (between quotes, handling escaped quotes)
        json_str = re.sub(r'"((?:[^"\\]|\\.)*)(?:")', replace_newlines_in_strings, json_str)
        
        return json_str.strip()
    
    def _parse_signal(self, response: str, symbol: str) -> Optional[TradingSignal]:
        """Parse LLM response into TradingSignal"""
        try:
            # Extract JSON from response (might have extra text)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response
            
            # Clean the JSON string to handle malformed responses
            json_str = self._clean_json_string(json_str)
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Create TradingSignal
            signal = TradingSignal(
                symbol=symbol,
                signal=data.get('signal', 'HOLD'),
                confidence=float(data.get('confidence', 0)),
                entry_price=data.get('entry_price'),
                target_price=data.get('target_price'),
                stop_loss=data.get('stop_loss'),
                position_size=int(data.get('position_size', 0)),
                reasoning=data.get('reasoning', ''),
                risk_level=data.get('risk_level', 'MEDIUM'),
                time_horizon=data.get('time_horizon', 'DAY_TRADE'),
                technical_score=float(data.get('technical_score', 0)),
                sentiment_score=float(data.get('sentiment_score', 0)),
                news_score=float(data.get('news_score', 0)),
                social_score=float(data.get('social_score', 0)),
                discord_score=float(data.get('discord_score', 0)),
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            logger.info(f"Generated signal for {symbol}: {signal.signal} (confidence: {signal.confidence}%)")
            return signal
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for {symbol}: {e}")
            logger.error(f"Response was: {response[:500]}...")
            
            # Try fallback extraction using regex
            try:
                import re
                signal_match = re.search(r'"signal":\s*"(BUY|SELL|HOLD)"', response, re.IGNORECASE)
                confidence_match = re.search(r'"confidence":\s*(\d+)', response)
                
                if signal_match and confidence_match:
                    fallback_signal = TradingSignal(
                        symbol=symbol,
                        signal=signal_match.group(1).upper(),
                        confidence=float(confidence_match.group(1)),
                        entry_price=None,
                        target_price=None,
                        stop_loss=None,
                        position_size=0,
                        reasoning="Fallback parsing due to malformed JSON response",
                        risk_level="HIGH",
                        time_horizon="DAY_TRADE",
                        technical_score=0,
                        sentiment_score=0,
                        news_score=0,
                        social_score=0,
                        discord_score=0,
                        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    )
                    logger.warning(f"Using fallback parsing for {symbol}: {fallback_signal.signal} ({fallback_signal.confidence}%)")
                    return fallback_signal
            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed: {fallback_error}")
            
            return None
        
        except Exception as e:
            logger.error(f"Error parsing signal: {e}")
            logger.error(f"Response was: {response[:500]}...")
            return None
    
    def batch_analyze(
        self,
        symbols: List[str],
        technical_data_dict: Dict[str, Dict],
        news_data_dict: Dict[str, List[Dict]],
        sentiment_data_dict: Dict[str, Dict],
        account_balance: float = 10000.0,
        risk_tolerance: str = "MEDIUM",
        current_positions: List[str] = None
    ) -> List[TradingSignal]:
        """
        Analyze multiple symbols and return signals
        
        Args:
            symbols: List of stock symbols
            technical_data_dict: Dict of symbol -> technical data
            news_data_dict: Dict of symbol -> news data
            sentiment_data_dict: Dict of symbol -> sentiment data
            account_balance: Available capital
            risk_tolerance: Risk level
            current_positions: List of symbols currently owned (for position-aware signals)
            
        Returns:
            List of TradingSignal objects
        """
        if current_positions is None:
            current_positions = []
            
        signals = []
        
        for symbol in symbols:
            try:
                signal = self.generate_signal(
                    symbol=symbol,
                    technical_data=technical_data_dict.get(symbol, {}),
                    news_data=news_data_dict.get(symbol, []),
                    sentiment_data=sentiment_data_dict.get(symbol, {}),
                    account_balance=account_balance,
                    risk_tolerance=risk_tolerance,
                    current_positions=current_positions
                )
                
                # Filter out invalid signals
                if signal and signal.signal != "HOLD" and signal.confidence >= 60:
                    # CRITICAL: Only allow SELL if we own the stock
                    if signal.signal == "SELL" and symbol not in current_positions:
                        logger.info(f"🚫 Filtering out SELL signal for {symbol} - not in current positions")
                        continue
                    signals.append(signal)
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals


def create_ai_signal_generator(api_key: Optional[str] = None) -> AITradingSignalGenerator:
    """
    Create AI signal generator
    
    Args:
        api_key: Optional API key
        
    Returns:
        AITradingSignalGenerator instance
    """
    return AITradingSignalGenerator(api_key=api_key)

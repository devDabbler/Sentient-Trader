"""
Crypto Trading Signals Generator
AI-powered signal generation specifically adapted for cryptocurrency markets

Key Differences from Stock Trading:
- 24/7 market (no trading hours restrictions)
- Higher volatility (3-5x typical stock movements)
- Social sentiment has greater impact
- On-chain metrics available
- Different fee structure (maker/taker)
- Instant settlement
"""

from loguru import logger
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import requests
from .llm_helper import get_llm_helper



@dataclass
class CryptoTradingSignal:
    """AI-generated cryptocurrency trading signal"""
    symbol: str  # e.g., 'BTC/USD'
    base_asset: str  # e.g., 'BTC'
    quote_asset: str  # e.g., 'USD'
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size_usd: float
    position_size_crypto: float
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    time_horizon: str  # SCALP, MOMENTUM, SWING, HOLD
    
    # Crypto-specific scores
    technical_score: float
    sentiment_score: float
    social_score: float
    volatility_score: float
    liquidity_score: float
    onchain_score: float = 0.0
    
    # Market conditions
    market_regime: str = "SIDEWAYS"  # BULL, BEAR, SIDEWAYS, VOLATILE
    fear_greed_index: Optional[int] = None
    
    timestamp: str = ""
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio"""
        if not self.entry_price or not self.target_price or not self.stop_loss:
            return 0.0
        
        reward = abs(self.target_price - self.entry_price)
        risk = abs(self.entry_price - self.stop_loss)
        
        if risk == 0:
            return 0.0
        
        return reward / risk
    
    @property
    def expected_return_pct(self) -> float:
        """Expected return percentage"""
        if not self.entry_price or not self.target_price:
            return 0.0
        
        return ((self.target_price - self.entry_price) / self.entry_price) * 100


class CryptoTradingSignalGenerator:
    """Generate trading signals specifically for cryptocurrency markets with hybrid LLM support"""
    
    def __init__(self, api_key: Optional[str] = None, config=None, use_local_llm: bool = True):
        """
        Initialize crypto signal generator with LLM Request Manager
        
        Args:
            api_key: Deprecated - API keys are now managed centrally
            config: Trading configuration object
            use_local_llm: Deprecated - provider selection is now automatic
        """
        if api_key:
            logger.warning("⚠️ api_key parameter is deprecated, keys are managed centrally")
        if not use_local_llm:
            logger.warning("⚠️ use_local_llm parameter is deprecated, provider selection is automatic")
        
        self.config = config
        
        # Initialize LLM Request Manager helper (HIGH priority for crypto trading signals)
        try:
            self.llm_helper = get_llm_helper("crypto_trading_signals", default_priority="HIGH")
            logger.success("🚀 Crypto Trading Signal Generator using LLM Request Manager")
        except Exception as e:
            logger.error(f"Failed to initialize LLM helper: {e}")
            raise
        
        # Crypto-specific parameters
        self.high_volatility_threshold = 5.0  # 5% daily volatility
        self.extreme_volatility_threshold = 10.0  # 10% daily volatility
    
    def generate_signal(
        self,
        symbol: str,
        technical_data: Dict,
        sentiment_data: Dict,
        social_data: Optional[Dict] = None,
        onchain_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        account_balance_usd: float = 1000.0,
        risk_tolerance: str = "MEDIUM",
        current_positions: Optional[List[str]] = None,
        time_horizon: str = "SCALP"
    ) -> Optional[CryptoTradingSignal]:
        """
        Generate AI trading signal for cryptocurrency
        
        Args:
            symbol: Crypto pair (e.g., 'BTC/USD')
            technical_data: Technical indicators adapted for crypto
            sentiment_data: News and market sentiment
            social_data: Twitter/Reddit/Discord sentiment
            onchain_data: Blockchain metrics (whale movements, exchange flows)
            market_data: Current price, volume, orderbook data
            account_balance_usd: Available capital
            risk_tolerance: LOW, MEDIUM, HIGH
            current_positions: List of currently held crypto assets
            time_horizon: SCALP, MOMENTUM, SWING, HOLD
            
        Returns:
            CryptoTradingSignal or None
        """
        try:
            # Parse symbol
            base_asset, quote_asset = self._parse_symbol(symbol)
            
            # Get Fear & Greed Index (crypto-specific sentiment indicator)
            fear_greed = self._get_fear_greed_index()
            
            # Determine market regime
            market_regime = self._determine_market_regime(technical_data, sentiment_data)
            
            # Build comprehensive analysis prompt
            prompt = self._build_crypto_analysis_prompt(
                symbol=symbol,
                base_asset=base_asset,
                technical_data=technical_data,
                sentiment_data=sentiment_data,
                social_data=social_data,
                onchain_data=onchain_data,
                market_data=market_data,
                fear_greed=fear_greed,
                market_regime=market_regime,
                account_balance=account_balance_usd,
                risk_tolerance=risk_tolerance,
                current_positions=current_positions or [],
                time_horizon=time_horizon
            )
            # Get AI analysis using LLM Request Manager
            # Use HIGH priority for crypto trading signals with symbol-based caching (2 min TTL)
            cache_key = f"crypto_signal_{symbol}_{int(time.time() // 120)}"  # Cache per 2-min window
            response = self.llm_helper.high_request(
                prompt,
                cache_key=cache_key,
                ttl=120,  # 2 minutes cache for crypto signals
                temperature=0.3  # Lower temperature for consistent signals
            )
            
            if response:
                # Parse AI response into crypto trading signal
                signal = self._parse_crypto_signal(
                    response=response,
                    symbol=symbol,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    market_data=market_data,
                    fear_greed=fear_greed,
                    market_regime=market_regime
                )
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating crypto signal for {symbol}: {e}")
            return None
    
    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parse crypto symbol into base and quote assets"""
        if '/' in symbol:
            parts = symbol.split('/')
            return parts[0], parts[1]
        else:
            # Assume USD if no separator
            return symbol, 'USD'
    
    def _get_fear_greed_index(self) -> Optional[int]:
        """
        Get the Crypto Fear & Greed Index
        
        Returns:
            Index value 0-100 (0=Extreme Fear, 100=Extreme Greed)
        """
        try:
            # Alternative Free Crypto Fear & Greed Index API
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    index_value = int(data['data'][0]['value'])
                    logger.info(f"Fear & Greed Index: {index_value}")
                    return index_value
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not fetch Fear & Greed Index: {e}")
            return None
    
    def _determine_market_regime(self, technical_data: Dict, sentiment_data: Dict) -> str:
        """
        Determine current market regime
        
        Returns:
            'BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE'
        """
        try:
            # Check price trend
            price_change = technical_data.get('price_change_24h', 0)
            volatility = technical_data.get('volatility_24h', 0)
            
            # Check moving averages
            price = technical_data.get('current_price', 0)
            ema_20 = technical_data.get('ema_20', 0)
            ema_50 = technical_data.get('ema_50', 0)
            
            # High volatility regime
            if volatility > self.extreme_volatility_threshold:
                return 'VOLATILE'
            
            # Bullish regime
            if price > ema_20 and ema_20 > ema_50 and price_change > 2:
                return 'BULL'
            
            # Bearish regime
            if price < ema_20 and ema_20 < ema_50 and price_change < -2:
                return 'BEAR'
            
            # Sideways/ranging regime
            return 'SIDEWAYS'
            
        except Exception as e:
            logger.warning(f"Error determining market regime: {e}")
            return 'SIDEWAYS'
    
    def _build_crypto_analysis_prompt(
        self,
        symbol: str,
        base_asset: str,
        technical_data: Dict,
        sentiment_data: Dict,
        social_data: Optional[Dict],
        onchain_data: Optional[Dict],
        market_data: Optional[Dict],
        fear_greed: Optional[int],
        market_regime: str,
        account_balance: float,
        risk_tolerance: str,
        current_positions: List[str],
        time_horizon: str
    ) -> str:
        """Build comprehensive AI analysis prompt for crypto trading"""
        
        prompt = f"""You are an expert cryptocurrency trading AI analyzing {symbol} ({base_asset}).

MARKET CONTEXT:
- Current Market Regime: {market_regime}
- Fear & Greed Index: {fear_greed if fear_greed else 'N/A'} (0=Extreme Fear, 100=Extreme Greed)
- Trading Timeframe: {time_horizon}
- Account Balance: ${account_balance:,.2f} USD
- Risk Tolerance: {risk_tolerance}
- Current Positions: {', '.join(current_positions) if current_positions else 'None'}

MARKET DATA:
"""
        
        if market_data:
            prompt += f"""- Current Price: ${market_data.get('last_price', 'N/A')}
- 24h Change: {market_data.get('change_pct_24h', 'N/A')}%
- 24h High: ${market_data.get('high_24h', 'N/A')}
- 24h Low: ${market_data.get('low_24h', 'N/A')}
- 24h Volume: {market_data.get('volume_24h', 'N/A')}
- Bid/Ask Spread: {market_data.get('spread_pct', 'N/A')}%
"""
        
        prompt += f"""
TECHNICAL ANALYSIS:
"""
        
        if technical_data:
            prompt += f"""- RSI(14): {technical_data.get('rsi', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}
- EMA(8): ${technical_data.get('ema_8', 'N/A')}
- EMA(20): ${technical_data.get('ema_20', 'N/A')}
- EMA(50): ${technical_data.get('ema_50', 'N/A')}
- Bollinger Bands: Upper=${technical_data.get('bb_upper', 'N/A')}, Lower=${technical_data.get('bb_lower', 'N/A')}
- Volume Ratio (24h avg): {technical_data.get('volume_ratio', 'N/A')}x
- ATR (Average True Range): {technical_data.get('atr', 'N/A')}
- Support Level: ${technical_data.get('support', 'N/A')}
- Resistance Level: ${technical_data.get('resistance', 'N/A')}
"""
        
        if social_data:
            prompt += f"""
SOCIAL SENTIMENT:
- Twitter/X Sentiment: {social_data.get('twitter_sentiment', 'N/A')} ({social_data.get('twitter_mentions', 0)} mentions)
- Reddit Sentiment: {social_data.get('reddit_sentiment', 'N/A')} ({social_data.get('reddit_posts', 0)} posts)
- Discord Activity: {social_data.get('discord_sentiment', 'N/A')}
- Social Volume Change: {social_data.get('social_volume_change', 'N/A')}%
"""
        
        if onchain_data:
            prompt += f"""
ON-CHAIN METRICS:
- Exchange Inflow (24h): {onchain_data.get('exchange_inflow', 'N/A')}
- Exchange Outflow (24h): {onchain_data.get('exchange_outflow', 'N/A')}
- Whale Transactions: {onchain_data.get('whale_transactions', 'N/A')}
- Active Addresses: {onchain_data.get('active_addresses', 'N/A')}
- Network Hash Rate Change: {onchain_data.get('hashrate_change', 'N/A')}%
"""
        
        prompt += f"""
NEWS & SENTIMENT:
- News Sentiment: {sentiment_data.get('overall_sentiment', 'N/A')}
- Major News: {sentiment_data.get('headline', 'No major news')}
- Catalyst Detected: {sentiment_data.get('catalyst_type', 'None')}

CRYPTO-SPECIFIC CONSIDERATIONS:
1. Market operates 24/7 - no trading hours restrictions
2. Higher volatility than traditional markets (adjust position sizing)
3. Social sentiment has significant impact on price
4. Weekend trading is normal (but often lower liquidity)
5. Instant settlement - no T+2 waiting period
6. Maker/taker fee structure (consider fee impact on small moves)
7. Flash crashes and sudden moves are more common
8. Liquidity can vary significantly by time of day

RISK MANAGEMENT FOR CRYPTO:
- Position Size: Max {self.config.MAX_POSITION_SIZE_PCT if self.config else 12}% of capital per trade
- Stop Loss: Wider stops needed due to volatility
- Take Profit: Higher targets reasonable (3-12% depending on timeframe)
- Volatility Adjustment: Scale position size inversely with volatility
- Fear & Greed: Extreme values (>75 or <25) signal potential reversals

Please analyze all available data and provide a trading recommendation in the following JSON format:

{{
    "signal": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "entry_price": price in USD,
    "target_price": price in USD,
    "stop_loss": price in USD,
    "position_size_pct": percentage of capital (0-{self.config.MAX_POSITION_SIZE_PCT if self.config else 12}),
    "risk_level": "LOW" or "MEDIUM" or "HIGH" or "EXTREME",
    "reasoning": "detailed explanation",
    "technical_score": 0-100,
    "sentiment_score": 0-100,
    "social_score": 0-100,
    "volatility_score": 0-100,
    "liquidity_score": 0-100,
    "key_factors": ["factor1", "factor2", "factor3"],
    "warnings": ["warning1", "warning2"] or []
}}

IMPORTANT: 
- For SCALP: tight stops (1-2%), quick targets (2-4%), hold 15-30 min
- For MOMENTUM: medium stops (3-5%), higher targets (5-10%), hold few hours
- For SWING: wider stops (5-8%), large targets (10-20%), hold 1-3 days
- Always consider the bid/ask spread in crypto markets
- Account for maker/taker fees (typically 0.16-0.26%)
- In high volatility (>8% daily), reduce position size by 30%
- In extreme fear (<25), look for buy opportunities (contrarian)
- In extreme greed (>75), be cautious of sell-off risks
"""
        
        return prompt
    
    # Removed _call_ai_model - now using LLM Request Manager
    
    def _parse_crypto_signal(
        self,
        response: str,
        symbol: str,
        base_asset: str,
        quote_asset: str,
        market_data: Optional[Dict],
        fear_greed: Optional[int],
        market_regime: str
    ) -> Optional[CryptoTradingSignal]:
        """
        Parse AI response into CryptoTradingSignal
        
        Args:
            response: AI response string
            symbol: Crypto pair
            base_asset: Base currency
            quote_asset: Quote currency
            market_data: Market data dict
            fear_greed: Fear & Greed Index
            market_regime: Current market regime
            
        Returns:
            CryptoTradingSignal or None
        """
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if not json_match:
                logger.error("No JSON found in AI response")
                return None
            
            data = json.loads(json_match.group())
            
            # Validate required fields
            required_fields = ['signal', 'confidence', 'entry_price', 'target_price', 
                             'stop_loss', 'position_size_pct', 'risk_level', 'reasoning']
            
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return None
            
            # Calculate position sizes
            position_size_pct = float(data['position_size_pct'])
            entry_price = float(data['entry_price'])
            
            # Adjust for volatility if config available
            if self.config and hasattr(self.config, 'USE_VOLATILITY_ADJUSTMENT'):
                if self.config.USE_VOLATILITY_ADJUSTMENT:
                    volatility = market_data.get('volatility_24h', 0) if market_data else 0
                    if volatility > self.high_volatility_threshold:
                        multiplier = self.config.HIGH_VOLATILITY_MULTIPLIER if hasattr(self.config, 'HIGH_VOLATILITY_MULTIPLIER') else 0.7
                        position_size_pct *= multiplier
                        logger.info(f"High volatility detected ({volatility}%), reducing position by {(1-multiplier)*100}%")
            
            # Calculate position sizes based on config
            account_balance = 1000.0  # Default, should be passed in
            if self.config and hasattr(self.config, 'TOTAL_CAPITAL'):
                account_balance = self.config.TOTAL_CAPITAL
            
            position_size_usd = account_balance * (position_size_pct / 100)
            position_size_crypto = position_size_usd / entry_price if entry_price > 0 else 0
            
            # Map time horizon from reasoning
            time_horizon = self._extract_time_horizon(data['reasoning'])
            
            # Create signal object
            signal = CryptoTradingSignal(
                symbol=symbol,
                base_asset=base_asset,
                quote_asset=quote_asset,
                signal=data['signal'].upper(),
                confidence=float(data['confidence']),
                entry_price=entry_price,
                target_price=float(data['target_price']),
                stop_loss=float(data['stop_loss']),
                position_size_usd=position_size_usd,
                position_size_crypto=position_size_crypto,
                reasoning=data['reasoning'],
                risk_level=data['risk_level'].upper(),
                time_horizon=time_horizon,
                technical_score=float(data.get('technical_score', 0)),
                sentiment_score=float(data.get('sentiment_score', 0)),
                social_score=float(data.get('social_score', 0)),
                volatility_score=float(data.get('volatility_score', 0)),
                liquidity_score=float(data.get('liquidity_score', 0)),
                onchain_score=float(data.get('onchain_score', 0)),
                market_regime=market_regime,
                fear_greed_index=fear_greed,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Generated {signal.signal} signal for {symbol} with {signal.confidence:.1f}% confidence")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error parsing crypto signal: {e}")
            return None
    
    def _extract_time_horizon(self, reasoning: str) -> str:
        """Extract time horizon from reasoning text"""
        reasoning_lower = reasoning.lower()
        
        if 'scalp' in reasoning_lower or 'quick' in reasoning_lower or 'minutes' in reasoning_lower:
            return 'SCALP'
        elif 'momentum' in reasoning_lower or 'hours' in reasoning_lower:
            return 'MOMENTUM'
        elif 'swing' in reasoning_lower or 'days' in reasoning_lower:
            return 'SWING'
        elif 'hold' in reasoning_lower or 'long' in reasoning_lower or 'weeks' in reasoning_lower:
            return 'HOLD'
        else:
            return 'SCALP'  # Default
    
    def validate_signal(self, signal: CryptoTradingSignal) -> Tuple[bool, List[str]]:
        """
        Validate a crypto trading signal before execution
        
        Args:
            signal: CryptoTradingSignal to validate
            
        Returns:
            (is_valid: bool, warnings: List[str])
        """
        warnings = []
        
        # Check confidence threshold
        min_confidence = 65 if self.config and hasattr(self.config, 'MIN_CONFIDENCE') else 65
        if signal.confidence < min_confidence:
            warnings.append(f"Confidence {signal.confidence}% below threshold {min_confidence}%")
        
        # Check risk/reward ratio
        if signal.risk_reward_ratio < 1.5:
            warnings.append(f"Poor risk/reward ratio: {signal.risk_reward_ratio:.2f} (should be >1.5)")
        
        # Check for extreme volatility
        if signal.volatility_score > 80:
            warnings.append("Extreme volatility detected - high risk trade")
        
        # Check Fear & Greed extremes
        if signal.fear_greed_index:
            if signal.fear_greed_index > 80 and signal.signal == 'BUY':
                warnings.append("Extreme Greed (>80) - buying at potential top")
            elif signal.fear_greed_index < 20 and signal.signal == 'SELL':
                warnings.append("Extreme Fear (<20) - selling at potential bottom")
        
        # Check liquidity
        if signal.liquidity_score < 40:
            warnings.append("Low liquidity - may face slippage on execution")
        
        # Check position size
        max_position_pct = 15.0
        if self.config and hasattr(self.config, 'MAX_POSITION_SIZE_PCT'):
            max_position_pct = self.config.MAX_POSITION_SIZE_PCT
        
        actual_pct = (signal.position_size_usd / (self.config.TOTAL_CAPITAL if self.config else 1000.0)) * 100
        if actual_pct > max_position_pct:
            warnings.append(f"Position size {actual_pct:.1f}% exceeds max {max_position_pct}%")
        
        is_valid = len(warnings) == 0 or (len(warnings) <= 2 and signal.confidence > 75)
        
        return is_valid, warnings

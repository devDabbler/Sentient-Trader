"""
Crypto Whale Wallet Tracker
Tracks large wallet movements to identify potential market shifts and big moves.

Features:
- Multi-chain support (Ethereum, BSC, Solana, Polygon)
- Wash trading detection (fake/manipulated signals)
- Exchange flow monitoring (inflow/outflow)
- Whale transaction alerts
- Hybrid validation with other signals

Integration with:
- Crypto scanners for opportunity detection
- Sentiment analysis for confirmation
- Technical indicators for validation
"""

import os
import httpx
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()


@dataclass
class WhaleTransaction:
    """Represents a whale wallet transaction"""
    symbol: str  # e.g., 'BTC', 'ETH', 'SOL'
    chain: str  # 'ethereum', 'bsc', 'solana', 'polygon'
    from_address: str
    to_address: str
    amount: float  # Token amount
    amount_usd: float  # USD value
    transaction_type: str  # 'DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'SWAP'
    exchange: Optional[str] = None  # Exchange name if known
    timestamp: str = ""
    transaction_hash: str = ""
    is_exchange: bool = False  # True if to/from known exchange
    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW (for manipulation detection)
    risk_flags: List[str] = None  # Flags indicating potential manipulation


@dataclass
class WhaleAlert:
    """Alert for significant whale activity"""
    symbol: str
    alert_type: str  # 'LARGE_DEPOSIT', 'LARGE_WITHDRAWAL', 'WHALE_MOVEMENT', 'EXCHANGE_FLOW'
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    amount_usd: float
    direction: str  # 'INFLOW', 'OUTFLOW', 'NEUTRAL'
    description: str
    timestamp: str
    confidence: float  # 0-1, higher = less likely to be manipulation
    supporting_signals: Dict = None  # Volume, sentiment, technical indicators


class CryptoWhaleTracker:
    """
    Tracks whale wallet movements across multiple chains.
    Detects manipulation patterns and validates with hybrid analysis.
    """
    
    # Known exchange addresses (partial list - would be expanded)
    EXCHANGE_ADDRESSES = {
        'ethereum': {
            'Binance': ['0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be'],
            'Coinbase': ['0x71660c4005ba85c37ccec55d0c4493e66fe775d3'],
            'Kraken': ['0xe853c56864a2ebe4576a807d26fdc4a0ada51919'],
            'Bitfinex': ['0x1151314c646ce4e0efd76d1af4760ae66a9fe30f'],
            'Gemini': ['0x07ee55aa48bb72dcc6e9d78256648910de513eca'],
        },
        'bsc': {
            'Binance': ['0x8894e0a0c962cb723c1976a4421c95949be2d4e3'],
        },
        'solana': {
            'Binance': ['CZ8k8FjL4wQX9YJLjHGxX4h8CzXqP2V5XJ'],
            'Coinbase': [''],
        }
    }
    
    # Minimum thresholds for whale transactions (USD)
    WHALE_THRESHOLDS = {
        'BTC': 1_000_000,  # $1M+
        'ETH': 500_000,    # $500K+
        'SOL': 100_000,    # $100K+
        'BNB': 100_000,
        'MATIC': 50_000,
        'DEFAULT': 100_000  # Default for other tokens
    }
    
    def __init__(self):
        """Initialize whale tracker"""
        # Etherscan and BSCScan have merged - use same API key for both
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
        self.bscscan_api_key = self.etherscan_api_key  # Use same key (merged services)
        self.solscan_api_key = os.getenv('SOLSCAN_API_KEY')
        
        # Rate limiting
        self.last_etherscan_call = 0
        self.last_bscscan_call = 0
        self.last_solscan_call = 0
        self.rate_limit_delay = 0.2  # 200ms between calls
        
        # Cache for tracking patterns
        self.recent_transactions = defaultdict(list)  # Track recent txs for pattern detection
        self.suspicious_patterns = defaultdict(int)  # Count suspicious patterns
        
        logger.info("🐋 Crypto Whale Tracker initialized")
        logger.info(f"   • Ethereum & BSC: {'Enabled' if self.etherscan_api_key else 'Disabled (no API key)'} (Etherscan API)")
        logger.info(f"   • Solana: {'Enabled' if self.solscan_api_key else 'Disabled (no API key)'} (Solscan API)")
    
    async def get_whale_transactions(
        self,
        symbol: str,
        chain: str = 'ethereum',
        hours: int = 24,
        min_amount_usd: Optional[float] = None
    ) -> List[WhaleTransaction]:
        """
        Get whale transactions for a specific token
        
        Args:
            symbol: Token symbol (e.g., 'ETH', 'BTC', 'SOL')
            chain: Blockchain ('ethereum', 'bsc', 'solana', 'polygon')
            hours: Lookback period in hours
            min_amount_usd: Minimum transaction size in USD
            
        Returns:
            List of WhaleTransaction objects
        """
        if min_amount_usd is None:
            min_amount_usd = self.WHALE_THRESHOLDS.get(symbol.upper(), self.WHALE_THRESHOLDS['DEFAULT'])
        
        try:
            if chain == 'ethereum':
                return await self._get_ethereum_whales(symbol, hours, min_amount_usd)
            elif chain == 'bsc':
                return await self._get_bsc_whales(symbol, hours, min_amount_usd)
            elif chain == 'solana':
                return await self._get_solana_whales(symbol, hours, min_amount_usd)
            else:
                logger.warning(f"Unsupported chain: {chain}")
                return []
        except Exception as e:
            logger.error(f"Error fetching whale transactions for {symbol} on {chain}: {e}")
            return []
    
    async def _get_ethereum_whales(
        self,
        symbol: str,
        hours: int,
        min_amount_usd: float
    ) -> List[WhaleTransaction]:
        """Get Ethereum whale transactions using Etherscan API"""
        if not self.etherscan_api_key:
            logger.warning("Etherscan API key not found")
            return []
        
        # Map common symbols to contract addresses
        # This would need to be expanded with actual token contracts
        token_contracts = {
            'ETH': None,  # Native token
            'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
            'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
            'WBTC': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
        }
        
        contract_address = token_contracts.get(symbol.upper())
        
        try:
            # Rate limiting
            await self._rate_limit('etherscan')
            
            # For now, return mock data structure
            # In production, this would call Etherscan API
            # Example: https://api.etherscan.io/api?module=account&action=tokentx&contractaddress={contract}&startblock=0&endblock=99999999&sort=desc&apikey={api_key}
            
            logger.info(f"🐋 Fetching Ethereum whale transactions for {symbol} (last {hours}h)")
            # TODO: Implement actual Etherscan API calls
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Ethereum whales: {e}")
            return []
    
    async def _get_bsc_whales(
        self,
        symbol: str,
        hours: int,
        min_amount_usd: float
    ) -> List[WhaleTransaction]:
        """Get BSC whale transactions using Etherscan API (merged service)"""
        if not self.bscscan_api_key:
            logger.warning("Etherscan API key not found (required for BSC)")
            return []
        
        try:
            await self._rate_limit('bscscan')
            logger.info(f"🐋 Fetching BSC whale transactions for {symbol} (last {hours}h) via Etherscan API")
            # TODO: Implement actual BscScan API calls (uses same Etherscan API key)
            # Note: BSC uses different base URL but same API key
            return []
        except Exception as e:
            logger.error(f"Error fetching BSC whales: {e}")
            return []
    
    async def _get_solana_whales(
        self,
        symbol: str,
        hours: int,
        min_amount_usd: float
    ) -> List[WhaleTransaction]:
        """Get Solana whale transactions using Solscan API"""
        if not self.solscan_api_key:
            logger.warning("Solscan API key not found")
            return []
        
        try:
            await self._rate_limit('solscan')
            logger.info(f"🐋 Fetching Solana whale transactions for {symbol} (last {hours}h)")
            # TODO: Implement actual Solscan API calls
            return []
        except Exception as e:
            logger.error(f"Error fetching Solana whales: {e}")
            return []
    
    async def _rate_limit(self, service: str):
        """Rate limit API calls"""
        current_time = time.time()
        
        if service == 'etherscan':
            elapsed = current_time - self.last_etherscan_call
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            self.last_etherscan_call = time.time()
        elif service == 'bscscan':
            elapsed = current_time - self.last_bscscan_call
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            self.last_bscscan_call = time.time()
        elif service == 'solscan':
            elapsed = current_time - self.last_solscan_call
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            self.last_solscan_call = time.time()
    
    def detect_manipulation(
        self,
        transactions: List[WhaleTransaction],
        symbol: str
    ) -> Tuple[List[WhaleTransaction], List[str]]:
        """
        Detect potential manipulation/fake signals
        
        Patterns to detect:
        1. Circular transfers (same tokens moving back and forth)
        2. Wash trading (rapid buy/sell cycles)
        3. Coordinated movements (multiple addresses moving simultaneously)
        4. Exchange manipulation (fake volume patterns)
        
        Returns:
            Tuple of (validated_transactions, risk_flags)
        """
        validated = []
        risk_flags = []
        
        if not transactions:
            return validated, risk_flags
        
        # Track recent transactions for pattern detection
        key = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
        self.recent_transactions[key].extend(transactions)
        
        # Keep only last 24h of transactions
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.recent_transactions[key] = [
            tx for tx in self.recent_transactions[key]
            if datetime.fromisoformat(tx.timestamp.replace('Z', '+00:00')) > cutoff_time
        ]
        
        recent_txs = self.recent_transactions[key]
        
        for tx in transactions:
            tx_risk_flags = []
            
            # Check for circular transfers
            if self._is_circular_transfer(tx, recent_txs):
                tx_risk_flags.append("CIRCULAR_TRANSFER")
                tx.confidence = "LOW"
            
            # Check for wash trading patterns
            if self._is_wash_trading(tx, recent_txs):
                tx_risk_flags.append("WASH_TRADING")
                tx.confidence = "LOW"
            
            # Check for coordinated movements
            if self._is_coordinated_movement(tx, recent_txs):
                tx_risk_flags.append("COORDINATED_MOVEMENT")
                tx.confidence = "MEDIUM"
            
            # Check if transaction is to/from known exchange (more reliable)
            if tx.is_exchange:
                tx.confidence = "HIGH"
            
            # If no risk flags, transaction is validated
            if not tx_risk_flags:
                validated.append(tx)
            else:
                risk_flags.extend(tx_risk_flags)
                tx.risk_flags = tx_risk_flags
                # Still add to validated but with lower confidence
                validated.append(tx)
        
        return validated, list(set(risk_flags))
    
    def _is_circular_transfer(self, tx: WhaleTransaction, recent_txs: List[WhaleTransaction]) -> bool:
        """Check if transaction is part of a circular transfer pattern"""
        # Look for transactions where tokens move from A->B then B->A
        for recent_tx in recent_txs[-10:]:  # Check last 10 transactions
            if (recent_tx.to_address.lower() == tx.from_address.lower() and
                recent_tx.from_address.lower() == tx.to_address.lower() and
                abs(recent_tx.amount - tx.amount) < (tx.amount * 0.1)):  # Within 10% of same amount
                return True
        return False
    
    def _is_wash_trading(self, tx: WhaleTransaction, recent_txs: List[WhaleTransaction]) -> bool:
        """Check for wash trading patterns (rapid buy/sell cycles)"""
        # Look for rapid transactions between same addresses
        same_pair_txs = [
            rt for rt in recent_txs[-20:]
            if ((rt.from_address.lower() == tx.from_address.lower() and
                 rt.to_address.lower() == tx.to_address.lower()) or
                (rt.from_address.lower() == tx.to_address.lower() and
                 rt.to_address.lower() == tx.from_address.lower()))
        ]
        
        if len(same_pair_txs) >= 3:  # 3+ rapid transactions suggest wash trading
            return True
        return False
    
    def _is_coordinated_movement(self, tx: WhaleTransaction, recent_txs: List[WhaleTransaction]) -> bool:
        """Check for coordinated movements (multiple addresses moving simultaneously)"""
        # Look for multiple transactions with similar amounts at similar times
        time_window = timedelta(minutes=5)
        tx_time = datetime.fromisoformat(tx.timestamp.replace('Z', '+00:00'))
        
        similar_txs = [
            rt for rt in recent_txs
            if (abs((datetime.fromisoformat(rt.timestamp.replace('Z', '+00:00')) - tx_time).total_seconds()) < time_window.total_seconds() and
                abs(rt.amount_usd - tx.amount_usd) < (tx.amount_usd * 0.2))  # Within 20% of same amount
        ]
        
        if len(similar_txs) >= 3:  # 3+ similar transactions suggest coordination
            return True
        return False
    
    def calculate_exchange_flow(
        self,
        transactions: List[WhaleTransaction],
        timeframe_hours: int = 24
    ) -> Dict[str, float]:
        """
        Calculate exchange inflow/outflow
        
        Returns:
            Dict with 'inflow', 'outflow', 'net_flow', 'flow_ratio'
        """
        inflow = 0.0
        outflow = 0.0
        
        for tx in transactions:
            if not tx.is_exchange:
                continue
            
            if tx.transaction_type == 'DEPOSIT':
                inflow += tx.amount_usd
            elif tx.transaction_type == 'WITHDRAWAL':
                outflow += tx.amount_usd
        
        net_flow = inflow - outflow
        flow_ratio = inflow / outflow if outflow > 0 else (inflow / 1.0 if inflow > 0 else 0.0)
        
        return {
            'inflow': inflow,
            'outflow': outflow,
            'net_flow': net_flow,
            'flow_ratio': flow_ratio,
            'timeframe_hours': timeframe_hours
        }
    
    async def generate_whale_alert(
        self,
        symbol: str,
        transactions: List[WhaleTransaction],
        volume_data: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        technical_data: Optional[Dict] = None,
        news_data: Optional[Dict] = None
    ) -> Optional[WhaleAlert]:
        """
        Generate whale alert with hybrid validation
        
        Combines whale data with:
        - Volume data (confirm volume spike)
        - Sentiment data (confirm social interest)
        - Technical data (confirm price action)
        
        Returns None if signals don't align (potential false positive)
        """
        if not transactions:
            return None
        
        # Calculate exchange flow
        exchange_flow = self.calculate_exchange_flow(transactions)
        
        # Filter out low-confidence transactions
        high_confidence_txs = [tx for tx in transactions if tx.confidence in ['HIGH', 'MEDIUM']]
        
        if not high_confidence_txs:
            logger.warning(f"No high-confidence whale transactions for {symbol}")
            return None
        
        # Calculate total whale activity
        total_amount = sum(tx.amount_usd for tx in high_confidence_txs)
        
        # Determine alert type and severity
        alert_type = 'WHALE_MOVEMENT'
        severity = 'MEDIUM'
        
        if exchange_flow['inflow'] > exchange_flow['outflow'] * 2:
            alert_type = 'LARGE_DEPOSIT'
            severity = 'HIGH'
        elif exchange_flow['outflow'] > exchange_flow['inflow'] * 2:
            alert_type = 'LARGE_WITHDRAWAL'
            severity = 'HIGH'
        
        # Hybrid validation - check if other signals support the whale movement
        supporting_signals = {}
        confidence_score = 0.5  # Base confidence
        
        # Volume validation (25% weight)
        if volume_data:
            volume_spike = volume_data.get('volume_ratio', 1.0)
            if volume_spike > 2.0:  # 2x+ volume spike
                supporting_signals['volume_spike'] = volume_spike
                confidence_score += 0.25
            elif volume_spike > 1.5:
                supporting_signals['volume_spike'] = volume_spike
                confidence_score += 0.15
        
        # News/sentiment validation (30% weight) - highest priority for validation
        if news_data:
            news_sentiment = news_data.get('overall_sentiment_score', 0.0)
            news_count = news_data.get('news_count', 0)
            if news_sentiment > 0.3:  # Bullish news/sentiment
                supporting_signals['news_sentiment'] = 'BULLISH'
                confidence_score += 0.3
            elif news_sentiment < -0.3:  # Bearish news/sentiment
                supporting_signals['news_sentiment'] = 'BEARISH'
                confidence_score += 0.15
            
            if news_count >= 5:  # Significant news volume
                supporting_signals['news_volume'] = news_count
                confidence_score += 0.1
        
        # Social sentiment validation (20% weight)
        if sentiment_data:
            sentiment_score = sentiment_data.get('overall_sentiment_score', 0.0)
            if sentiment_score > 0.3:  # Bullish sentiment
                supporting_signals['sentiment'] = 'BULLISH'
                confidence_score += 0.2
            elif sentiment_score < -0.3:  # Bearish sentiment
                supporting_signals['sentiment'] = 'BEARISH'
                confidence_score += 0.1
        
        # Technical validation (15% weight)
        if technical_data:
            price_change = technical_data.get('change_pct_24h', 0.0)
            if abs(price_change) > 5.0:  # Significant price movement
                supporting_signals['price_action'] = price_change
                confidence_score += 0.15
            elif abs(price_change) > 2.0:
                supporting_signals['price_action'] = price_change
                confidence_score += 0.1
        
        # RSI validation (10% weight)
        if technical_data and 'rsi' in technical_data:
            rsi = technical_data['rsi']
            if rsi < 30 or rsi > 70:  # Oversold/overbought
                supporting_signals['rsi_extreme'] = rsi
                confidence_score += 0.1
        
        # Normalize confidence to 0-1
        confidence_score = min(1.0, confidence_score)
        
        # Only generate alert if confidence is above threshold
        if confidence_score < 0.4:  # Low confidence = potential false positive
            logger.info(f"Low confidence whale signal for {symbol} (score: {confidence_score:.2f}) - skipping alert")
            return None
        
        # Create alert
        direction = 'INFLOW' if exchange_flow['net_flow'] > 0 else 'OUTFLOW' if exchange_flow['net_flow'] < 0 else 'NEUTRAL'
        
        description = f"{alert_type} detected: ${total_amount:,.0f} in whale activity"
        if exchange_flow['net_flow'] != 0:
            description += f" (Net {direction}: ${abs(exchange_flow['net_flow']):,.0f})"
        
        if supporting_signals:
            description += f" | Supported by: {', '.join(supporting_signals.keys())}"
        
        alert = WhaleAlert(
            symbol=symbol,
            alert_type=alert_type,
            severity=severity,
            amount_usd=total_amount,
            direction=direction,
            description=description,
            timestamp=datetime.now().isoformat(),
            confidence=confidence_score,
            supporting_signals=supporting_signals
        )
        
        logger.info(f"🐋 Whale Alert: {symbol} - {alert_type} (confidence: {confidence_score:.2f})")
        
        return alert
    
    async def get_whale_insights(
        self,
        symbol: str,
        chain: str = 'ethereum',
        hours: int = 24
    ) -> Dict:
        """
        Get comprehensive whale insights for a token
        
        Returns:
            Dict with whale activity summary, alerts, and validation
        """
        try:
            # Get whale transactions
            transactions = await self.get_whale_transactions(symbol, chain, hours)
            
            # Detect manipulation
            validated_txs, risk_flags = self.detect_manipulation(transactions, symbol)
            
            # Calculate exchange flow
            exchange_flow = self.calculate_exchange_flow(validated_txs, hours)
            
            # Summary
            total_whale_volume = sum(tx.amount_usd for tx in validated_txs)
            high_confidence_count = sum(1 for tx in validated_txs if tx.confidence == 'HIGH')
            medium_confidence_count = sum(1 for tx in validated_txs if tx.confidence == 'MEDIUM')
            low_confidence_count = sum(1 for tx in validated_txs if tx.confidence == 'LOW')
            
            return {
                'symbol': symbol,
                'chain': chain,
                'timeframe_hours': hours,
                'total_transactions': len(validated_txs),
                'high_confidence_txs': high_confidence_count,
                'medium_confidence_txs': medium_confidence_count,
                'low_confidence_txs': low_confidence_count,
                'total_whale_volume_usd': total_whale_volume,
                'exchange_flow': exchange_flow,
                'risk_flags': risk_flags,
                'whale_activity_score': self._calculate_whale_activity_score(validated_txs, exchange_flow),
                'transactions': [
                    {
                        'amount_usd': tx.amount_usd,
                        'type': tx.transaction_type,
                        'confidence': tx.confidence,
                        'timestamp': tx.timestamp
                    }
                    for tx in validated_txs[:10]  # Top 10
                ]
            }
        except Exception as e:
            logger.error(f"Error getting whale insights for {symbol}: {e}")
            return {}
    
    def _calculate_whale_activity_score(
        self,
        transactions: List[WhaleTransaction],
        exchange_flow: Dict
    ) -> float:
        """
        Calculate whale activity score (0-100)
        
        Higher score = more significant whale activity
        """
        if not transactions:
            return 0.0
        
        score = 0.0
        
        # Volume component (40%)
        total_volume = sum(tx.amount_usd for tx in transactions)
        if total_volume > 10_000_000:  # $10M+
            score += 40
        elif total_volume > 5_000_000:  # $5M+
            score += 30
        elif total_volume > 1_000_000:  # $1M+
            score += 20
        elif total_volume > 500_000:  # $500K+
            score += 10
        
        # Exchange flow component (30%)
        net_flow_abs = abs(exchange_flow['net_flow'])
        if net_flow_abs > 5_000_000:  # $5M+
            score += 30
        elif net_flow_abs > 2_000_000:  # $2M+
            score += 20
        elif net_flow_abs > 1_000_000:  # $1M+
            score += 10
        
        # Confidence component (30%)
        high_confidence = sum(1 for tx in transactions if tx.confidence == 'HIGH')
        total_count = len(transactions)
        if total_count > 0:
            confidence_ratio = high_confidence / total_count
            score += confidence_ratio * 30
        
        return min(100.0, score)


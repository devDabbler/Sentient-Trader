"""
FreqAI Signal Bridge
Bridges FreqAI predictions from Freqtrade to Sentient Trader's signal pipeline
Receives webhook notifications and forwards to Discord/trading systems
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from loguru import logger
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Sentient Trader imports
try:
    from services.crypto_strategies import TradingSignal
    from services.discord_service import send_discord_alert
except ImportError:
    # Fallback for standalone testing
    TradingSignal = None
    send_discord_alert = None


class FreqtradeWebhookType(str, Enum):
    """Types of Freqtrade webhook events"""
    ENTRY = "entry"
    ENTRY_FILL = "entry_fill"
    ENTRY_CANCEL = "entry_cancel"
    EXIT = "exit"
    EXIT_FILL = "exit_fill"
    EXIT_CANCEL = "exit_cancel"
    STATUS = "status"


@dataclass
class FreqAISignal:
    """FreqAI prediction signal"""
    pair: str
    prediction: float
    confidence: float
    timestamp: datetime
    model_id: str
    features: Dict[str, float]
    do_predict: int  # 1 = confident, 0 = uncertain
    
    def to_trading_signal(self, current_price: float) -> Optional['TradingSignal']:
        """Convert FreqAI signal to TradingSignal format"""
        if TradingSignal is None:
            logger.warning("TradingSignal not available, cannot convert")
            return None
        
        # Determine signal type based on prediction
        if self.prediction > 0.02:  # >2% predicted gain
            signal_type = "BUY"
        elif self.prediction < -0.02:  # >2% predicted loss
            signal_type = "SELL"
        else:
            return None  # No clear signal
        
        # Calculate confidence from prediction strength and do_predict
        base_confidence = min(abs(self.prediction) * 1000, 100)  # Scale to 0-100
        if self.do_predict != 1:
            base_confidence *= 0.5  # Reduce confidence if model is uncertain
        
        # Calculate entry, stop, and target
        if signal_type == "BUY":
            entry_price = current_price
            stop_loss = current_price * 0.95  # 5% stop
            take_profit = current_price * (1 + abs(self.prediction))
        else:
            entry_price = current_price
            stop_loss = current_price * 1.05  # 5% stop
            take_profit = current_price * (1 - abs(self.prediction))
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        return TradingSignal(
            symbol=self.pair.replace("/", ""),
            strategy=f"FreqAI ({self.model_id})",
            signal_type=signal_type,
            confidence=base_confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=reward / risk if risk > 0 else 1.0,
            reasoning=f"FreqAI prediction: {self.prediction:.2%} | Confidence: {self.do_predict}",
            indicators=self.features,
            timestamp=self.timestamp,
            risk_level="MEDIUM" if abs(self.prediction) < 0.05 else "HIGH"
        )


class FreqtradeWebhookPayload(BaseModel):
    """Incoming webhook payload from Freqtrade"""
    value: str
    pair: Optional[str] = None
    trade_id: Optional[int] = None
    open_rate: Optional[float] = None
    current_rate: Optional[float] = None
    profit_ratio: Optional[float] = None
    profit_amount: Optional[float] = None
    stake_amount: Optional[float] = None
    direction: Optional[str] = None
    enter_tag: Optional[str] = None
    exit_reason: Optional[str] = None
    order_type: Optional[str] = None
    amount: Optional[float] = None
    
    # FreqAI specific fields
    freqai_prediction: Optional[float] = None
    freqai_do_predict: Optional[int] = None


class FreqAISignalBridge:
    """
    Bridge between Freqtrade/FreqAI and Sentient Trader
    
    Handles:
    - Receiving webhook notifications from Freqtrade
    - Converting FreqAI predictions to TradingSignals
    - Forwarding alerts to Discord
    - Storing signals for the UI
    """
    
    def __init__(self, discord_webhook_url: Optional[str] = None):
        self.discord_webhook_url = discord_webhook_url or os.getenv("DISCORD_WEBHOOK_CRYPTO_ALERTS")
        self.recent_signals: Dict[str, FreqAISignal] = {}
        self.trade_history: list = []
        self._signal_callbacks: list = []
    
    def register_callback(self, callback):
        """Register a callback for new signals"""
        self._signal_callbacks.append(callback)
    
    async def process_webhook(self, payload: FreqtradeWebhookPayload) -> Dict[str, Any]:
        """Process incoming Freqtrade webhook"""
        webhook_type = FreqtradeWebhookType(payload.value)
        
        logger.info(f"Received Freqtrade webhook: {webhook_type.value} for {payload.pair}")
        
        result = {
            "status": "processed",
            "type": webhook_type.value,
            "pair": payload.pair,
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle different webhook types
        if webhook_type == FreqtradeWebhookType.ENTRY:
            await self._handle_entry(payload)
        elif webhook_type == FreqtradeWebhookType.ENTRY_FILL:
            await self._handle_entry_fill(payload)
        elif webhook_type == FreqtradeWebhookType.EXIT:
            await self._handle_exit(payload)
        elif webhook_type == FreqtradeWebhookType.EXIT_FILL:
            await self._handle_exit_fill(payload)
        elif webhook_type == FreqtradeWebhookType.STATUS:
            await self._handle_status(payload)
        
        return result
    
    async def _handle_entry(self, payload: FreqtradeWebhookPayload):
        """Handle entry signal"""
        message = (
            f"🎯 **Freqtrade Entry Signal**\n"
            f"**Pair:** {payload.pair}\n"
            f"**Direction:** {payload.direction or 'LONG'}\n"
            f"**Rate:** ${payload.open_rate:.4f}\n"
            f"**Stake:** ${payload.stake_amount:.2f}\n"
        )
        
        if payload.freqai_prediction is not None:
            message += f"**FreqAI Prediction:** {payload.freqai_prediction:.2%}\n"
        
        if payload.enter_tag:
            message += f"**Entry Tag:** {payload.enter_tag}\n"
        
        await self._send_discord(message)
    
    async def _handle_entry_fill(self, payload: FreqtradeWebhookPayload):
        """Handle entry fill confirmation"""
        message = (
            f"✅ **Trade Opened**\n"
            f"**Pair:** {payload.pair}\n"
            f"**Trade ID:** {payload.trade_id}\n"
            f"**Entry Rate:** ${payload.open_rate:.4f}\n"
            f"**Amount:** {payload.amount:.6f}\n"
            f"**Stake:** ${payload.stake_amount:.2f}\n"
        )
        
        self.trade_history.append({
            "type": "entry",
            "pair": payload.pair,
            "trade_id": payload.trade_id,
            "rate": payload.open_rate,
            "amount": payload.amount,
            "timestamp": datetime.now().isoformat()
        })
        
        await self._send_discord(message)
    
    async def _handle_exit(self, payload: FreqtradeWebhookPayload):
        """Handle exit signal"""
        profit_emoji = "📈" if (payload.profit_ratio or 0) > 0 else "📉"
        
        message = (
            f"{profit_emoji} **Freqtrade Exit Signal**\n"
            f"**Pair:** {payload.pair}\n"
            f"**Current Rate:** ${payload.current_rate:.4f}\n"
            f"**Exit Reason:** {payload.exit_reason or 'Manual'}\n"
        )
        
        await self._send_discord(message)
    
    async def _handle_exit_fill(self, payload: FreqtradeWebhookPayload):
        """Handle exit fill confirmation"""
        profit_emoji = "🎉" if (payload.profit_ratio or 0) > 0 else "😔"
        profit_pct = (payload.profit_ratio or 0) * 100
        
        message = (
            f"{profit_emoji} **Trade Closed**\n"
            f"**Pair:** {payload.pair}\n"
            f"**Trade ID:** {payload.trade_id}\n"
            f"**Exit Rate:** ${payload.current_rate:.4f}\n"
            f"**Profit:** {profit_pct:.2f}% (${payload.profit_amount:.2f})\n"
            f"**Reason:** {payload.exit_reason or 'Unknown'}\n"
        )
        
        self.trade_history.append({
            "type": "exit",
            "pair": payload.pair,
            "trade_id": payload.trade_id,
            "rate": payload.current_rate,
            "profit_ratio": payload.profit_ratio,
            "profit_amount": payload.profit_amount,
            "exit_reason": payload.exit_reason,
            "timestamp": datetime.now().isoformat()
        })
        
        await self._send_discord(message)
    
    async def _handle_status(self, payload: FreqtradeWebhookPayload):
        """Handle status update"""
        logger.info(f"Freqtrade status update: {payload}")
    
    async def _send_discord(self, message: str):
        """Send message to Discord"""
        if not self.discord_webhook_url:
            logger.warning("No Discord webhook configured for FreqAI signals")
            return
        
        try:
            if send_discord_alert:
                await send_discord_alert(self.discord_webhook_url, message)
            else:
                # Fallback using aiohttp directly
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.discord_webhook_url,
                        json={"content": message}
                    )
            logger.info("Sent Discord notification for Freqtrade event")
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
    
    def get_recent_signals(self) -> Dict[str, FreqAISignal]:
        """Get recent FreqAI signals"""
        return self.recent_signals.copy()
    
    def get_trade_history(self) -> list:
        """Get trade history"""
        return self.trade_history.copy()


# FastAPI app for webhook endpoint
app = FastAPI(title="FreqAI Signal Bridge")
bridge = FreqAISignalBridge()


@app.post("/freqtrade/webhook")
async def freqtrade_webhook(request: Request):
    """Receive webhook from Freqtrade"""
    try:
        body = await request.json()
        payload = FreqtradeWebhookPayload(**body)
        result = await bridge.process_webhook(payload)
        return result
    except Exception as e:
        logger.error(f"Error processing Freqtrade webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/freqtrade/signals")
async def get_signals():
    """Get recent FreqAI signals"""
    return {
        "signals": {k: asdict(v) for k, v in bridge.get_recent_signals().items()},
        "trade_history": bridge.get_trade_history()
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "freqai-signal-bridge"}


def run_bridge_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the webhook server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_bridge_server()

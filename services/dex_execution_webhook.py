"""
DEX Hunter Execution Webhook Service
Handles token execution requests via webhooks (for future bundler integration)

This service provides high-level webhook endpoints for executing DEX Hunter discoveries:
- Snipe execution
- Arbitrage execution
- Safe-swap execution

Currently uses placeholders - ready for future bundler service integration
"""

import os
import asyncio
import json
from typing import Dict, Optional, Tuple
from datetime import datetime
from dataclasses import asdict, dataclass
from loguru import logger
from enum import Enum
import httpx


class ExecutionStatus(Enum):
    """Execution status codes"""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    EXECUTING = "EXECUTING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExecutionStrategy(Enum):
    """Execution strategies (for future implementation)"""
    SNIPE = "SNIPE"  # Buy token immediately
    ARBITRAGE = "ARBITRAGE"  # Execute arbitrage cycle
    SAFE_SWAP = "SAFE_SWAP"  # Validate + swap with protections


@dataclass
class ExecutionRequest:
    """Represents a DEX execution request"""
    request_id: str
    token_mint: str  # Solana token mint address
    strategy: ExecutionStrategy
    amount_usd: float  # Amount to invest/trade
    slippage_bps: int = 50  # 0.5% default
    source: str = "DEX_HUNTER"  # Where request originated
    timestamp: str = None
    metadata: Dict = None  # Additional context
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'request_id': self.request_id,
            'token_mint': self.token_mint,
            'strategy': self.strategy.value,
            'amount_usd': self.amount_usd,
            'slippage_bps': self.slippage_bps,
            'source': self.source,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class ExecutionResult:
    """Represents result of an execution attempt"""
    request_id: str
    status: ExecutionStatus
    success: bool
    message: str
    transaction_hash: Optional[str] = None
    tokens_received: Optional[float] = None
    cost_usd: Optional[float] = None
    roi_pct: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'request_id': self.request_id,
            'status': self.status.value,
            'success': self.success,
            'message': self.message,
            'transaction_hash': self.transaction_hash,
            'tokens_received': self.tokens_received,
            'cost_usd': self.cost_usd,
            'roi_pct': self.roi_pct,
            'timestamp': self.timestamp
        }


class DexExecutionWebhook:
    """
    Webhook handler for DEX Hunter execution requests
    
    HIGH-LEVEL ARCHITECTURE:
    1. Receives execution request from DEX Hunter
    2. Validates token safety
    3. Routes to appropriate execution service via webhook
    4. Returns status to caller
    
    INTEGRATION POINTS (Placeholder Ready):
    - /execute/snipe → Route to Jito bundler webhook
    - /execute/arbitrage → Route to arbitrage bot webhook
    - /execute/status → Check execution status
    """
    
    def __init__(
        self,
        snipe_webhook_url: Optional[str] = None,
        arbitrage_webhook_url: Optional[str] = None,
        timeout_seconds: int = 30
    ):
        """
        Initialize webhook handler
        
        Args:
            snipe_webhook_url: URL to external snipe executor (e.g., Jito bundler service)
            arbitrage_webhook_url: URL to external arbitrage executor
            timeout_seconds: Request timeout in seconds
        """
        # Load from env if not provided
        self.snipe_webhook_url = snipe_webhook_url or os.getenv('DEX_EXECUTION_SNIPE_WEBHOOK')
        self.arbitrage_webhook_url = arbitrage_webhook_url or os.getenv('DEX_EXECUTION_ARBITRAGE_WEBHOOK')
        self.timeout = timeout_seconds
        
        # Execution queue (in-memory for now, could migrate to database)
        self.execution_queue: Dict[str, ExecutionRequest] = {}
        self.execution_results: Dict[str, ExecutionResult] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'pending': 0
        }
        
        self._log_config()
    
    def _log_config(self):
        """Log configuration status"""
        snipe_status = "✅ Configured" if self.snipe_webhook_url else "⚠️ Not configured"
        arb_status = "✅ Configured" if self.arbitrage_webhook_url else "⚠️ Not configured"
        
        logger.info(f"[EXECUTION] DEX Execution Webhook initialized")
        logger.info(f"[EXECUTION]   Snipe Service: {snipe_status}")
        logger.info(f"[EXECUTION]   Arbitrage Service: {arb_status}")
        logger.info(f"[EXECUTION]   Ready to receive execution requests")
    
    async def execute_snipe(
        self,
        token_mint: str,
        amount_usd: float,
        slippage_bps: int = 50,
        request_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str, ExecutionRequest]:
        """
        Queue a snipe execution request
        
        HIGH-LEVEL FLOW:
        1. Create execution request
        2. Validate inputs
        3. Queue for execution
        4. Route to external bundler service
        5. Return status
        
        Args:
            token_mint: Solana token mint address
            amount_usd: USD amount to invest
            slippage_bps: Slippage in basis points
            request_id: Optional unique request ID
            metadata: Optional context (from DEX Hunter scores, etc.)
            
        Returns:
            (success, message, ExecutionRequest)
        """
        if not request_id:
            request_id = f"snipe_{int(datetime.now().timestamp() * 1000)}"
        
        logger.info(f"[EXECUTION] 📨 Snipe request: {token_mint[:8]}... for ${amount_usd:.2f}")
        
        # Create request
        req = ExecutionRequest(
            request_id=request_id,
            token_mint=token_mint,
            strategy=ExecutionStrategy.SNIPE,
            amount_usd=amount_usd,
            slippage_bps=slippage_bps,
            metadata=metadata or {}
        )
        
        # Queue it
        self.execution_queue[request_id] = req
        self.stats['total_requests'] += 1
        self.stats['pending'] += 1
        
        # Route to external service if configured
        if self.snipe_webhook_url:
            try:
                success, message = await self._route_to_webhook(
                    self.snipe_webhook_url,
                    req,
                    service_name="Snipe Executor"
                )
                return success, message, req
            except Exception as e:
                logger.error(f"[EXECUTION] Error routing snipe request: {e}")
                return False, f"Routing failed: {e}", req
        else:
            logger.warning(f"[EXECUTION] ⚠️ Snipe webhook not configured - request queued only")
            return True, "Request queued (awaiting execution service configuration)", req
    
    async def execute_arbitrage(
        self,
        token_mint: str,
        amount_usd: float,
        request_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str, ExecutionRequest]:
        """
        Queue an arbitrage execution request
        
        Args:
            token_mint: Solana token mint address
            amount_usd: USD amount to trade
            request_id: Optional unique request ID
            metadata: Optional context
            
        Returns:
            (success, message, ExecutionRequest)
        """
        if not request_id:
            request_id = f"arb_{int(datetime.now().timestamp() * 1000)}"
        
        logger.info(f"[EXECUTION] 📨 Arbitrage request: {token_mint[:8]}... for ${amount_usd:.2f}")
        
        # Create request
        req = ExecutionRequest(
            request_id=request_id,
            token_mint=token_mint,
            strategy=ExecutionStrategy.ARBITRAGE,
            amount_usd=amount_usd,
            metadata=metadata or {}
        )
        
        # Queue it
        self.execution_queue[request_id] = req
        self.stats['total_requests'] += 1
        self.stats['pending'] += 1
        
        # Route to external service if configured
        if self.arbitrage_webhook_url:
            try:
                success, message = await self._route_to_webhook(
                    self.arbitrage_webhook_url,
                    req,
                    service_name="Arbitrage Executor"
                )
                return success, message, req
            except Exception as e:
                logger.error(f"[EXECUTION] Error routing arbitrage request: {e}")
                return False, f"Routing failed: {e}", req
        else:
            logger.warning(f"[EXECUTION] ⚠️ Arbitrage webhook not configured - request queued only")
            return True, "Request queued (awaiting execution service configuration)", req
    
    async def _route_to_webhook(
        self,
        webhook_url: str,
        request: ExecutionRequest,
        service_name: str = "Execution Service"
    ) -> Tuple[bool, str]:
        """
        Route execution request to external webhook service
        
        This is the integration point for Jito, Solayer, or custom bundler services
        
        Args:
            webhook_url: Target webhook URL
            request: ExecutionRequest to send
            service_name: Service name for logging
            
        Returns:
            (success, message)
        """
        try:
            logger.debug(f"[EXECUTION] 🌐 Routing to {service_name}: {webhook_url}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    webhook_url,
                    json=request.to_dict(),
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'Sentient-Trader-DEX-Hunter'
                    }
                )
                response.raise_for_status()
                
                # Parse response
                result_data = response.json()
                
                logger.info(
                    f"[EXECUTION] ✅ {service_name} accepted: "
                    f"{request.request_id} → {result_data.get('status', 'QUEUED')}"
                )
                
                return True, f"Request routed to {service_name}"
                
        except httpx.RequestError as e:
            logger.error(f"[EXECUTION] ❌ {service_name} connection failed: {e}")
            return False, f"Connection error: {e}"
        except Exception as e:
            logger.error(f"[EXECUTION] ❌ {service_name} error: {e}")
            return False, f"Service error: {e}"
    
    def get_status(self, request_id: str) -> Optional[ExecutionResult]:
        """
        Get execution status for a request
        
        Args:
            request_id: Execution request ID
            
        Returns:
            ExecutionResult or None if not found
        """
        return self.execution_results.get(request_id)
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            'stats': self.stats.copy(),
            'queued_requests': len(self.execution_queue),
            'completed_requests': len(self.execution_results),
            'snipe_webhook_configured': bool(self.snipe_webhook_url),
            'arbitrage_webhook_configured': bool(self.arbitrage_webhook_url)
        }
    
    def configure_webhook(
        self,
        service_type: str,  # 'snipe' or 'arbitrage'
        webhook_url: str
    ) -> bool:
        """
        Configure or update webhook URL at runtime
        
        Args:
            service_type: 'snipe' or 'arbitrage'
            webhook_url: Webhook URL to configure
            
        Returns:
            True if successful
        """
        try:
            if service_type.lower() == 'snipe':
                self.snipe_webhook_url = webhook_url
                logger.info(f"[EXECUTION] ✅ Snipe webhook configured: {webhook_url}")
                return True
            elif service_type.lower() == 'arbitrage':
                self.arbitrage_webhook_url = webhook_url
                logger.info(f"[EXECUTION] ✅ Arbitrage webhook configured: {webhook_url}")
                return True
            else:
                logger.warning(f"[EXECUTION] Unknown service type: {service_type}")
                return False
        except Exception as e:
            logger.error(f"[EXECUTION] Error configuring webhook: {e}")
            return False


# Singleton instance
_execution_webhook: Optional[DexExecutionWebhook] = None


def get_dex_execution_webhook() -> DexExecutionWebhook:
    """Get or create DEX execution webhook singleton"""
    global _execution_webhook
    if _execution_webhook is None:
        _execution_webhook = DexExecutionWebhook()
    return _execution_webhook


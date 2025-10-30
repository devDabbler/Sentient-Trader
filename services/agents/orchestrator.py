"""Agent orchestrator - coordinates all agents"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from services.event_bus import EventBus, get_event_bus
from services.intraday_data_service import IntradayDataService, get_data_service
from services.cash_manager import CashManager, CashManagerConfig
from services.risk_limits import RiskManager, get_risk_manager
from services.journal_service import JournalService, get_journal_service
from services.agents.messages import Timeframe
from services.agents.data_agent import DataAgent
from services.agents.setup_agent import SetupAgent
from services.agents.risk_agent import RiskAgent
from services.agents.execution_agent import ExecutionAgent
from services.agents.journal_agent import JournalAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates all agents and manages their lifecycle.
    """
    
    def __init__(
        self,
        symbols: List[str],
        tradier_client,
        initial_settled_cash: float = 10000.0,
        account_equity: float = 10000.0,
        cash_buckets: int = 3,
        t_plus_days: int = 2
    ):
        """
        Args:
            symbols: List of symbols to trade
            tradier_client: Broker client
            initial_settled_cash: Initial settled cash
            account_equity: Total account equity
            cash_buckets: Number of cash buckets for rotation
            t_plus_days: Settlement period (T+2 for stocks)
        """
        self.symbols = symbols
        self.tradier_client = tradier_client
        
        # Initialize core services
        self.event_bus = get_event_bus()
        self.data_service = get_data_service()
        self.risk_manager = get_risk_manager()
        self.journal_service = get_journal_service()
        
        # Initialize cash manager
        cash_config = CashManagerConfig(
            initial_settled_cash=initial_settled_cash,
            num_buckets=cash_buckets,
            t_plus_days=t_plus_days,
            use_settled_only=True
        )
        self.cash_manager = CashManager(cash_config)
        
        # Create agents
        self.data_agent = DataAgent(
            event_bus=self.event_bus,
            data_service=self.data_service,
            symbols=symbols,
            timeframes=[Timeframe.MIN_5, Timeframe.MIN_15, Timeframe.MIN_60]
        )
        
        self.setup_agent = SetupAgent(
            event_bus=self.event_bus,
            data_service=self.data_service
        )
        
        self.risk_agent = RiskAgent(
            event_bus=self.event_bus,
            cash_manager=self.cash_manager,
            risk_manager=self.risk_manager,
            account_equity=account_equity
        )
        
        self.execution_agent = ExecutionAgent(
            event_bus=self.event_bus,
            tradier_client=tradier_client,
            max_retries=3,
            retry_delay_seconds=2.0
        )
        
        self.journal_agent = JournalAgent(
            event_bus=self.event_bus,
            journal_service=self.journal_service,
            cash_manager=self.cash_manager,
            risk_manager=self.risk_manager
        )
        
        self._running = False
    
    async def start(self):
        """Start all agents"""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        logger.info("Starting AgentOrchestrator")
        
        # Start agents in order
        await self.data_agent.start()
        await self.setup_agent.start()
        await self.risk_agent.start()
        await self.execution_agent.start()
        await self.journal_agent.start()
        
        logger.info("All agents started successfully")
    
    async def stop(self):
        """Stop all agents"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping AgentOrchestrator")
        
        # Stop agents in reverse order
        await self.journal_agent.stop()
        await self.execution_agent.stop()
        await self.risk_agent.stop()
        await self.setup_agent.stop()
        await self.data_agent.stop()
        
        logger.info("All agents stopped")
    
    async def run(self):
        """Run orchestrator indefinitely"""
        await self.start()
        
        try:
            # Keep running until interrupted
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        finally:
            await self.stop()
    
    def get_status(self) -> dict:
        """Get orchestrator status"""
        return {
            'running': self._running,
            'symbols': self.symbols,
            'settled_cash': self.cash_manager.get_settled_cash(),
            'risk_state': self.risk_manager.get_state_summary(),
            'event_stats': self.event_bus.get_stats()
        }
    
    def get_journal_stats(self, days: int = 30):
        """Get journal statistics"""
        from datetime import datetime, timedelta
        start_date = datetime.now() - timedelta(days=days)
        return self.journal_service.get_statistics(start_date=start_date)


async def run_orchestrator(
    symbols: List[str],
    tradier_client,
    initial_cash: float = 10000.0,
    account_equity: float = 10000.0
):
    """
    Helper function to run orchestrator.
    
    Example:
        symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
        await run_orchestrator(symbols, tradier_client, 10000.0, 10000.0)
    """
    orchestrator = AgentOrchestrator(
        symbols=symbols,
        tradier_client=tradier_client,
        initial_settled_cash=initial_cash,
        account_equity=account_equity
    )
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await orchestrator.stop()


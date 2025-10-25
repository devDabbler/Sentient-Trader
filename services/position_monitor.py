"""
Position Monitor - Tracks active trades and generates alerts for position updates.

Monitors positions from Tradier and sends alerts for:
- Position opened/closed
- Profit targets hit
- Stop losses triggered
- Significant P&L changes
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from models.alerts import AlertType, AlertPriority, TradingAlert
from services.alert_system import AlertSystem

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Tracks the state of a position for alert generation"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    last_alert_time: Optional[datetime] = None
    last_alert_pnl_pct: float = 0.0
    stop_loss: Optional[float] = None
    profit_target: Optional[float] = None
    
    @property
    def pnl_dollars(self) -> float:
        """Calculate P&L in dollars"""
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float:
        """Calculate P&L as percentage"""
        if self.entry_price == 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable"""
        return self.pnl_dollars > 0


class PositionMonitor:
    """Monitors active positions and generates alerts"""
    
    def __init__(self, alert_system: AlertSystem, tradier_client=None):
        """
        Initialize position monitor
        
        Args:
            alert_system: Alert system to send notifications
            tradier_client: Tradier client for fetching positions
        """
        self.alert_system = alert_system
        self.tradier_client = tradier_client
        self.tracked_positions: Dict[str, PositionState] = {}
        
        # Alert thresholds
        self.pnl_alert_threshold = 5.0  # Alert every 5% P&L change
        self.significant_loss_threshold = -10.0  # Alert on 10% loss
        self.significant_gain_threshold = 15.0  # Alert on 15% gain
    
    def set_client(self, tradier_client):
        """Set or update the Tradier client"""
        self.tradier_client = tradier_client
    
    def update_positions(self) -> Tuple[bool, List[TradingAlert]]:
        """
        Fetch current positions and check for alert conditions.
        
        Returns:
            (success, list of alerts generated)
        """
        if not self.tradier_client:
            logger.warning("No Tradier client configured")
            return False, []
        
        try:
            # Fetch current positions from Tradier
            success, positions_data = self.tradier_client.get_positions()
            
            if not success:
                logger.error("Failed to fetch positions from Tradier")
                return False, []
            
            # Parse positions
            current_symbols = set()
            alerts = []
            
            # Handle Tradier response format
            positions = []
            if isinstance(positions_data, dict):
                if 'positions' in positions_data:
                    pos_data = positions_data['positions']
                    if isinstance(pos_data, dict) and 'position' in pos_data:
                        positions = pos_data['position']
                        # Ensure it's a list
                        if isinstance(positions, dict):
                            positions = [positions]
                    elif isinstance(pos_data, list):
                        positions = pos_data
            elif isinstance(positions_data, list):
                positions = positions_data
            
            # Process each position
            for pos in positions:
                symbol = pos.get('symbol', '')
                if not symbol:
                    continue
                
                current_symbols.add(symbol)
                quantity = float(pos.get('quantity', 0))
                cost_basis = float(pos.get('cost_basis', 0))
                
                # Calculate entry price
                entry_price = cost_basis / quantity if quantity != 0 else 0
                
                # Get current price from position data or fetch quote
                current_price = float(pos.get('last', entry_price))
                if current_price == 0 or current_price == entry_price:
                    # Try to get latest quote
                    quote = self.tradier_client.get_quote(symbol)
                    if quote and 'last' in quote:
                        current_price = float(quote['last'])
                
                # Check if this is a new position
                if symbol not in self.tracked_positions:
                    # New position opened
                    position_state = PositionState(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=entry_price,
                        current_price=current_price,
                        entry_time=datetime.now()
                    )
                    self.tracked_positions[symbol] = position_state
                    
                    # Generate position opened alert
                    alert = self._create_position_opened_alert(position_state)
                    alerts.append(alert)
                    self.alert_system.trigger_alert(alert)
                    
                else:
                    # Update existing position
                    position_state = self.tracked_positions[symbol]
                    position_state.current_price = current_price
                    position_state.quantity = quantity
                    
                    # Check for alert conditions
                    position_alerts = self._check_position_alerts(position_state)
                    alerts.extend(position_alerts)
            
            # Check for closed positions
            closed_symbols = set(self.tracked_positions.keys()) - current_symbols
            for symbol in closed_symbols:
                position_state = self.tracked_positions[symbol]
                alert = self._create_position_closed_alert(position_state)
                alerts.append(alert)
                self.alert_system.trigger_alert(alert)
                del self.tracked_positions[symbol]
            
            logger.info(f"Position update: {len(current_symbols)} open, {len(alerts)} alerts generated")
            return True, alerts
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}", exc_info=True)
            return False, []
    
    def _check_position_alerts(self, position: PositionState) -> List[TradingAlert]:
        """Check if position should trigger any alerts"""
        alerts = []
        
        # Check stop loss
        if position.stop_loss and position.current_price <= position.stop_loss:
            alert = self._create_stop_loss_alert(position)
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
            position.last_alert_time = datetime.now()
        
        # Check profit target
        if position.profit_target and position.current_price >= position.profit_target:
            alert = self._create_profit_target_alert(position)
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
            position.last_alert_time = datetime.now()
        
        # Check significant P&L changes
        pnl_pct = position.pnl_percent
        
        # Alert on significant loss
        if pnl_pct <= self.significant_loss_threshold:
            alert = self._create_pnl_alert(position, "SIGNIFICANT_LOSS")
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
            position.last_alert_pnl_pct = pnl_pct
            position.last_alert_time = datetime.now()
        
        # Alert on significant gain
        elif pnl_pct >= self.significant_gain_threshold:
            alert = self._create_pnl_alert(position, "SIGNIFICANT_GAIN")
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
            position.last_alert_pnl_pct = pnl_pct
            position.last_alert_time = datetime.now()
        
        # Alert on threshold changes
        elif abs(pnl_pct - position.last_alert_pnl_pct) >= self.pnl_alert_threshold:
            alert = self._create_pnl_alert(position, "UPDATE")
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
            position.last_alert_pnl_pct = pnl_pct
            position.last_alert_time = datetime.now()
        
        return alerts
    
    def _create_position_opened_alert(self, position: PositionState) -> TradingAlert:
        """Create alert for newly opened position"""
        return TradingAlert(
            ticker=position.symbol,
            alert_type=AlertType.POSITION_OPENED,
            priority=AlertPriority.HIGH,
            message=f"✅ Position OPENED: {abs(position.quantity):.0f} shares @ ${position.entry_price:.2f}",
            confidence_score=100.0,
            details={
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "total_cost": position.entry_price * abs(position.quantity),
                "entry_time": position.entry_time.isoformat()
            }
        )
    
    def _create_position_closed_alert(self, position: PositionState) -> TradingAlert:
        """Create alert for closed position"""
        pnl_dollars = position.pnl_dollars
        pnl_percent = position.pnl_percent
        emoji = "🟢" if pnl_dollars >= 0 else "🔴"
        
        return TradingAlert(
            ticker=position.symbol,
            alert_type=AlertType.POSITION_CLOSED,
            priority=AlertPriority.CRITICAL,
            message=f"{emoji} Position CLOSED: {pnl_percent:+.2f}% (${pnl_dollars:+,.2f})",
            confidence_score=100.0,
            details={
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "exit_price": position.current_price,
                "pnl_dollars": pnl_dollars,
                "pnl_percent": pnl_percent,
                "hold_time": str(datetime.now() - position.entry_time)
            }
        )
    
    def _create_profit_target_alert(self, position: PositionState) -> TradingAlert:
        """Create alert for profit target hit"""
        return TradingAlert(
            ticker=position.symbol,
            alert_type=AlertType.PROFIT_TARGET,
            priority=AlertPriority.CRITICAL,
            message=f"🎯 PROFIT TARGET HIT: ${position.current_price:.2f} (Target: ${position.profit_target:.2f})",
            confidence_score=100.0,
            details={
                "current_price": position.current_price,
                "profit_target": position.profit_target,
                "pnl_dollars": position.pnl_dollars,
                "pnl_percent": position.pnl_percent
            }
        )
    
    def _create_stop_loss_alert(self, position: PositionState) -> TradingAlert:
        """Create alert for stop loss hit"""
        return TradingAlert(
            ticker=position.symbol,
            alert_type=AlertType.STOP_LOSS_HIT,
            priority=AlertPriority.CRITICAL,
            message=f"🛑 STOP LOSS HIT: ${position.current_price:.2f} (Stop: ${position.stop_loss:.2f})",
            confidence_score=100.0,
            details={
                "current_price": position.current_price,
                "stop_loss": position.stop_loss,
                "pnl_dollars": position.pnl_dollars,
                "pnl_percent": position.pnl_percent
            }
        )
    
    def _create_pnl_alert(self, position: PositionState, alert_subtype: str) -> TradingAlert:
        """Create alert for P&L updates"""
        pnl_dollars = position.pnl_dollars
        pnl_percent = position.pnl_percent
        
        if alert_subtype == "SIGNIFICANT_LOSS":
            emoji = "⚠️"
            priority = AlertPriority.HIGH
            message = f"{emoji} SIGNIFICANT LOSS: {pnl_percent:.2f}% (${pnl_dollars:,.2f})"
        elif alert_subtype == "SIGNIFICANT_GAIN":
            emoji = "🚀"
            priority = AlertPriority.HIGH
            message = f"{emoji} SIGNIFICANT GAIN: +{pnl_percent:.2f}% (+${pnl_dollars:,.2f})"
        else:
            emoji = "📊"
            priority = AlertPriority.MEDIUM
            message = f"{emoji} Position Update: {pnl_percent:+.2f}% (${pnl_dollars:+,.2f})"
        
        return TradingAlert(
            ticker=position.symbol,
            alert_type=AlertType.POSITION_UPDATE,
            priority=priority,
            message=message,
            confidence_score=100.0,
            details={
                "current_price": position.current_price,
                "entry_price": position.entry_price,
                "pnl_dollars": pnl_dollars,
                "pnl_percent": pnl_percent,
                "quantity": position.quantity
            }
        )
    
    def set_stop_loss(self, symbol: str, stop_price: float) -> bool:
        """Set stop loss for a position"""
        if symbol in self.tracked_positions:
            self.tracked_positions[symbol].stop_loss = stop_price
            logger.info(f"Set stop loss for {symbol} at ${stop_price:.2f}")
            return True
        return False
    
    def set_profit_target(self, symbol: str, target_price: float) -> bool:
        """Set profit target for a position"""
        if symbol in self.tracked_positions:
            self.tracked_positions[symbol].profit_target = target_price
            logger.info(f"Set profit target for {symbol} at ${target_price:.2f}")
            return True
        return False
    
    def get_position_summary(self) -> Dict:
        """Get summary of all tracked positions"""
        total_pnl = sum(pos.pnl_dollars for pos in self.tracked_positions.values())
        profitable_count = sum(1 for pos in self.tracked_positions.values() if pos.is_profitable)
        
        return {
            "total_positions": len(self.tracked_positions),
            "profitable_positions": profitable_count,
            "losing_positions": len(self.tracked_positions) - profitable_count,
            "total_pnl": total_pnl,
            "positions": {
                symbol: {
                    "pnl_dollars": pos.pnl_dollars,
                    "pnl_percent": pos.pnl_percent,
                    "quantity": pos.quantity,
                    "current_price": pos.current_price
                }
                for symbol, pos in self.tracked_positions.items()
            }
        }


# Global position monitor instance
_global_position_monitor = None

def get_position_monitor(alert_system=None, tradier_client=None) -> PositionMonitor:
    """Get or create global position monitor instance"""
    global _global_position_monitor
    if _global_position_monitor is None:
        from services.alert_system import get_alert_system
        alert_sys = alert_system or get_alert_system()
        _global_position_monitor = PositionMonitor(alert_sys, tradier_client)
    elif tradier_client and _global_position_monitor.tradier_client is None:
        _global_position_monitor.set_client(tradier_client)
    return _global_position_monitor

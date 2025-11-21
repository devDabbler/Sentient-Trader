"""
ORB+FVG Strategy - Discord Alerts & Journaling Integration

Integrates the 15-min ORB+FVG strategy with:
- Discord webhook alerts
- Unified trade journal
- Alert system
"""

from loguru import logger
from datetime import datetime
from typing import Optional
import os

from services.orb_fvg_strategy import ORBFVGSignal
from services.unified_trade_journal import UnifiedTradeJournal, UnifiedTradeEntry, TradeType, TradeStatus
from services.alert_system import AlertSystem
from models.alerts import TradingAlert, AlertType, AlertPriority
from src.integrations.discord_webhook import send_discord_alert


class ORBFVGAlertManager:
    """Manages alerts and journaling for ORB+FVG trades"""
    
    def __init__(self, 
                 journal: Optional[UnifiedTradeJournal] = None,
                 alert_system: Optional[AlertSystem] = None):
        """
        Initialize alert manager
        
        Args:
            journal: UnifiedTradeJournal instance (creates new if None)
            alert_system: AlertSystem instance (creates new if None)
        """
        self.journal = journal or UnifiedTradeJournal()
        self.alert_system = alert_system or AlertSystem()
    
    def send_signal_alert(self, signal: ORBFVGSignal):
        """
        Send Discord alert for ORB+FVG signal
        
        Args:
            signal: ORBFVGSignal to alert on
        """
        try:
            # Determine priority based on confidence
            if signal.confidence >= 80:
                priority = AlertPriority.CRITICAL
            elif signal.confidence >= 70:
                priority = AlertPriority.HIGH
            elif signal.confidence >= 60:
                priority = AlertPriority.MEDIUM
            else:
                priority = AlertPriority.LOW
            
            # Create alert message
            message = (
                f"{'🟢 LONG' if signal.signal_type == 'LONG' else '🔴 SHORT'} Setup - "
                f"15-Min ORB+FVG Strategy\n"
                f"Entry: ${signal.entry_price:.2f} | "
                f"Target: ${signal.target_price:.2f} ({signal.risk_reward_ratio:.1f}R) | "
                f"Stop: ${signal.stop_loss:.2f}"
            )
            
            # Build detailed information
            details = {
                'entry_price': signal.entry_price,
                'target_price': signal.target_price,
                'stop_loss': signal.stop_loss,
                'risk_reward': signal.risk_reward_ratio,
                'position_size': f"Risk 1-2% of account (${signal.risk_amount:.2f} per trade)",
                'reasoning': self._build_reasoning(signal),
                'strategy': '15-Min ORB + FVG',
                'orb_high': signal.orb_high,
                'orb_low': signal.orb_low,
                'orb_range': f"{signal.orb_range_pct:.2f}%",
                'fvg_present': 'Yes' if signal.fvg else 'No',
                'fvg_aligned': 'Yes' if signal.fvg_alignment else 'No',
                'volume': f"{signal.volume_ratio:.1f}x average",
                'time': signal.timestamp.strftime('%I:%M %p ET')
            }
            
            # Create TradingAlert
            alert = TradingAlert(
                ticker=signal.symbol,
                alert_type=AlertType.BUY_SIGNAL if signal.signal_type == 'LONG' else AlertType.SELL_SIGNAL,
                priority=priority,
                message=message,
                confidence_score=signal.confidence,
                details=details,
                timestamp=signal.timestamp
            )
            
            # Trigger alert (will send to Discord via callback)
            self.alert_system.trigger_alert(alert)
            
            logger.info(f"✅ Discord alert sent for {signal.symbol} ORB+FVG {signal.signal_type} signal")
            
        except Exception as e:
            logger.error("Error sending ORB+FVG alert for {signal.symbol}: {}", str(e), exc_info=True)
    
    def log_trade_entry(self, 
                       signal: ORBFVGSignal,
                       actual_entry_price: Optional[float] = None,
                       quantity: float = 0,
                       broker: str = "TRADIER") -> str:
        """
        Log trade entry to unified journal
        
        Args:
            signal: ORBFVGSignal that triggered the trade
            actual_entry_price: Actual fill price (uses signal entry if None)
            quantity: Number of shares/contracts
            broker: Broker used (TRADIER, IBKR, etc.)
            
        Returns:
            trade_id for the logged entry
        """
        try:
            entry_price = actual_entry_price or signal.entry_price
            position_size = entry_price * quantity if quantity > 0 else 0
            
            # Calculate risk percentages
            risk_pct = (signal.risk_amount / entry_price) * 100
            reward_pct = (signal.reward_amount / entry_price) * 100
            
            # Generate trade ID
            trade_id = f"ORB_FVG_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Create trade entry
            trade_entry = UnifiedTradeEntry(
                trade_id=trade_id,
                trade_type=TradeType.STOCK.value,
                symbol=signal.symbol,
                side='BUY' if signal.signal_type == 'LONG' else 'SELL',
                entry_time=signal.timestamp,
                entry_price=entry_price,
                quantity=quantity,
                position_size_usd=position_size,
                
                # Risk management
                stop_loss=signal.stop_loss,
                take_profit=signal.target_price,
                risk_pct=risk_pct,
                reward_pct=reward_pct,
                risk_reward_ratio=signal.risk_reward_ratio,
                
                # Strategy
                strategy='ORB_FVG_15MIN',
                setup_type='ORB_BREAKOUT',
                timeframe='15MIN',
                
                # Market conditions at entry
                rsi_entry=None,  # Can be added if analysis object available
                volume_change_entry=signal.volume_ratio,
                trend_entry=signal.signal_type,
                
                # Metadata
                broker=broker,
                notes=self._build_trade_notes(signal),
                tags=['ORB', 'FVG', '15MIN', signal.signal_type],
                status=TradeStatus.OPEN.value
            )
            
            # Add to journal
            success = self.journal.log_trade_entry(trade_entry)
            
            if not success:
                logger.error(f"Failed to log trade entry: {trade_id}")
                return ""
            
            logger.info(f"📓 Trade entry logged: {trade_id} - {signal.symbol} {signal.signal_type}")
            
            # Send trade execution alert to Discord
            self._send_execution_alert(signal, trade_entry)
            
            return trade_id
            
        except Exception as e:
            logger.error("Error logging trade entry for {signal.symbol}: {}", str(e), exc_info=True)
            return ""
    
    def log_trade_exit(self,
                      trade_id: str,
                      exit_price: float,
                      exit_reason: str = "TARGET_HIT",
                      exit_time: Optional[datetime] = None):
        """
        Log trade exit to unified journal
        
        Args:
            trade_id: Trade ID from log_trade_entry
            exit_price: Actual exit price
            exit_reason: Reason for exit (TARGET_HIT, STOP_HIT, MANUAL, etc.)
            exit_time: Exit timestamp (uses now if None)
        """
        try:
            exit_time = exit_time or datetime.now()
            
            # Update trade in journal
            self.journal.close_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_reason=exit_reason
            )
            
            logger.info(f"📓 Trade exit logged: {trade_id} - {exit_reason} at ${exit_price:.2f}")
            
        except Exception as e:
            logger.error("Error logging trade exit for {trade_id}: {}", str(e), exc_info=True)
    
    def _build_reasoning(self, signal: ORBFVGSignal) -> str:
        """Build reasoning text for signal"""
        reasoning_parts = [
            f"15-Minute Opening Range Breakout ({signal.signal_type})",
            f"ORB Range: ${signal.orb_low:.2f} - ${signal.orb_high:.2f} ({signal.orb_range_pct:.2f}%)",
            f"Breakout confirmed with {signal.volume_ratio:.1f}x average volume"
        ]
        
        if signal.fvg:
            reasoning_parts.append(
                f"Fair Value Gap detected and aligned ({signal.fvg.gap_type}, strength: {signal.fvg.strength:.0f}/100)"
            )
        else:
            reasoning_parts.append("No FVG confirmation - breakout-only signal (use caution)")
        
        reasoning_parts.extend([
            f"Risk/Reward: {signal.risk_reward_ratio:.1f}:1",
            f"Confidence: {signal.confidence:.1f}%",
            f"Trading window: 9:45-11:00 AM optimal"
        ])
        
        return " | ".join(reasoning_parts)
    
    def _build_trade_notes(self, signal: ORBFVGSignal) -> str:
        """Build notes for trade journal entry"""
        notes = [
            f"15-Min ORB+FVG Strategy - {signal.signal_type} Signal",
            f"ORB High: ${signal.orb_high:.2f}, Low: ${signal.orb_low:.2f}",
            f"Range: {signal.orb_range_pct:.2f}%",
            f"Volume: {signal.volume_ratio:.1f}x average",
        ]
        
        if signal.fvg:
            notes.append(f"FVG: {signal.fvg.gap_type.title()} (${signal.fvg.bottom:.2f}-${signal.fvg.top:.2f})")
            notes.append(f"FVG Strength: {signal.fvg.strength:.0f}/100")
        else:
            notes.append("No FVG - Breakout only")
        
        notes.append(f"Confidence: {signal.confidence:.1f}%")
        notes.append(f"Time: {signal.timestamp.strftime('%I:%M %p ET')}")
        
        return " | ".join(notes)
    
    def _send_execution_alert(self, signal: ORBFVGSignal, trade_entry: UnifiedTradeEntry):
        """Send Discord alert for trade execution"""
        try:
            message = (
                f"✅ **TRADE EXECUTED** - {signal.symbol}\n"
                f"Strategy: 15-Min ORB+FVG\n"
                f"Direction: {'🟢 LONG' if signal.signal_type == 'LONG' else '🔴 SHORT'}\n"
                f"Filled: ${trade_entry.entry_price:.2f}"
            )
            
            details = {
                'order_id': trade_entry.trade_id,
                'price': trade_entry.entry_price,
                'quantity': trade_entry.quantity,
                'direction': signal.signal_type,
                'position_size': trade_entry.position_size_usd,
                'stop_loss': signal.stop_loss,
                'target': signal.target_price,
                'risk_reward': signal.risk_reward_ratio,
                'strategy': '15-Min ORB+FVG',
                'broker': trade_entry.broker
            }
            
            alert = TradingAlert(
                ticker=signal.symbol,
                alert_type=AlertType.TRADE_EXECUTED,
                priority=AlertPriority.HIGH,
                message=message,
                confidence_score=signal.confidence,
                details=details,
                timestamp=trade_entry.entry_time
            )
            
            self.alert_system.trigger_alert(alert)
            
            logger.info(f"✅ Trade execution alert sent for {signal.symbol}")
            
        except Exception as e:
            logger.error("Error sending execution alert: {}", str(e), exc_info=True)
    
    def get_orb_fvg_stats(self) -> dict:
        """
        Get statistics for ORB+FVG trades from journal
        
        Returns:
            Dictionary with strategy-specific stats
        """
        try:
            # Get all ORB+FVG trades
            all_trades = self.journal.get_trades(strategy='ORB_FVG_15MIN')
            orb_fvg_trades = all_trades
            
            if not orb_fvg_trades:
                return {
                    'total_trades': 0,
                    'message': 'No ORB+FVG trades in journal yet'
                }
            
            # Calculate stats
            closed_trades = [t for t in orb_fvg_trades if t.status == TradeStatus.CLOSED.value]
            winners = [t for t in closed_trades if t.realized_pnl and t.realized_pnl > 0]
            losers = [t for t in closed_trades if t.realized_pnl and t.realized_pnl < 0]
            
            win_rate = (len(winners) / len(closed_trades) * 100) if closed_trades else 0
            avg_win = sum(t.realized_pnl for t in winners) / len(winners) if winners else 0
            avg_loss = sum(t.realized_pnl for t in losers) / len(losers) if losers else 0
            total_pnl = sum(t.realized_pnl for t in closed_trades if t.realized_pnl)
            
            return {
                'total_trades': len(orb_fvg_trades),
                'open_trades': len([t for t in orb_fvg_trades if t.status == TradeStatus.OPEN.value]),
                'closed_trades': len(closed_trades),
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            }
            
        except Exception as e:
            logger.error("Error getting ORB+FVG stats: {}", str(e), exc_info=True)
            return {'error': str(e)}


def create_orb_fvg_alert_manager() -> ORBFVGAlertManager:
    """
    Factory function to create ORBFVGAlertManager with proper dependencies
    
    Returns:
        Configured ORBFVGAlertManager instance
    """
    try:
        # Create journal
        journal = UnifiedTradeJournal()
        
        # Create alert system
        alert_system = AlertSystem()
        
        # Create and return manager
        manager = ORBFVGAlertManager(journal=journal, alert_system=alert_system)
        
        logger.info("✅ ORB+FVG Alert Manager initialized")
        return manager
        
    except Exception as e:
        logger.error("Error creating ORB+FVG alert manager: {}", str(e), exc_info=True)
        # Return basic manager without dependencies
        return ORBFVGAlertManager()

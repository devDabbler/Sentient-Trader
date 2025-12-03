"""
Position Supabase Sync Service

Provides Supabase persistence for crypto and stock positions.
Syncs positions to cloud database for:
- Complete audit trail of all positions
- Stop loss / take profit tracking
- AI decision history
- Cross-device access to position data
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    from clients.supabase_client import get_supabase_client
except ImportError:
    get_supabase_client = None


class PositionSupabaseSync:
    """
    Handles syncing position data to Supabase for both crypto and stock positions.
    Provides a complete record of all positions with stop losses, take profits, and AI decisions.
    """
    
    def __init__(self):
        """Initialize Supabase sync service"""
        self.client = None
        self.enabled = False
        self._init_client()
    
    def _init_client(self):
        """Initialize Supabase client"""
        try:
            if get_supabase_client:
                self.client = get_supabase_client()
                if self.client:
                    self.enabled = True
                    logger.info("✅ Position Supabase Sync initialized")
                else:
                    logger.warning("⚠️ Supabase client not available - positions will only be saved locally")
            else:
                logger.warning("⚠️ Supabase client module not available")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase sync: {e}")
            self.enabled = False
    
    # =========================================================================
    # CRYPTO POSITION SYNC
    # =========================================================================
    
    def sync_crypto_position(self, position_data: Dict) -> bool:
        """
        Sync a crypto position to Supabase.
        Creates new record or updates existing one.
        
        Args:
            position_data: Dictionary with position details including:
                - trade_id, pair, side, volume
                - entry_price, entry_time, stop_loss, take_profit
                - current_price, status, etc.
        
        Returns:
            True if sync successful
        """
        if not self.enabled:
            return False
        
        try:
            trade_id = position_data.get('trade_id')
            if not trade_id:
                logger.error("Cannot sync position without trade_id")
                return False
            
            # Prepare data for Supabase
            supabase_data = {
                'trade_id': trade_id,
                'pair': position_data.get('pair'),
                'side': position_data.get('side'),
                'volume': position_data.get('volume'),
                'entry_price': position_data.get('entry_price'),
                'entry_time': self._format_datetime(position_data.get('entry_time')),
                'entry_order_id': position_data.get('entry_order_id'),
                'stop_loss': position_data.get('stop_loss'),
                'take_profit': position_data.get('take_profit'),
                'trailing_stop_pct': position_data.get('trailing_stop_pct', 2.0),
                'breakeven_trigger_pct': position_data.get('breakeven_trigger_pct', 3.0),
                'current_price': position_data.get('current_price'),
                'highest_price': position_data.get('highest_price'),
                'lowest_price': position_data.get('lowest_price'),
                'last_check_time': self._format_datetime(position_data.get('last_check_time')),
                'position_intent': position_data.get('position_intent', 'SWING'),
                'strategy': position_data.get('strategy'),
                'status': position_data.get('status', 'ACTIVE'),
                'moved_to_breakeven': position_data.get('moved_to_breakeven', False),
                'partial_exit_taken': position_data.get('partial_exit_taken', False),
                'partial_exit_pct': position_data.get('partial_exit_pct', 0.0),
                'last_ai_action': position_data.get('last_ai_action', 'HOLD'),
                'last_ai_reasoning': position_data.get('last_ai_reasoning'),
                'last_ai_confidence': position_data.get('last_ai_confidence', 0.0),
                'ai_adjustment_count': position_data.get('ai_adjustment_count', 0),
                'max_favorable_pct': position_data.get('max_favorable_pct', 0.0),
                'max_adverse_pct': position_data.get('max_adverse_pct', 0.0),
                'updated_at': datetime.now().isoformat()
            }
            
            # Remove None values
            supabase_data = {k: v for k, v in supabase_data.items() if v is not None}
            
            # Upsert (insert or update)
            result = self.client.table('crypto_positions').upsert(
                supabase_data,
                on_conflict='trade_id'
            ).execute()
            
            logger.debug(f"✅ Synced crypto position: {position_data.get('pair')} ({trade_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync crypto position: {e}")
            return False
    
    def close_crypto_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl: Optional[float] = None,
        realized_pnl_pct: Optional[float] = None
    ) -> bool:
        """
        Mark a crypto position as closed in Supabase.
        
        Args:
            trade_id: Position trade ID
            exit_price: Exit price
            exit_reason: Why position was closed
            realized_pnl: Realized P&L in USD
            realized_pnl_pct: Realized P&L percentage
        
        Returns:
            True if update successful
        """
        if not self.enabled:
            return False
        
        try:
            update_data = {
                'status': 'CLOSED',
                'exit_price': exit_price,
                'exit_time': datetime.now().isoformat(),
                'exit_reason': exit_reason,
                'updated_at': datetime.now().isoformat()
            }
            
            if realized_pnl is not None:
                update_data['realized_pnl'] = realized_pnl
            if realized_pnl_pct is not None:
                update_data['realized_pnl_pct'] = realized_pnl_pct
            
            result = self.client.table('crypto_positions').update(
                update_data
            ).eq('trade_id', trade_id).execute()
            
            logger.info(f"✅ Closed crypto position in Supabase: {trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close crypto position in Supabase: {e}")
            return False
    
    # =========================================================================
    # STOCK POSITION SYNC
    # =========================================================================
    
    def sync_stock_position(self, position_data: Dict, paper_mode: bool = True, broker: str = "UNKNOWN") -> bool:
        """
        Sync a stock position to Supabase.
        
        Args:
            position_data: Dictionary with position details
            paper_mode: Whether this is a paper trade
            broker: Broker name (TRADIER, IBKR, etc.)
        
        Returns:
            True if sync successful
        """
        if not self.enabled:
            return False
        
        try:
            trade_id = position_data.get('trade_id')
            if not trade_id:
                logger.error("Cannot sync stock position without trade_id")
                return False
            
            supabase_data = {
                'trade_id': trade_id,
                'symbol': position_data.get('symbol'),
                'side': position_data.get('side'),
                'quantity': position_data.get('quantity'),
                'entry_price': position_data.get('entry_price'),
                'entry_time': self._format_datetime(position_data.get('entry_time')),
                'order_id': position_data.get('order_id'),
                'stop_loss': position_data.get('stop_loss'),
                'take_profit': position_data.get('take_profit'),
                'trailing_stop_pct': position_data.get('trailing_stop_pct', 2.0),
                'breakeven_trigger_pct': position_data.get('breakeven_trigger_pct', 3.0),
                'current_price': position_data.get('current_price'),
                'highest_price': position_data.get('highest_price'),
                'lowest_price': position_data.get('lowest_price'),
                'last_check_time': self._format_datetime(position_data.get('last_check_time')),
                'strategy': position_data.get('strategy'),
                'reasoning': position_data.get('reasoning'),
                'confidence': position_data.get('confidence'),
                'status': position_data.get('status', 'ACTIVE'),
                'moved_to_breakeven': position_data.get('moved_to_breakeven', False),
                'partial_exit_taken': position_data.get('partial_exit_taken', False),
                'paper_mode': paper_mode,
                'broker': broker,
                'updated_at': datetime.now().isoformat()
            }
            
            # Remove None values
            supabase_data = {k: v for k, v in supabase_data.items() if v is not None}
            
            result = self.client.table('stock_positions').upsert(
                supabase_data,
                on_conflict='trade_id'
            ).execute()
            
            logger.debug(f"✅ Synced stock position: {position_data.get('symbol')} ({trade_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync stock position: {e}")
            return False
    
    def close_stock_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        exit_order_id: Optional[str] = None,
        realized_pnl: Optional[float] = None,
        realized_pnl_pct: Optional[float] = None
    ) -> bool:
        """Mark a stock position as closed in Supabase."""
        if not self.enabled:
            return False
        
        try:
            update_data = {
                'status': 'CLOSED',
                'exit_price': exit_price,
                'exit_time': datetime.now().isoformat(),
                'exit_reason': exit_reason,
                'updated_at': datetime.now().isoformat()
            }
            
            if exit_order_id:
                update_data['exit_order_id'] = exit_order_id
            if realized_pnl is not None:
                update_data['realized_pnl'] = realized_pnl
            if realized_pnl_pct is not None:
                update_data['realized_pnl_pct'] = realized_pnl_pct
            
            result = self.client.table('stock_positions').update(
                update_data
            ).eq('trade_id', trade_id).execute()
            
            logger.info(f"✅ Closed stock position in Supabase: {trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close stock position in Supabase: {e}")
            return False
    
    # =========================================================================
    # POSITION HISTORY / AUDIT TRAIL
    # =========================================================================
    
    def log_position_action(
        self,
        trade_id: str,
        asset_type: str,  # 'CRYPTO' or 'STOCK'
        symbol: str,
        action: str,  # 'ENTRY', 'STOP_UPDATED', 'EXIT', etc.
        price_at_action: Optional[float] = None,
        old_stop_loss: Optional[float] = None,
        new_stop_loss: Optional[float] = None,
        old_take_profit: Optional[float] = None,
        new_take_profit: Optional[float] = None,
        ai_confidence: Optional[float] = None,
        ai_reasoning: Optional[str] = None,
        pnl_at_action: Optional[float] = None,
        pnl_pct_at_action: Optional[float] = None,
        trigger_source: str = 'UNKNOWN'
    ) -> bool:
        """
        Log a position action to the history/audit trail.
        
        Args:
            trade_id: Position trade ID
            asset_type: 'CRYPTO' or 'STOCK'
            symbol: Trading symbol/pair
            action: Action type (ENTRY, STOP_UPDATED, TARGET_UPDATED, PARTIAL_EXIT, EXIT, etc.)
            price_at_action: Current price when action occurred
            old_stop_loss: Previous stop loss (for updates)
            new_stop_loss: New stop loss (for updates)
            ai_confidence: AI confidence if AI-triggered
            ai_reasoning: AI reasoning if AI-triggered
            trigger_source: What triggered this action (AI, MANUAL, STOP_LOSS, TAKE_PROFIT)
        
        Returns:
            True if logged successfully
        """
        if not self.enabled:
            return False
        
        try:
            history_data = {
                'trade_id': trade_id,
                'asset_type': asset_type,
                'symbol': symbol,
                'action': action,
                'action_time': datetime.now().isoformat(),
                'trigger_source': trigger_source
            }
            
            # Add optional fields
            if price_at_action is not None:
                history_data['price_at_action'] = price_at_action
            if old_stop_loss is not None:
                history_data['old_stop_loss'] = old_stop_loss
            if new_stop_loss is not None:
                history_data['new_stop_loss'] = new_stop_loss
            if old_take_profit is not None:
                history_data['old_take_profit'] = old_take_profit
            if new_take_profit is not None:
                history_data['new_take_profit'] = new_take_profit
            if ai_confidence is not None:
                history_data['ai_confidence'] = ai_confidence
            if ai_reasoning is not None:
                history_data['ai_reasoning'] = ai_reasoning[:500] if ai_reasoning else None
            if pnl_at_action is not None:
                history_data['pnl_at_action'] = pnl_at_action
            if pnl_pct_at_action is not None:
                history_data['pnl_pct_at_action'] = pnl_pct_at_action
            
            result = self.client.table('position_history').insert(history_data).execute()
            
            logger.debug(f"📝 Logged position action: {symbol} - {action}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log position action: {e}")
            return False
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def get_active_crypto_positions(self) -> List[Dict]:
        """Get all active crypto positions from Supabase."""
        if not self.enabled:
            return []
        
        try:
            result = self.client.table('crypto_positions').select('*').eq('status', 'ACTIVE').execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get active crypto positions: {e}")
            return []
    
    def get_active_stock_positions(self) -> List[Dict]:
        """Get all active stock positions from Supabase."""
        if not self.enabled:
            return []
        
        try:
            result = self.client.table('stock_positions').select('*').eq('status', 'ACTIVE').execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get active stock positions: {e}")
            return []
    
    def get_position_by_symbol(self, symbol: str, asset_type: str = 'CRYPTO') -> Optional[Dict]:
        """Get position by symbol (returns most recent active position)."""
        if not self.enabled:
            return None
        
        try:
            table = 'crypto_positions' if asset_type == 'CRYPTO' else 'stock_positions'
            field = 'pair' if asset_type == 'CRYPTO' else 'symbol'
            
            result = self.client.table(table).select('*').eq(field, symbol).eq('status', 'ACTIVE').order('entry_time', desc=True).limit(1).execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get position by symbol: {e}")
            return None
    
    def get_position_history(self, trade_id: str) -> List[Dict]:
        """Get full history for a position."""
        if not self.enabled:
            return []
        
        try:
            result = self.client.table('position_history').select('*').eq('trade_id', trade_id).order('action_time', desc=False).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get position history: {e}")
            return []
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _format_datetime(self, dt) -> Optional[str]:
        """Format datetime for Supabase."""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        if isinstance(dt, datetime):
            return dt.isoformat()
        return str(dt)


# Singleton instance
_supabase_sync_instance: Optional[PositionSupabaseSync] = None


def get_position_supabase_sync() -> PositionSupabaseSync:
    """Get or create the singleton PositionSupabaseSync instance."""
    global _supabase_sync_instance
    if _supabase_sync_instance is None:
        _supabase_sync_instance = PositionSupabaseSync()
    return _supabase_sync_instance


"""
Order Cleanup Utility
Handles cancellation of rejected and stuck orders from Tradier
"""

from loguru import logger
from typing import Tuple, List, Dict


class OrderCleanup:
    """Utility to clean up rejected and stuck orders"""
    
    def __init__(self, tradier_client):
        """
        Initialize Order Cleanup
        
        Args:
            tradier_client: TradierClient instance
        """
        self.tradier_client = tradier_client
    
    def cancel_rejected_orders(self) -> Tuple[bool, Dict]:
        """
        Cancel all rejected orders in the account
        
        Returns:
            Tuple of (success: bool, result: Dict with cancellation details)
        """
        try:
            logger.info("=" * 80)
            logger.info("🧹 CLEANING UP REJECTED ORDERS")
            logger.info("=" * 80)
            
            # Get all orders
            success, orders = self.tradier_client.get_orders(status="all")
            if not success or not orders:
                logger.warning("No orders found or error retrieving orders")
                return False, {"error": "Could not retrieve orders"}
            
            # Filter for rejected orders
            rejected_orders = [
                order for order in orders 
                if order.get('status', '').lower() == 'rejected'
            ]
            
            if not rejected_orders:
                logger.info("✅ No rejected orders found")
                return True, {"cancelled": 0, "rejected_orders": []}
            
            logger.warning(f"Found {len(rejected_orders)} rejected orders")
            
            # Cancel each rejected order
            cancelled_count = 0
            cancelled_details = []
            
            for order in rejected_orders:
                order_id = order.get('id')
                symbol = order.get('symbol')
                side = order.get('side')
                quantity = order.get('quantity')
                
                logger.info(f"Cancelling rejected order: {symbol} {side} {quantity} (ID: {order_id})")
                
                success, result = self.tradier_client.cancel_order(str(order_id))
                
                if success:
                    cancelled_count += 1
                    cancelled_details.append({
                        'order_id': order_id,
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'status': 'cancelled'
                    })
                    logger.info(f"✅ Cancelled order {order_id} for {symbol}")
                else:
                    logger.error(f"❌ Failed to cancel order {order_id}: {result}")
                    cancelled_details.append({
                        'order_id': order_id,
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error')
                    })
            
            logger.info("=" * 80)
            logger.info(f"🧹 CLEANUP COMPLETE: {cancelled_count}/{len(rejected_orders))} orders cancelled")
            logger.info("=" * 80)
            
            return True, {
                "cancelled": cancelled_count,
                "total_rejected": len(rejected_orders),
                "details": cancelled_details
            }
            
        except Exception as e:
            logger.error("Error during order cleanup: {}", str(e), exc_info=True)
            return False, {"error": str(e)}
    
    def cancel_stuck_orders(self, max_age_minutes: int = 60) -> Tuple[bool, Dict]:
        """
        Cancel orders that have been pending for too long (stuck orders)
        
        Args:
            max_age_minutes: Orders older than this (in minutes) are considered stuck
        
        Returns:
            Tuple of (success: bool, result: Dict with cancellation details)
        """
        try:
            from datetime import datetime, timedelta
            
            logger.info("=" * 80)
            logger.info(f"🧹 CLEANING UP STUCK ORDERS (older than {max_age_minutes} minutes)")
            logger.info("=" * 80)
            
            # Get all pending orders
            success, orders = self.tradier_client.get_orders(status="pending")
            if not success or not orders:
                logger.info("No pending orders found")
                return True, {"cancelled": 0, "stuck_orders": []}
            
            # Filter for stuck orders (pending for too long)
            now = datetime.utcnow()
            stuck_orders = []
            
            for order in orders:
                create_date_str = order.get('create_date', '')
                if create_date_str:
                    try:
                        # Parse ISO format date
                        create_date = datetime.fromisoformat(create_date_str.replace('Z', '+00:00'))
                        age = now - create_date.replace(tzinfo=None)
                        
                        if age > timedelta(minutes=max_age_minutes):
                            stuck_orders.append(order)
                    except:
                        pass
            
            if not stuck_orders:
                logger.info("✅ No stuck orders found")
                return True, {"cancelled": 0, "stuck_orders": []}
            
            logger.warning(f"Found {len(stuck_orders)} stuck orders")
            
            # Cancel each stuck order
            cancelled_count = 0
            cancelled_details = []
            
            for order in stuck_orders:
                order_id = order.get('id')
                symbol = order.get('symbol')
                side = order.get('side')
                quantity = order.get('quantity')
                create_date = order.get('create_date')
                
                logger.info(f"Cancelling stuck order: {symbol} {side} {quantity} (ID: {order_id}, created: {create_date})")
                
                success, result = self.tradier_client.cancel_order(str(order_id))
                
                if success:
                    cancelled_count += 1
                    cancelled_details.append({
                        'order_id': order_id,
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'create_date': create_date,
                        'status': 'cancelled'
                    })
                    logger.info(f"✅ Cancelled stuck order {order_id} for {symbol}")
                else:
                    logger.error(f"❌ Failed to cancel order {order_id}: {result}")
                    cancelled_details.append({
                        'order_id': order_id,
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'create_date': create_date,
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error')
                    })
            
            logger.info("=" * 80)
            logger.info(f"🧹 CLEANUP COMPLETE: {cancelled_count}/{len(stuck_orders))} stuck orders cancelled")
            logger.info("=" * 80)
            
            return True, {
                "cancelled": cancelled_count,
                "total_stuck": len(stuck_orders),
                "details": cancelled_details
            }
            
        except Exception as e:
            logger.error("Error during stuck order cleanup: {}", str(e), exc_info=True)
            return False, {"error": str(e)}

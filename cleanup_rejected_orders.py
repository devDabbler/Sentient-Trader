#!/usr/bin/env python3
"""
Standalone script to clean up rejected orders from Tradier
Run this when you have stuck/rejected orders that need to be cleared

Usage:
    python cleanup_rejected_orders.py
"""

import os
import sys
from dotenv import load_dotenv
from loguru import logger

# Setup logging
from utils.logging_config import setup_logging
setup_logging()

# Load environment variables
load_dotenv()

# Import clients
from src.integrations.tradier_client import TradierClient
from services.order_cleanup import OrderCleanup


def main():
    """Main cleanup function"""
    
    logger.info("=" * 80)
    logger.info("🧹 TRADIER ORDER CLEANUP UTILITY")
    logger.info("=" * 80)
    
    # Get Tradier credentials (try paper first, then production, then legacy)
    # Note: Paper is tried first for cleanup utility (safer for testing)
    tradier_account_id = (
        os.getenv('TRADIER_PAPER_ACCOUNT_ID') or 
        os.getenv('TRADIER_PROD_ACCOUNT_ID') or 
        os.getenv('TRADIER_ACCOUNT_ID')
    )
    tradier_api_key = (
        os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or 
        os.getenv('TRADIER_PROD_ACCESS_TOKEN') or 
        os.getenv('TRADIER_ACCESS_TOKEN')
    )
    
    # Determine which mode we're using
    if os.getenv('TRADIER_PAPER_ACCOUNT_ID') and os.getenv('TRADIER_PAPER_ACCESS_TOKEN'):
        mode = "PAPER (Sandbox)"
    elif os.getenv('TRADIER_PROD_ACCOUNT_ID') and os.getenv('TRADIER_PROD_ACCESS_TOKEN'):
        mode = "PRODUCTION"
    else:
        mode = "LEGACY"
    
    logger.info(f"Using {mode} credentials")
    
    if not tradier_api_key or not tradier_account_id:
        logger.error("❌ Missing Tradier credentials in .env file")
        logger.error("   Required one of:")
        logger.error("   - TRADIER_PROD_ACCOUNT_ID + TRADIER_PROD_ACCESS_TOKEN")
        logger.error("   - TRADIER_PAPER_ACCOUNT_ID + TRADIER_PAPER_ACCESS_TOKEN")
        logger.error("   - TRADIER_ACCOUNT_ID + TRADIER_ACCESS_TOKEN (legacy)")
        return False
    
    try:
        # Initialize Tradier client with correct API URL based on credentials
        logger.info("Connecting to Tradier...")
        
        # Determine API URL based on which credentials are being used
        if os.getenv('TRADIER_PAPER_ACCOUNT_ID') and os.getenv('TRADIER_PAPER_ACCESS_TOKEN'):
            api_url = os.getenv('TRADIER_PAPER_API_URL', 'https://sandbox.tradier.com')
            from src.integrations.trading_config import TradingMode
            trading_mode = TradingMode.PAPER
        else:
            api_url = os.getenv('TRADIER_PROD_API_URL', 'https://api.tradier.com')
            from src.integrations.trading_config import TradingMode
            trading_mode = TradingMode.PRODUCTION
        
        tradier_client = TradierClient(
            access_token=tradier_api_key,
            account_id=tradier_account_id,
            api_url=api_url,
            trading_mode=trading_mode
        )
        
        # Validate connection
        success, balance = tradier_client.get_account_balance()
        if not success:
            logger.error("❌ Failed to connect to Tradier")
            return False
        
        logger.info(f"✅ Connected to Tradier account {tradier_account_id}")
        logger.info(f"   Account Balance: ${balance.get('total_equity', 0):,.2f}")
        
        # Initialize cleanup utility
        cleanup = OrderCleanup(tradier_client)
        
        # Show menu
        print("\n" + "=" * 80)
        print("SELECT CLEANUP OPTION:")
        print("=" * 80)
        print("1. Cancel all REJECTED orders")
        print("2. Cancel all STUCK orders (pending for >60 minutes)")
        print("3. Cancel both rejected AND stuck orders")
        print("4. Exit without cleanup")
        print("=" * 80)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            logger.info("\n🧹 Cancelling rejected orders...")
            success, result = cleanup.cancel_rejected_orders()
            if success:
                logger.info(f"✅ Cancelled {result['cancelled']} rejected orders")
            else:
                logger.error(f"❌ Cleanup failed: {result.get('error')}")
                return False
        
        elif choice == '2':
            logger.info("\n🧹 Cancelling stuck orders...")
            success, result = cleanup.cancel_stuck_orders(max_age_minutes=60)
            if success:
                logger.info(f"✅ Cancelled {result['cancelled']} stuck orders")
            else:
                logger.error(f"❌ Cleanup failed: {result.get('error')}")
                return False
        
        elif choice == '3':
            logger.info("\n🧹 Cancelling rejected orders...")
            success1, result1 = cleanup.cancel_rejected_orders()
            
            logger.info("\n🧹 Cancelling stuck orders...")
            success2, result2 = cleanup.cancel_stuck_orders(max_age_minutes=60)
            
            if success1 and success2:
                total_cancelled = result1['cancelled'] + result2['cancelled']
                logger.info(f"\n✅ Total cancelled: {total_cancelled} orders")
            else:
                logger.error("❌ Cleanup had errors")
                return False
        
        elif choice == '4':
            logger.info("Exiting without cleanup")
            return True
        
        else:
            logger.error("Invalid choice")
            return False
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ CLEANUP COMPLETE")
        logger.info("=" * 80)
        
        # Verify cleanup
        logger.info("\nVerifying cleanup...")
        success, orders = tradier_client.get_orders(status="rejected")
        if success:
            remaining_rejected = len(orders) if orders else 0
            logger.info(f"Remaining rejected orders: {remaining_rejected}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

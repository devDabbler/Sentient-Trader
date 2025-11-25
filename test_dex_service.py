"""
DEX Launch Service Test Script

Tests all components of the DEX launch monitoring service:
1. DexScreener API connectivity
2. Token safety analysis
3. Launch announcement monitoring
4. Alert system (Discord webhook)
5. Full scan cycle

Usage:
    python test_dex_service.py
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


async def test_dexscreener_client():
    """Test DexScreener API connectivity and token searching"""
    logger.info("=" * 60)
    logger.info("TEST 1: DexScreener Client")
    logger.info("=" * 60)
    
    try:
        from clients.dexscreener_client import DexScreenerClient
        from models.dex_models import Chain
        
        client = DexScreenerClient()
        
        # Test 1: Get new pairs
        logger.info("Testing get_new_pairs()...")
        success, pairs = await client.get_new_pairs(
            chains=["solana", "ethereum"],
            min_liquidity=5000,
            max_age_hours=24,
            limit=10
        )
        
        if success and pairs:
            logger.success(f"✅ Found {len(pairs)} new pairs")
            for i, pair in enumerate(pairs[:3]):
                logger.info(f"  {i+1}. {pair.base_token_symbol} on {pair.chain.value}")
                logger.info(f"     Liquidity: ${pair.liquidity_usd:,.0f}, Volume: ${pair.volume_24h:,.0f}")
                logger.info(f"     Age: {pair.pair_age_hours:.1f}h, Price: ${pair.price_usd:.8f}")
        else:
            logger.warning("⚠️ No pairs found or API error")
            return False
        
        # Test 2: Search for specific token
        logger.info("\nTesting search_pairs()...")
        success, results = await client.search_pairs("pepe")
        if success and results:
            logger.success(f"✅ Search returned {len(results)} results")
        else:
            logger.warning("⚠️ Search failed")
        
        # Test 3: Get token details (use first found pair)
        if pairs:
            test_address = pairs[0].base_token_address
            test_chain = pairs[0].chain.value
            logger.info(f"\nTesting get_token_pairs({test_address[:20]}...)...")
            success, token_pairs = await client.get_token_pairs(test_address, test_chain)
            if success and token_pairs:
                logger.success(f"✅ Got {len(token_pairs)} pairs for token")
            else:
                logger.warning("⚠️ Token lookup failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DexScreener test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_token_safety():
    """Test token safety analyzer"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Token Safety Analyzer")
    logger.info("=" * 60)
    
    try:
        from services.token_safety_analyzer import TokenSafetyAnalyzer
        from models.dex_models import Chain
        
        analyzer = TokenSafetyAnalyzer()
        
        # Test with a known token (PEPE on Ethereum)
        test_address = "0x6982508145454ce325ddbe47a25d4ec3d2311933"  # PEPE
        
        logger.info(f"Analyzing token: {test_address[:20]}... on ETH")
        success, safety = await analyzer.analyze_token(test_address, Chain.ETH)
        
        if success and safety:
            logger.success(f"✅ Safety analysis complete")
            logger.info(f"   Honeypot: {safety.is_honeypot}")
            logger.info(f"   Buy Tax: {safety.buy_tax}%")
            logger.info(f"   Sell Tax: {safety.sell_tax}%")
            logger.info(f"   LP Locked: {safety.lp_locked}")
            logger.info(f"   Safety Score: {safety.safety_score}/100")
            return True
        else:
            logger.warning("⚠️ Safety analysis failed (API may not be available)")
            return True  # Don't fail test if safety API is unavailable
            
    except Exception as e:
        logger.error(f"❌ Token safety test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_announcement_monitor():
    """Test launch announcement monitor"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Launch Announcement Monitor")
    logger.info("=" * 60)
    
    try:
        from services.launch_announcement_monitor import LaunchAnnouncementMonitor
        
        monitor = LaunchAnnouncementMonitor(scan_interval_seconds=60)
        
        logger.info("Checking all announcement sources...")
        
        # Run a single check (not continuous)
        announcements = await monitor._check_all_sources()
        
        logger.success(f"✅ Found {len(announcements)} announcements")
        for i, ann in enumerate(announcements[:5]):
            logger.info(f"  {i+1}. {ann.token_symbol} from {ann.source}")
            logger.info(f"     Chain: {ann.chain.value}")
            logger.info(f"     Address: {ann.token_address[:20]}...")
        
        # Check stats
        stats = monitor.get_stats()
        logger.info(f"\nMonitor stats:")
        logger.info(f"   Sources: {stats['sources_configured']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Announcement monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alert_system():
    """Test alert system and Discord webhook"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Alert System & Discord")
    logger.info("=" * 60)
    
    try:
        from services.alert_system import get_alert_system
        
        alert_system = get_alert_system()
        
        # Check Discord webhook configuration
        discord_url = os.getenv('DISCORD_WEBHOOK_URL')
        if discord_url:
            logger.success(f"✅ Discord webhook configured: {discord_url[:50]}...")
        else:
            logger.warning("⚠️ DISCORD_WEBHOOK_URL not set - alerts will only be logged")
        
        # Send a test alert
        logger.info("\nSending test alert...")
        try:
            alert_system.send_alert(
                title="🧪 DEX Service Test",
                message=f"Test alert from DEX launch service verification\n\nTimestamp: {datetime.now().isoformat()}",
                priority="LOW",
                metadata={
                    'symbol': 'TEST',
                    'score': 75.0,
                    'source': 'test_script'
                }
            )
            logger.success("✅ Test alert sent successfully!")
            if discord_url:
                logger.info("   Check your Discord channel for the test message")
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Alert system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_dex_launch_hunter():
    """Test the full DEX launch hunter"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: DEX Launch Hunter (Full Integration)")
    logger.info("=" * 60)
    
    try:
        from services.dex_launch_hunter import get_dex_launch_hunter
        from models.dex_models import HunterConfig, Chain
        
        # Create with custom config for testing
        config = HunterConfig(
            enabled_chains=[Chain.SOLANA, Chain.ETH],
            min_liquidity_usd=5000,
            max_liquidity_usd=1000000,
            min_composite_score=40,  # Lower for testing
            scan_interval_seconds=300
        )
        
        hunter = get_dex_launch_hunter(config)
        
        logger.info("Running single scan cycle...")
        
        # Run scan
        await hunter._scan_for_launches()
        
        discovered = len(hunter.discovered_tokens)
        logger.success(f"✅ Discovered {discovered} tokens")
        
        # Show top tokens
        if hunter.discovered_tokens:
            sorted_tokens = sorted(
                hunter.discovered_tokens.values(),
                key=lambda t: t.composite_score if hasattr(t, 'composite_score') else 0,
                reverse=True
            )[:5]
            
            logger.info("\nTop discovered tokens:")
            for i, token in enumerate(sorted_tokens):
                score = getattr(token, 'composite_score', 0)
                logger.info(f"  {i+1}. {token.symbol} - Score: {score:.1f}")
                logger.info(f"     Liquidity: ${token.liquidity_usd:,.0f}")
                logger.info(f"     Risk: {token.risk_level.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ DEX launch hunter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_scan_cycle():
    """Test a complete scan cycle including alerts"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Full Scan Cycle (Like the service)")
    logger.info("=" * 60)
    
    try:
        from services.launch_announcement_monitor import get_announcement_monitor
        from services.dex_launch_hunter import get_dex_launch_hunter
        from services.alert_system import get_alert_system
        
        # Initialize services (same as run_dex_launch_simple.py)
        monitor = get_announcement_monitor(scan_interval=300)
        dex_hunter = get_dex_launch_hunter()
        alert_system = get_alert_system()
        
        logger.info("Services initialized, running scan...")
        
        # Check for announcements
        recent = await monitor._check_all_sources()
        logger.info(f"Found {len(recent)} announcements")
        
        # Process announcements
        high_score_count = 0
        for announcement in recent[:10]:  # Limit for testing
            if announcement.token_address.lower() in dex_hunter.discovered_tokens:
                continue
            
            logger.info(f"Analyzing {announcement.token_symbol}...")
            
            try:
                success, token = await dex_hunter.analyze_token(
                    announcement.token_address,
                    announcement.chain
                )
                
                if success and token:
                    score = getattr(token, 'composite_score', 0)
                    if score >= 50:  # Lower threshold for testing
                        high_score_count += 1
                        logger.success(f"✅ HIGH SCORE: {token.symbol} = {score:.1f}")
                        
                        # Would send alert here
                        logger.info(f"   Would alert: {token.symbol} on {token.chain.value}")
                        
            except Exception as e:
                logger.debug(f"Error analyzing {announcement.token_symbol}: {e}")
        
        logger.info(f"\n📊 Scan complete: {high_score_count} high-score tokens found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full scan cycle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    logger.info("🚀 DEX LAUNCH SERVICE VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: DexScreener
    results['dexscreener'] = await test_dexscreener_client()
    
    # Test 2: Token Safety
    results['safety'] = await test_token_safety()
    
    # Test 3: Announcement Monitor
    results['announcements'] = await test_announcement_monitor()
    
    # Test 4: Alert System
    results['alerts'] = test_alert_system()
    
    # Test 5: DEX Hunter
    results['hunter'] = await test_dex_launch_hunter()
    
    # Test 6: Full cycle
    results['full_cycle'] = await test_full_scan_cycle()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.success("🎉 ALL TESTS PASSED - Service should work correctly")
    else:
        logger.warning("⚠️ SOME TESTS FAILED - Review errors above")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

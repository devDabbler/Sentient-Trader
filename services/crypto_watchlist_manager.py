"""
Crypto Watchlist Manager - Persistent storage and management for crypto watchlists

Provides Supabase-based storage with full CRUD operations and historical tracking.
Similar to the stock watchlist manager but optimized for cryptocurrencies.
"""

import pandas as pd
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
from loguru import logger
from clients.supabase_client import get_supabase_client



class CryptoWatchlistManager:
    """Manages cryptocurrency watchlists with Supabase persistence"""
    
    def __init__(self):
        """
        Initialize crypto watchlist manager with Supabase
        """
        self.supabase = get_supabase_client()
        if not self.supabase:
            logger.error("CryptoWatchlistManager failed to initialize: Supabase client is not available.")
    
    def _check_client(self) -> bool:
        """Check if Supabase client is available"""
        if not self.supabase:
            logger.warning("Supabase client not initialized. Skipping database operation.")
            return False
        return True
    
    def test_connection(self) -> bool:
        """Test if we can connect to Supabase and access the crypto_watchlist table"""
        if not self._check_client():
            return False
        try:
            response = self.supabase.table('crypto_watchlist').select('symbol').limit(1).execute()
            logger.info("Crypto watchlist Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"Crypto watchlist Supabase connection test failed: {e}")
            return False
    
    def add_crypto(self, symbol: str, opportunity_data: Optional[Dict] = None) -> bool:
        """
        Add a crypto to watchlist
        
        Args:
            symbol: Crypto pair symbol (e.g., 'BTC/USD')
            opportunity_data: Optional CryptoOpportunity data dict
            
        Returns:
            True if added successfully, False otherwise
        """
        logger.info(f"🟢 add_crypto CALLED for symbol: {symbol}")
        logger.info(f"📥 Received opportunity_data: {opportunity_data is not None}")
        
        if not self._check_client():
            logger.error(f"❌ Supabase client check FAILED for {symbol}")
            return False
        
        logger.info(f"✅ Supabase client check PASSED")
        
        try:
            # Parse symbol
            if '/' in symbol:
                base_asset, quote_asset = symbol.split('/')
            else:
                base_asset = symbol
                quote_asset = 'USD'
            
            logger.info(f"📋 Parsed symbol: base={base_asset}, quote={quote_asset}")
            
            # CRITICAL FIX: Strip Kraken suffixes (.F, .S, .M) from symbol before storing
            # These suffixes indicate futures, staking, margin but break ticker API calls
            clean_symbol = symbol.replace('.F/', '/').replace('.S/', '/').replace('.M/', '/')
            if clean_symbol != symbol:
                logger.info(f"🔧 Cleaned symbol: {symbol} → {clean_symbol}")
                symbol = clean_symbol
            
            now = datetime.now(timezone.utc).isoformat()
            
            # Prepare data
            data = {
                'symbol': symbol,
                'base_asset': base_asset,
                'quote_asset': quote_asset,
                'date_added': now,
                'last_updated': now
            }
            
            logger.info(f"🔧 Base data prepared with {len(data)} fields")
            
            # Add opportunity data if provided
            if opportunity_data:
                logger.info(f"📊 Processing opportunity_data with {len(opportunity_data)} fields")
                fields_added = []
                
                # Map opportunity fields to database fields
                if 'current_price' in opportunity_data:
                    data['current_price'] = opportunity_data['current_price']
                    fields_added.append('current_price')
                if 'change_pct_24h' in opportunity_data:
                    data['change_pct_24h'] = opportunity_data['change_pct_24h']
                    fields_added.append('change_pct_24h')
                if 'volume_24h' in opportunity_data:
                    data['volume_24h'] = opportunity_data['volume_24h']
                    fields_added.append('volume_24h')
                if 'volume_ratio' in opportunity_data:
                    data['volume_ratio'] = opportunity_data['volume_ratio']
                    fields_added.append('volume_ratio')
                if 'volatility_24h' in opportunity_data:
                    data['volatility_24h'] = opportunity_data['volatility_24h']
                    fields_added.append('volatility_24h')
                if 'rsi' in opportunity_data:
                    data['rsi'] = opportunity_data['rsi']
                    fields_added.append('rsi')
                if 'momentum_score' in opportunity_data:
                    data['momentum_score'] = opportunity_data['momentum_score']
                    fields_added.append('momentum_score')
                if 'technical_score' in opportunity_data:
                    data['technical_score'] = opportunity_data['technical_score']
                    fields_added.append('technical_score')
                if 'score' in opportunity_data:
                    data['composite_score'] = opportunity_data['score']
                    fields_added.append('composite_score')
                if 'confidence' in opportunity_data:
                    data['confidence_level'] = opportunity_data['confidence']
                    fields_added.append('confidence_level')
                if 'risk_level' in opportunity_data:
                    data['risk_level'] = opportunity_data['risk_level']
                    fields_added.append('risk_level')
                if 'strategy' in opportunity_data:
                    data['strategy'] = opportunity_data['strategy']
                    fields_added.append('strategy')
                if 'reason' in opportunity_data:
                    data['reasoning'] = opportunity_data['reason']
                    fields_added.append('reasoning')
                
                data['last_price_update'] = now
                
                logger.info(f"✅ Added {len(fields_added)} fields from opportunity_data: {fields_added}")
            else:
                logger.warning(f"⚠️ No opportunity_data provided")
            
            logger.info(f"💾 Final data dict has {len(data)} fields: {list(data.keys())}")
            logger.info(f"🔍 Data sample: symbol={data.get('symbol')}, price={data.get('current_price')}, strategy={data.get('strategy')}")
            
            # Upsert to database
            logger.info(f"📤 Executing Supabase upsert for {symbol}...")
            response = self.supabase.table('crypto_watchlist').upsert(
                data,
                on_conflict='symbol'
            ).execute()
            
            logger.info(f"📬 Supabase response received: {response}")
            logger.info(f"✅ Successfully added {symbol} to crypto watchlist")
            return True
                
        except Exception as e:
            logger.error(f"❌ EXCEPTION in add_crypto for {symbol}: {e}", exc_info=True)
            return False
    
    def remove_crypto(self, symbol: str) -> bool:
        """
        Remove a crypto from watchlist
        
        Args:
            symbol: Crypto pair symbol
            
        Returns:
            True if removed successfully, False otherwise
        """
        if not self._check_client():
            return False
        
        try:
            # Delete from watchlist
            self.supabase.table('crypto_watchlist').delete().eq('symbol', symbol).execute()
            
            # Delete associated data
            self.supabase.table('crypto_score_history').delete().eq('symbol', symbol).execute()
            self.supabase.table('crypto_tags').delete().eq('symbol', symbol).execute()
            self.supabase.table('crypto_alerts').delete().eq('symbol', symbol).execute()
            
            logger.info(f"Removed {symbol} from crypto watchlist")
            return True
                
        except Exception as e:
            logger.error(f"Error removing {symbol} from crypto watchlist: {e}")
            return False
    
    def get_all_cryptos(self) -> List[Dict]:
        """
        Get all cryptos in watchlist
        
        Returns:
            List of dicts of all watchlist cryptos
        """
        if not self._check_client():
            return []
        
        try:
            response = self.supabase.table('crypto_watchlist')\
                .select('*')\
                .order('composite_score', desc=True)\
                .execute()
            
            if response.data:
                return response.data
            return []
                
        except Exception as e:
            logger.error(f"Error getting crypto watchlist: {e}")
            return []
    
    def get_crypto(self, symbol: str) -> Optional[Dict]:
        """
        Get a specific crypto from watchlist
        
        Args:
            symbol: Crypto pair symbol
            
        Returns:
            Dict of crypto data or None
        """
        if not self._check_client():
            return None
        
        try:
            response = self.supabase.table('crypto_watchlist')\
                .select('*')\
                .eq('symbol', symbol)\
                .single()\
                .execute()
            
            return response.data if response.data else None
                
        except Exception as e:
            logger.error(f"Error getting {symbol}: {e}")
            return None
    
    def update_crypto_price(self, symbol: str, price_data: Dict) -> bool:
        """
        Update crypto price and technical data
        
        Args:
            symbol: Crypto pair symbol
            price_data: Dict with price, indicators, etc.
            
        Returns:
            True if updated successfully
        """
        if not self._check_client():
            return False
        
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            # Build update data
            update_data = {
                'last_price_update': now,
                'last_updated': now
            }
            
            # Map fields
            field_mappings = {
                'current_price': 'current_price',
                'change_pct_24h': 'change_pct_24h',
                'volume_24h': 'volume_24h',
                'volume_ratio': 'volume_ratio',
                'volatility_24h': 'volatility_24h',
                'rsi': 'rsi',
                'ema_8': 'ema_8',
                'ema_20': 'ema_20',
                'vwap': 'vwap',
                'bb_upper': 'bb_upper',
                'bb_middle': 'bb_middle',
                'bb_lower': 'bb_lower',
                'macd_line': 'macd_line',
                'macd_signal': 'macd_signal',
                'macd_histogram': 'macd_histogram',
                'momentum_score': 'momentum_score',
                'technical_score': 'technical_score',
                'composite_score': 'composite_score',
                'confidence_level': 'confidence_level',
                'risk_level': 'risk_level'
            }
            
            for key, db_field in field_mappings.items():
                if key in price_data:
                    update_data[db_field] = price_data[key]
            
            # Update
            response = self.supabase.table('crypto_watchlist')\
                .update(update_data)\
                .eq('symbol', symbol)\
                .execute()
            
            # Record to history
            if 'current_price' in price_data or 'composite_score' in price_data:
                self._record_score_history(symbol, price_data)
            
            return True
                
        except Exception as e:
            logger.error(f"Error updating {symbol} price: {e}")
            return False
    
    def _record_score_history(self, symbol: str, data: Dict):
        """Record scores to history table"""
        if not self._check_client():
            return
        
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            history_data = {
                'symbol': symbol,
                'date': now,
                'price': data.get('current_price'),
                'momentum_score': data.get('momentum_score'),
                'technical_score': data.get('technical_score'),
                'composite_score': data.get('composite_score'),
                'confidence_level': data.get('confidence_level')
            }
            
            # Remove None values
            history_data = {k: v for k, v in history_data.items() if v is not None}
            
            self.supabase.table('crypto_score_history').insert(history_data).execute()
                
        except Exception as e:
            logger.error(f"Error recording score history for {symbol}: {e}")
    
    def add_tag(self, symbol: str, tag: str) -> bool:
        """Add a tag to a crypto"""
        if not self._check_client():
            return False
        
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            data = {
                'symbol': symbol,
                'tag': tag,
                'date_added': now
            }
            
            self.supabase.table('crypto_tags').insert(data).execute()
            return True
                
        except Exception as e:
            logger.error(f"Error adding tag to {symbol}: {e}")
            return False
    
    def remove_tag(self, symbol: str, tag: str) -> bool:
        """Remove a tag from a crypto"""
        if not self._check_client():
            return False
        
        try:
            self.supabase.table('crypto_tags')\
                .delete()\
                .eq('symbol', symbol)\
                .eq('tag', tag)\
                .execute()
            return True
                
        except Exception as e:
            logger.error(f"Error removing tag from {symbol}: {e}")
            return False
    
    def get_tags(self, symbol: str) -> List[str]:
        """Get all tags for a crypto"""
        if not self._check_client():
            return []
        
        try:
            response = self.supabase.table('crypto_tags')\
                .select('tag')\
                .eq('symbol', symbol)\
                .execute()
            
            return [row['tag'] for row in response.data] if response.data else []
                
        except Exception as e:
            logger.error(f"Error getting tags for {symbol}: {e}")
            return []
    
    def update_notes(self, symbol: str, notes: str) -> bool:
        """Update notes for a crypto"""
        if not self._check_client():
            return False
        
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            data = {
                'notes': notes,
                'last_updated': now
            }
            
            self.supabase.table('crypto_watchlist')\
                .update(data)\
                .eq('symbol', symbol)\
                .execute()
            return True
                
        except Exception as e:
            logger.error(f"Error updating notes for {symbol}: {e}")
            return False
    
    def get_watchlist(self) -> List[Dict]:
        """
        Get full watchlist with all data (alias for get_all_cryptos for compatibility)
        
        Returns:
            List of dicts with all watchlist crypto data
        """
        return self.get_all_cryptos()
    
    def get_watchlist_symbols(self) -> List[str]:
        """Get list of all watchlist symbols"""
        if not self._check_client():
            return []
        
        try:
            response = self.supabase.table('crypto_watchlist')\
                .select('symbol')\
                .order('symbol')\
                .execute()
            
            return [row['symbol'] for row in response.data] if response.data else []
                
        except Exception as e:
            logger.error(f"Error getting watchlist symbols: {e}")
            return []
    
    def is_in_watchlist(self, symbol: str) -> bool:
        """Check if a crypto is in the watchlist"""
        if not self._check_client():
            return False
        
        try:
            response = self.supabase.table('crypto_watchlist')\
                .select('symbol', count='exact')\
                .eq('symbol', symbol)\
                .execute()
            
            return response.count > 0 if response.count is not None else False
                
        except Exception as e:
            logger.error(f"Error checking if {symbol} is in watchlist: {e}")
            return False
    
    def get_count(self) -> int:
        """Get total count of cryptos in watchlist"""
        if not self._check_client():
            return 0
        
        try:
            response = self.supabase.table('crypto_watchlist')\
                .select('*', count='exact')\
                .execute()
            
            return response.count if response.count is not None else 0
                
        except Exception as e:
            logger.error(f"Error getting watchlist count: {e}")
            return 0
    
    def clear_watchlist(self) -> bool:
        """Clear all cryptos from watchlist"""
        if not self._check_client():
            return False
        
        try:
            # Clear all tables (note: this might need to be done in order due to foreign keys)
            self.supabase.table('crypto_alerts').delete().neq('id', -1).execute()
            self.supabase.table('crypto_tags').delete().neq('id', -1).execute()
            self.supabase.table('crypto_score_history').delete().neq('id', -1).execute()
            self.supabase.table('crypto_watchlist').delete().neq('id', -1).execute()
            
            logger.info("Cleared crypto watchlist")
            return True
                
        except Exception as e:
            logger.error(f"Error clearing crypto watchlist: {e}")
            return False
    
    def get_score_history(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get score history for a crypto"""
        if not self._check_client():
            return pd.DataFrame()
        
        try:
            response = self.supabase.table('crypto_score_history')\
                .select('*')\
                .eq('symbol', symbol)\
                .order('date', desc=True)\
                .limit(limit)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting score history for {symbol}: {e}")
            return pd.DataFrame()
    
    def export_to_json(self, filepath: str) -> bool:
        """Export watchlist to JSON file"""
        try:
            data = self.get_all_cryptos()
            
            if not data:
                logger.warning("No cryptos in watchlist to export")
                return False
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(data)} cryptos to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting watchlist: {e}")
            return False
    
    def import_from_json(self, filepath: str) -> bool:
        """Import watchlist from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            count = 0
            for crypto in data:
                symbol = crypto.get('symbol')
                if symbol:
                    if self.add_crypto(symbol, crypto):
                        count += 1
            
            logger.info(f"Imported {count} cryptos from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing watchlist: {e}")
            return False
    
    def create_alert(self, symbol: str, alert_type: str, condition: str, 
                    target_value: float) -> bool:
        """Create a price alert for a crypto"""
        if not self._check_client():
            return False
        
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            data = {
                'symbol': symbol,
                'alert_type': alert_type,
                'condition': condition,
                'target_value': target_value,
                'triggered': False,
                'date_created': now,
                'enabled': True
            }
            
            self.supabase.table('crypto_alerts').insert(data).execute()
            logger.info(f"Created alert for {symbol}: {condition} {target_value}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating alert for {symbol}: {e}")
            return False
    
    def get_active_alerts(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get active alerts, optionally filtered by symbol"""
        if not self._check_client():
            return []
        
        try:
            query = self.supabase.table('crypto_alerts')\
                .select('*')\
                .eq('enabled', True)\
                .eq('triggered', False)
            
            if symbol:
                query = query.eq('symbol', symbol)
            
            response = query.execute()
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []

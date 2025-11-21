"""
Ticker Management System

Provides persistent storage and management for saved tickers,
watchlists, and quick-access lists using a Supabase backend.
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from loguru import logger
from postgrest import CountMethod
from clients.supabase_client import get_supabase_client


class TickerManager:
    """Manages saved tickers and watchlists with a Supabase backend"""
    
    def __init__(self):
        """Initialize ticker manager"""
        self.supabase = get_supabase_client()
        if not self.supabase:
            logger.error("TickerManager failed to initialize: Supabase client is not available.")

    def _check_client(self) -> bool:
        if not self.supabase:
            logger.warning("Supabase client not initialized. Skipping database operation.")
            return False
        return True
    
    def test_connection(self) -> bool:
        """Test if we can connect to Supabase and access the saved_tickers table"""
        if not self._check_client():
            return False
        try:
            # Try to query the table to see if it exists and is accessible
            response = self.supabase.table('saved_tickers').select('ticker').limit(1).execute()
            logger.info(f"Supabase connection test successful. Response: {response}")
            return True
        except Exception as e:
            logger.error(f"Supabase connection test failed: {e}")
            logger.error("Error type: {}", str(type(e).__name__))
            return False

    def add_ticker(self, ticker: Optional[str], name: Optional[str] = None, sector: Optional[str] = None, 
                   ticker_type: str = 'stock', notes: Optional[str] = None, tags: Optional[List[str]] = None,
                   auto_trade_enabled: bool = False, auto_trade_strategy: Optional[str] = None) -> bool:
        if not self._check_client(): 
            logger.error("Supabase client not available")
            return False

        if not ticker or not isinstance(ticker, str):
            logger.error(f"Invalid ticker provided: {ticker}. Ticker must be a non-empty string.")
            return False
        try:
            ticker = ticker.upper()
            current_time = datetime.now(timezone.utc).isoformat()
            
            data_to_upsert = {
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'type': ticker_type,
                'notes': notes,
                'tags': tags,
                'last_accessed': current_time,
                'access_count': 0,  # Initialize access_count
                'auto_trade_enabled': auto_trade_enabled,
                'auto_trade_strategy': auto_trade_strategy,
            }
            
            data_to_upsert = {k: v for k, v in data_to_upsert.items() if v is not None}
            logger.info(f"Attempting to upsert ticker data: {data_to_upsert}")

            response = self.supabase.table('saved_tickers').upsert(
                data_to_upsert, 
                on_conflict='ticker'
            ).execute()

            logger.info(f"Successfully added/updated ticker: {ticker}")
            logger.info(f"Supabase response: {response}")
            return True
        except Exception as e:
            logger.error(f"Error adding/updating ticker {ticker}: {e}")
            logger.error("Error type: {}", str(type(e).__name__))
            logger.error(f"Error details: {e}")
            return False

    def remove_ticker(self, ticker: str) -> bool:
        if not self._check_client(): return False
        try:
            self.supabase.table('saved_tickers').delete().eq('ticker', ticker.upper()).execute()
            logger.info(f"Removed ticker: {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error removing ticker {ticker}: {e}")
            return False

    def get_ticker(self, ticker: str) -> Optional[Dict]:
        if not self._check_client(): return None
        try:
            response = self.supabase.table('saved_tickers').select('*').eq('ticker', ticker.upper()).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting ticker {ticker}: {e}")
            return None

    def get_all_tickers(self, ticker_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        if not self._check_client(): return []
        try:
            query = self.supabase.table('saved_tickers').select('*')
            if ticker_type:
                query = query.eq('type', ticker_type)
            
            # Try to order by access_count, fallback to ticker if column doesn't exist
            try:
                response = query.order('access_count', desc=True).order('ticker').limit(limit).execute()
            except Exception as order_error:
                logger.warning(f"Could not order by access_count, using ticker order: {order_error}")
                response = query.order('ticker').limit(limit).execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error getting all tickers: {e}")
            return []

    def get_recent_tickers(self, limit: int = 10) -> List[str]:
        if not self._check_client(): return []
        try:
            response = self.supabase.table('saved_tickers').select('ticker').order('last_accessed', desc=True).limit(limit).execute()
            return [item['ticker'] for item in response.data]
        except Exception as e:
            logger.error(f"Error getting recent tickers: {e}")
            return []

    def get_popular_tickers(self, limit: int = 10) -> List[str]:
        if not self._check_client(): return []
        try:
            # Try to filter by access_count, fallback to all tickers if column doesn't exist
            try:
                response = self.supabase.table('saved_tickers').select('ticker').gt('access_count', 0).order('access_count', desc=True).limit(limit).execute()
            except Exception as filter_error:
                logger.warning(f"Could not filter by access_count, returning recent tickers: {filter_error}")
                response = self.supabase.table('saved_tickers').select('ticker').order('last_accessed', desc=True).limit(limit).execute()
            
            return [item['ticker'] for item in response.data]
        except Exception as e:
            logger.error(f"Error getting popular tickers: {e}")
            return []

    def record_access(self, ticker: str):
        if not self._check_client(): return
        try:
            # Try RPC function first, fallback to direct update if RPC doesn't exist
            try:
                self.supabase.rpc('increment_access_count', {'ticker_symbol': ticker.upper()}).execute()
            except Exception as rpc_error:
                logger.warning(f"RPC function not available, using direct update: {rpc_error}")
                # Fallback: direct update
                current_time = datetime.now(timezone.utc).isoformat()
                self.supabase.table('saved_tickers').update({
                    'access_count': 1,
                    'last_accessed': current_time
                }).eq('ticker', ticker.upper()).execute()
        except Exception as e:
            logger.error(f"Error recording access for {ticker}: {e}")
    
    def set_auto_trade(self, ticker: str, enabled: bool, strategy: Optional[str] = None) -> bool:
        """Enable/disable auto-trading for a specific ticker"""
        if not self._check_client(): return False
        try:
            update_data: Dict[str, Any] = {'auto_trade_enabled': enabled}
            if strategy:
                update_data['auto_trade_strategy'] = strategy
            
            self.supabase.table('saved_tickers').update(update_data).eq('ticker', ticker.upper()).execute()
            logger.info(f"Updated auto-trade for {ticker}: enabled={enabled}, strategy={strategy}")
            return True
        except Exception as e:
            # Check if error is due to missing columns
            error_str = str(e)
            if 'auto_trade_enabled' in error_str or 'auto_trade_strategy' in error_str:
                logger.warning(f"Auto-trade columns not found in database. Please run the migration script: migrations/add_auto_trade_columns.sql")
            else:
                logger.error(f"Error updating auto-trade for {ticker}: {e}")
            return False
    
    def get_auto_trade_tickers(self) -> List[Dict]:
        """Get all tickers with auto-trading enabled"""
        if not self._check_client(): return []
        try:
            response = self.supabase.table('saved_tickers').select('*').eq('auto_trade_enabled', True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting auto-trade tickers: {e}")
            return []

    def search_tickers(self, query: str) -> List[Dict]:
        if not self._check_client(): return []
        try:
            search_pattern = f"%{query.upper()}%"
            response = self.supabase.table('saved_tickers').select('*').or_(
                f"ticker.ilike.{search_pattern}",
                f"name.ilike.{search_pattern}"
            ).execute()
            return response.data
        except Exception as e:
            logger.error("Error searching tickers for '{query}': {}", str(e))
            return []

    def create_watchlist(self, name: str, description: Optional[str] = None) -> bool:
        if not self._check_client(): return False
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            self.supabase.table('watchlists').insert({
                'name': name,
                'description': description,
                'date_created': current_time,
                'last_updated': current_time
            }).execute()
            logger.info(f"Created watchlist: {name}")
            return True
        except Exception as e:
            logger.error(f"Error creating watchlist {name}: {e}")
            return False

    def get_watchlists(self) -> List[Dict]:
        if not self._check_client(): return []
        try:
            response = self.supabase.table('watchlists').select('*').order('name').execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting watchlists: {e}")
            return []

    def add_to_watchlist(self, watchlist_name: str, ticker: str) -> bool:
        if not self._check_client(): return False
        try:
            w_res = self.supabase.table('watchlists').select('id').eq('name', watchlist_name).single().execute()
            if not w_res.data: return False
            watchlist_id = w_res.data['id']

            self.supabase.table('watchlist_items').insert({
                'watchlist_id': watchlist_id,
                'ticker': ticker.upper(),
                'date_added': datetime.now(timezone.utc).isoformat()
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Error adding {ticker} to watchlist {watchlist_name}: {e}")
            return False

    def get_watchlist_tickers(self, watchlist_name: str) -> List[str]:
        if not self._check_client(): return []
        try:
            response = self.supabase.rpc('get_watchlist_tickers_by_name', {'watchlist_name_param': watchlist_name}).execute()
            if response.data and isinstance(response.data, list):
                return [str(item.get('ticker', '')) for item in response.data if isinstance(item, dict) and 'ticker' in item]
            return []
        except Exception as e:
            logger.error(f"Error getting tickers for watchlist {watchlist_name}: {e}")
            return []

    def remove_from_watchlist(self, watchlist_name: str, ticker: str) -> bool:
        if not self._check_client(): return False
        try:
            w_res = self.supabase.table('watchlists').select('id').eq('name', watchlist_name).single().execute()
            if not w_res.data: return False
            watchlist_id = w_res.data['id']

            self.supabase.table('watchlist_items').delete().match({'watchlist_id': watchlist_id, 'ticker': ticker.upper()}).execute()
            return True
        except Exception as e:
            logger.error(f"Error removing {ticker} from watchlist {watchlist_name}: {e}")
            return False

    def delete_watchlist(self, watchlist_name: str) -> bool:
        if not self._check_client(): return False
        try:
            self.supabase.table('watchlists').delete().eq('name', watchlist_name).execute()
            logger.info(f"Deleted watchlist: {watchlist_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting watchlist {watchlist_name}: {e}")
            return False

    def update_analysis(self, ticker: str, analysis: Dict) -> bool:
        """
        Update analysis data for a ticker using only core columns that exist in schema.
        Database columns: ticker, name, sector, type, notes, tags, last_accessed, access_count
        Analysis fields: ml_score, momentum, volume_ratio, rsi, sentiment_score, last_analyzed
        """
        if not self._check_client(): return False
        try:
            ticker = ticker.upper()
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Use upsert to handle both new and existing tickers
            # Only use core columns that exist in the database
            update_data = {
                'ticker': ticker,
                'last_accessed': current_time,
            }
            
            # Add only the analysis fields that have corresponding DB columns
            # These were likely added to your Supabase schema
            safe_field_mapping = {
                'confidence_score': ('ml_score', float),
                'change_pct': ('momentum', float),
                'volume': ('volume_ratio', int),
                'rsi': ('rsi', float),
                'sentiment_score': ('sentiment_score', float),
            }
            
            for analysis_key, (db_field, type_cast) in safe_field_mapping.items():
                try:
                    if analysis_key in analysis and analysis[analysis_key] is not None:
                        update_data[db_field] = type_cast(analysis[analysis_key])
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping {db_field} due to type conversion error: {e}")
            
            # Try to add last_analyzed if column exists
            try:
                update_data['last_analyzed'] = current_time
            except:
                pass

            self.supabase.table('saved_tickers').upsert(
                update_data,
                on_conflict='ticker'
            ).execute()
            
            logger.info("✓ Updated analysis for {}: score={update_data.get('ml_score', 'N/A')}, momentum={update_data.get('momentum', 'N/A')}%", str(ticker))
            return True
                
        except Exception as e:
            logger.error(f"Error updating analysis for {ticker}: {e}")
            logger.debug("Attempted data: {}", str(update_data if 'update_data' in locals() else 'N/A'))
            return False

    def update_ai_entry_analysis(self, ticker: str, entry_analysis: Dict) -> bool:
        """
        Update AI entry analysis data for a ticker.

        Args:
            ticker: The stock ticker symbol.
            entry_analysis: A dictionary containing the AI entry analysis results.
                            Expected keys: 'confidence', 'action', 'reasons', 'targets'.

        Returns:
            True if the update was successful, False otherwise.
        """
        if not self._check_client():
            return False

        try:
            ticker = ticker.upper()
            current_time = datetime.now(timezone.utc).isoformat()

            update_data = {
                'ticker': ticker,
                'ai_entry_timestamp': current_time,
            }

            # Map analysis keys to database columns and their types
            field_mapping = {
                'confidence': ('ai_entry_confidence', float),
                'action': ('ai_entry_action', str),
                'reasons': ('ai_entry_reasons', list),
                'targets': ('ai_entry_targets', dict),
            }

            for key, (db_field, type_cast) in field_mapping.items():
                if key in entry_analysis and entry_analysis[key] is not None:
                    try:
                        value = entry_analysis[key]
                        if isinstance(value, (list, dict)):
                            update_data[db_field] = json.dumps(value)
                        else:
                            update_data[db_field] = type_cast(value)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Skipping {db_field} for {ticker} due to type conversion error: {e}")

            self.supabase.table('saved_tickers').upsert(
                update_data,
                on_conflict='ticker'
            ).execute()

            logger.info(f"Successfully updated AI entry analysis for {ticker}.")
            return True

        except Exception as e:
            error_str = str(e)
            if "column" in error_str and "does not exist" in error_str:
                 logger.warning(
                    f"AI entry columns not found in 'saved_tickers' table. "
                    f"Please run the migration: migrations/add_ai_entry_columns.sql"
                )
            else:
                logger.error(f"Error updating AI entry analysis for {ticker}: {e}")
            
            logger.debug("Attempted data for {}: {update_data if 'update_data' in locals() else 'N/A'}", str(ticker))
            return False

    def should_update_analysis(self, ticker: str, max_age_hours: float = 1.0) -> bool:
        """
        Check if ticker analysis needs updating based on staleness.
        
        Args:
            ticker: Ticker symbol
            max_age_hours: Maximum age in hours before considering analysis stale
            
        Returns:
            True if analysis should be updated, False otherwise
        """
        if not self._check_client(): 
            return True  # Update if we can't check
        try:
            response = self.supabase.table('saved_tickers').select('last_analyzed').eq('ticker', ticker.upper()).single().execute()
            if response.data and response.data.get('last_analyzed'):
                last_analyzed = datetime.fromisoformat(response.data['last_analyzed']).replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - last_analyzed).total_seconds() / 3600
                logger.debug(f"Ticker {ticker} analysis age: {age_hours:.2f} hours")
                return age_hours >= max_age_hours
            # No last_analyzed timestamp, needs update
            return True
        except Exception as e:
            logger.debug(f"Error checking analysis staleness for {ticker}: {e}")
            return True  # Update if there's an error
    def get_statistics(self) -> Dict:
        if not self._check_client(): return {}
        try:
            # This can be optimized with a single RPC call in the future
            total_tickers = self.supabase.table('saved_tickers').select('id', count=CountMethod.exact).execute().count
            stock_count = self.supabase.table('saved_tickers').select('id', count=CountMethod.exact).eq('type', 'stock').execute().count
            penny_count = self.supabase.table('saved_tickers').select('id', count=CountMethod.exact).eq('type', 'penny_stock').execute().count
            watchlist_count = self.supabase.table('watchlists').select('id', count=CountMethod.exact).execute().count
            
            return {
                'total_tickers': total_tickers,
                'stocks': stock_count,
                'penny_stocks': penny_count,
                'watchlists': watchlist_count
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
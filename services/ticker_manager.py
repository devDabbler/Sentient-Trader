"""
Ticker Management System

Provides persistent storage and management for saved tickers,
watchlists, and quick-access lists using a Supabase backend.
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging
from clients.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

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

    def add_ticker(self, ticker: str, name: str = None, sector: str = None, 
                   ticker_type: str = 'stock', notes: str = None, tags: List[str] = None) -> bool:
        if not self._check_client(): return False
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
            }
            
            data_to_upsert = {k: v for k, v in data_to_upsert.items() if v is not None}

            self.supabase.table('saved_tickers').upsert(
                data_to_upsert, 
                on_conflict='ticker'
            ).execute()

            logger.info(f"Added/updated ticker: {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error adding/updating ticker {ticker}: {e}")
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

    def get_all_tickers(self, ticker_type: str = None, limit: int = 100) -> List[Dict]:
        if not self._check_client(): return []
        try:
            query = self.supabase.table('saved_tickers').select('*')
            if ticker_type:
                query = query.eq('type', ticker_type)
            
            response = query.order('access_count', desc=True).order('ticker').limit(limit).execute()
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
            response = self.supabase.table('saved_tickers').select('ticker').gt('access_count', 0).order('access_count', desc=True).limit(limit).execute()
            return [item['ticker'] for item in response.data]
        except Exception as e:
            logger.error(f"Error getting popular tickers: {e}")
            return []

    def record_access(self, ticker: str):
        if not self._check_client(): return
        try:
            self.supabase.rpc('increment_access_count', {'ticker_symbol': ticker.upper()}).execute()
        except Exception as e:
            logger.error(f"Error recording access for {ticker}: {e}")

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
            logger.error(f"Error searching tickers for '{query}': {e}")
            return []

    def create_watchlist(self, name: str, description: str = None) -> bool:
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
            return [item['ticker'] for item in response.data]
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

    def update_ml_analysis(self, ticker: str, ml_score: float = None, 
                          momentum: float = None, volume_ratio: float = None, 
                          rsi: float = None) -> bool:
        if not self._check_client(): return False
        try:
            update_data = {
                'ml_score': ml_score,
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'last_analyzed': datetime.now(timezone.utc).isoformat()
            }
            update_data = {k: v for k, v in update_data.items() if v is not None}

            self.supabase.table('saved_tickers').update(update_data).eq('ticker', ticker.upper()).execute()
            logger.info(f"Updated ML analysis for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error updating ML analysis for {ticker}: {e}")
            return False

    def get_statistics(self) -> Dict:
        if not self._check_client(): return {}
        try:
            # This can be optimized with a single RPC call in the future
            total_tickers = self.supabase.table('saved_tickers').select('id', count='exact').execute().count
            stock_count = self.supabase.table('saved_tickers').select('id', count='exact').eq('type', 'stock').execute().count
            penny_count = self.supabase.table('saved_tickers').select('id', count='exact').eq('type', 'penny_stock').execute().count
            watchlist_count = self.supabase.table('watchlists').select('id', count='exact').execute().count
            
            return {
                'total_tickers': total_tickers,
                'stocks': stock_count,
                'penny_stocks': penny_count,
                'watchlists': watchlist_count
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
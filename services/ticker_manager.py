"""
Ticker Management System

Provides persistent storage and management for saved tickers,
watchlists, and quick-access lists.
"""

import sqlite3
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TickerManager:
    """Manages saved tickers and watchlists with SQLite persistence"""
    
    def __init__(self, db_path: str = "data/tickers.db"):
        """
        Initialize ticker manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Saved tickers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS saved_tickers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    name TEXT,
                    sector TEXT,
                    type TEXT DEFAULT 'stock',
                    notes TEXT,
                    tags TEXT,
                    date_added TEXT,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    ml_score REAL,
                    momentum REAL,
                    volume_ratio REAL,
                    rsi REAL,
                    last_analyzed TEXT
                )
            """)
            
            # Add ML analysis columns to existing table if they don't exist
            try:
                cursor.execute("ALTER TABLE saved_tickers ADD COLUMN ml_score REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE saved_tickers ADD COLUMN momentum REAL")
            except sqlite3.OperationalError:
                pass
            
            try:
                cursor.execute("ALTER TABLE saved_tickers ADD COLUMN volume_ratio REAL")
            except sqlite3.OperationalError:
                pass
            
            try:
                cursor.execute("ALTER TABLE saved_tickers ADD COLUMN rsi REAL")
            except sqlite3.OperationalError:
                pass
            
            try:
                cursor.execute("ALTER TABLE saved_tickers ADD COLUMN last_analyzed TEXT")
            except sqlite3.OperationalError:
                pass
            
            # Watchlists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    date_created TEXT,
                    last_updated TEXT
                )
            """)
            
            # Watchlist items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    date_added TEXT,
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists(id),
                    UNIQUE(watchlist_id, ticker)
                )
            """)
            
            conn.commit()
            logger.info(f"Ticker database initialized at {self.db_path}")
    
    def add_ticker(self, ticker: str, name: str = None, sector: str = None, 
                   ticker_type: str = 'stock', notes: str = None, tags: List[str] = None) -> bool:
        """
        Add or update a ticker
        
        Args:
            ticker: Ticker symbol
            name: Company/asset name
            sector: Sector/category
            ticker_type: Type (stock, option, penny_stock, crypto, etc.)
            notes: User notes
            tags: List of tags
        
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                ticker = ticker.upper()
                current_time = datetime.now(timezone.utc).isoformat()
                tags_json = json.dumps(tags) if tags else None
                
                # Check if exists
                cursor.execute("SELECT id FROM saved_tickers WHERE ticker = ?", (ticker,))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing - preserve original date_added, only update last_accessed
                    cursor.execute("""
                        UPDATE saved_tickers 
                        SET name = ?, sector = ?, type = ?, notes = ?, tags = ?, last_accessed = ?
                        WHERE ticker = ?
                    """, (name, sector, ticker_type, notes, tags_json, current_time, ticker))
                else:
                    # Insert new
                    cursor.execute("""
                        INSERT INTO saved_tickers 
                        (ticker, name, sector, type, notes, tags, date_added, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (ticker, name, sector, ticker_type, notes, tags_json, current_time, current_time))
                
                conn.commit()
                logger.info(f"Added/updated ticker: {ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding ticker: {e}")
            return False
    
    def remove_ticker(self, ticker: str) -> bool:
        """Remove a ticker"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM saved_tickers WHERE ticker = ?", (ticker.upper(),))
                conn.commit()
                logger.info(f"Removed ticker: {ticker}")
                return True
        except Exception as e:
            logger.error(f"Error removing ticker: {e}")
            return False
    
    def get_ticker(self, ticker: str) -> Optional[Dict]:
        """Get ticker details"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM saved_tickers WHERE ticker = ?", (ticker.upper(),))
                row = cursor.fetchone()
                
                if row:
                    result = dict(row)
                    # Parse tags JSON
                    if result.get('tags'):
                        result['tags'] = json.loads(result['tags'])
                    return result
                return None
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return None
    
    def get_all_tickers(self, ticker_type: str = None, limit: int = 100) -> List[Dict]:
        """
        Get all tickers, optionally filtered by type
        
        Args:
            ticker_type: Filter by type (stock, penny_stock, option, etc.)
            limit: Maximum number to return
        
        Returns:
            List of ticker dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if ticker_type:
                    query = "SELECT * FROM saved_tickers WHERE type = ? ORDER BY access_count DESC, ticker ASC LIMIT ?"
                    cursor.execute(query, (ticker_type, limit))
                else:
                    query = "SELECT * FROM saved_tickers ORDER BY access_count DESC, ticker ASC LIMIT ?"
                    cursor.execute(query, (limit,))
                
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    result = dict(row)
                    if result.get('tags'):
                        try:
                            result['tags'] = json.loads(result['tags'])
                        except:
                            result['tags'] = []
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Error getting all tickers: {e}")
            return []
    
    def get_recent_tickers(self, limit: int = 10) -> List[str]:
        """Get recently accessed tickers"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ticker FROM saved_tickers 
                    ORDER BY last_accessed DESC 
                    LIMIT ?
                """, (limit,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting recent tickers: {e}")
            return []
    
    def get_popular_tickers(self, limit: int = 10) -> List[str]:
        """Get most frequently accessed tickers"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ticker FROM saved_tickers 
                    WHERE access_count > 0
                    ORDER BY access_count DESC 
                    LIMIT ?
                """, (limit,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting popular tickers: {e}")
            return []
    
    def record_access(self, ticker: str):
        """Record that a ticker was accessed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_time = datetime.now(timezone.utc).isoformat()
                cursor.execute("""
                    UPDATE saved_tickers 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE ticker = ?
                """, (current_time, ticker.upper()))
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording access: {e}")
    
    def search_tickers(self, query: str) -> List[Dict]:
        """Search tickers by symbol, name, or tags"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                search_pattern = f"%{query.upper()}%"
                cursor.execute("""
                    SELECT * FROM saved_tickers 
                    WHERE ticker LIKE ? OR name LIKE ? OR tags LIKE ?
                    ORDER BY ticker ASC
                """, (search_pattern, search_pattern, search_pattern))
                
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    result = dict(row)
                    if result.get('tags'):
                        try:
                            result['tags'] = json.loads(result['tags'])
                        except:
                            result['tags'] = []
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Error searching tickers: {e}")
            return []
    
    # Watchlist methods
    
    def create_watchlist(self, name: str, description: str = None) -> bool:
        """Create a new watchlist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_time = datetime.now(timezone.utc).isoformat()
                cursor.execute("""
                    INSERT INTO watchlists (name, description, date_created, last_updated)
                    VALUES (?, ?, ?, ?)
                """, (name, description, current_time, current_time))
                conn.commit()
                logger.info(f"Created watchlist: {name}")
                return True
        except Exception as e:
            logger.error(f"Error creating watchlist: {e}")
            return False
    
    def get_watchlists(self) -> List[Dict]:
        """Get all watchlists"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM watchlists ORDER BY name ASC")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting watchlists: {e}")
            return []
    
    def add_to_watchlist(self, watchlist_name: str, ticker: str) -> bool:
        """Add ticker to watchlist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get watchlist ID
                cursor.execute("SELECT id FROM watchlists WHERE name = ?", (watchlist_name,))
                result = cursor.fetchone()
                
                if not result:
                    return False
                
                watchlist_id = result[0]
                current_time = datetime.now(timezone.utc).isoformat()
                
                # Add ticker to watchlist
                cursor.execute("""
                    INSERT OR IGNORE INTO watchlist_items (watchlist_id, ticker, date_added)
                    VALUES (?, ?, ?)
                """, (watchlist_id, ticker.upper(), current_time))
                
                # Update watchlist timestamp
                cursor.execute("""
                    UPDATE watchlists SET last_updated = ? WHERE id = ?
                """, (current_time, watchlist_id))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
            return False
    
    def get_watchlist_tickers(self, watchlist_name: str) -> List[str]:
        """Get all tickers in a watchlist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT wi.ticker FROM watchlist_items wi
                    JOIN watchlists w ON wi.watchlist_id = w.id
                    WHERE w.name = ?
                    ORDER BY wi.date_added DESC
                """, (watchlist_name,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting watchlist tickers: {e}")
            return []
    
    def remove_from_watchlist(self, watchlist_name: str, ticker: str) -> bool:
        """Remove ticker from watchlist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM watchlist_items 
                    WHERE watchlist_id = (SELECT id FROM watchlists WHERE name = ?)
                    AND ticker = ?
                """, (watchlist_name, ticker.upper()))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error removing from watchlist: {e}")
            return False
    
    def delete_watchlist(self, watchlist_name: str) -> bool:
        """Delete a watchlist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Delete items first
                cursor.execute("""
                    DELETE FROM watchlist_items 
                    WHERE watchlist_id = (SELECT id FROM watchlists WHERE name = ?)
                """, (watchlist_name,))
                # Delete watchlist
                cursor.execute("DELETE FROM watchlists WHERE name = ?", (watchlist_name,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting watchlist: {e}")
            return False
    
    def update_ml_analysis(self, ticker: str, ml_score: float = None, 
                          momentum: float = None, volume_ratio: float = None, 
                          rsi: float = None) -> bool:
        """Update ML analysis results for a ticker"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_time = datetime.now(timezone.utc).isoformat()
                
                cursor.execute("""
                    UPDATE saved_tickers 
                    SET ml_score = ?, momentum = ?, volume_ratio = ?, rsi = ?, last_analyzed = ?
                    WHERE ticker = ?
                """, (ml_score, momentum, volume_ratio, rsi, current_time, ticker.upper()))
                
                conn.commit()
                logger.info(f"Updated ML analysis for {ticker}")
                return True
        except Exception as e:
            logger.error(f"Error updating ML analysis: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get ticker database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM saved_tickers")
                total_tickers = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM saved_tickers WHERE type = 'stock'")
                stock_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM saved_tickers WHERE type = 'penny_stock'")
                penny_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM watchlists")
                watchlist_count = cursor.fetchone()[0]
                
                return {
                    'total_tickers': total_tickers,
                    'stocks': stock_count,
                    'penny_stocks': penny_count,
                    'watchlists': watchlist_count
                }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

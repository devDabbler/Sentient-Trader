"""
Watchlist Manager - Persistent storage and management for penny stock watchlists

Provides SQLite-based storage with full CRUD operations and historical tracking.
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path



class WatchlistManager:
    """Manages penny stock watchlists with SQLite persistence"""
    
    def __init__(self, db_path: str = "data/watchlist.db"):
        """
        Initialize watchlist manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Watchlist table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    sector TEXT,
                    price REAL,
                    pct_change REAL,
                    volume INTEGER,
                    avg_volume INTEGER,
                    float_m REAL,
                    market_cap REAL,
                    pe_ratio REAL,
                    revenue_growth REAL,
                    profit_margin REAL,
                    analyst_rating TEXT,
                    analyst_target REAL,
                    technical_score REAL,
                    rsi REAL,
                    ma_signal TEXT,
                    news_sentiment TEXT,
                    news_count INTEGER,
                    news_summary TEXT,
                    buzz TEXT,
                    catalyst TEXT,
                    insider TEXT,
                    cash_debt TEXT,
                    risk TEXT,
                    dilution TEXT,
                    verified TEXT,
                    momentum_score REAL,
                    valuation_score REAL,
                    catalyst_score REAL,
                    composite_score REAL,
                    confidence_level TEXT,
                    reasoning TEXT,
                    risk_narrative TEXT,
                    notes TEXT,
                    date_added TEXT,
                    last_updated TEXT
                )
            """)
            
            # Historical scores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    price REAL,
                    momentum_score REAL,
                    valuation_score REAL,
                    catalyst_score REAL,
                    composite_score REAL,
                    confidence_level TEXT,
                    FOREIGN KEY (ticker) REFERENCES watchlist(ticker)
                )
            """)
            
            # Tags table for categorization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (ticker) REFERENCES watchlist(ticker),
                    UNIQUE(ticker, tag)
                )
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def add_stock(self, stock_data: Dict) -> bool:
        """
        Add or update a stock in the watchlist
        
        Args:
            stock_data: Dictionary containing stock information
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                ticker = stock_data.get('ticker', '').upper()
                if not ticker:
                    logger.error("Ticker is required")
                    return False
                
                # Check if stock exists
                cursor.execute("SELECT id FROM watchlist WHERE ticker = ?", (ticker,))
                exists = cursor.fetchone()
                
                current_time = datetime.now().isoformat()
                
                if exists:
                    # Update existing stock
                    update_fields = []
                    update_values = []
                    
                    for key, value in stock_data.items():
                        if key != 'ticker' and key != 'date_added':
                            update_fields.append(f"{key} = ?")
                            update_values.append(value)
                    
                    update_fields.append("last_updated = ?")
                    update_values.append(current_time)
                    update_values.append(ticker)
                    
                    query = f"UPDATE watchlist SET {', '.join(update_fields)} WHERE ticker = ?"
                    cursor.execute(query, update_values)
                    logger.info(f"Updated {ticker} in watchlist")
                else:
                    # Insert new stock
                    stock_data['date_added'] = current_time
                    stock_data['last_updated'] = current_time
                    
                    fields = ', '.join(stock_data.keys())
                    placeholders = ', '.join(['?' for _ in stock_data])
                    query = f"INSERT INTO watchlist ({fields}) VALUES ({placeholders})"
                    
                    cursor.execute(query, list(stock_data.values()))
                    logger.info(f"Added {ticker} to watchlist")
                
                # Add to score history
                if 'composite_score' in stock_data:
                    self._add_score_history(cursor, ticker, stock_data)
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error adding stock to watchlist: {e}")
            return False
    
    def _add_score_history(self, cursor, ticker: str, data: Dict):
        """Add score snapshot to history"""
        try:
            cursor.execute("""
                INSERT INTO score_history 
                (ticker, date, price, momentum_score, valuation_score, 
                 catalyst_score, composite_score, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                datetime.now().isoformat(),
                data.get('price'),
                data.get('momentum_score'),
                data.get('valuation_score'),
                data.get('catalyst_score'),
                data.get('composite_score'),
                data.get('confidence_level')
            ))
        except Exception as e:
            logger.error(f"Error adding score history: {e}")
    
    def remove_stock(self, ticker: str) -> bool:
        """
        Remove a stock from the watchlist
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
                cursor.execute("DELETE FROM score_history WHERE ticker = ?", (ticker.upper(),))
                cursor.execute("DELETE FROM tags WHERE ticker = ?", (ticker.upper(),))
                conn.commit()
                logger.info(f"Removed {ticker} from watchlist")
                return True
        except Exception as e:
            logger.error(f"Error removing stock: {e}")
            return False
    
    def get_stock(self, ticker: str) -> Optional[Dict]:
        """
        Get a single stock's data
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with stock data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM watchlist WHERE ticker = ?", (ticker.upper(),))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting stock: {e}")
            return None
    
    def get_all_stocks(self, sort_by: str = 'composite_score', 
                       ascending: bool = False) -> pd.DataFrame:
        """
        Get all stocks in the watchlist
        
        Args:
            sort_by: Column to sort by
            ascending: Sort order
        
        Returns:
            DataFrame with all stocks
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM watchlist"
                df = pd.read_sql_query(query, conn)
                
                if not df.empty and sort_by in df.columns:
                    df = df.sort_values(by=sort_by, ascending=ascending)
                
                return df
        except Exception as e:
            logger.error(f"Error getting all stocks: {e}")
            return pd.DataFrame()
    
    def filter_stocks(self, filters: Dict) -> pd.DataFrame:
        """
        Filter stocks based on criteria
        
        Args:
            filters: Dictionary of filter criteria
                - min_composite_score: Minimum composite score
                - max_composite_score: Maximum composite score
                - confidence_levels: List of confidence levels
                - sectors: List of sectors
                - min_momentum: Minimum momentum score
                - min_valuation: Minimum valuation score
                - min_catalyst: Minimum catalyst score
                - risk_levels: List of risk levels
        
        Returns:
            Filtered DataFrame
        """
        try:
            df = self.get_all_stocks()
            
            if df.empty:
                return df
            
            # Apply filters
            if 'min_composite_score' in filters:
                df = df[df['composite_score'] >= filters['min_composite_score']]
            
            if 'max_composite_score' in filters:
                df = df[df['composite_score'] <= filters['max_composite_score']]
            
            if 'confidence_levels' in filters and filters['confidence_levels']:
                df = df[df['confidence_level'].isin(filters['confidence_levels'])]
            
            if 'sectors' in filters and filters['sectors']:
                df = df[df['sector'].isin(filters['sectors'])]
            
            if 'min_momentum' in filters:
                df = df[df['momentum_score'] >= filters['min_momentum']]
            
            if 'min_valuation' in filters:
                df = df[df['valuation_score'] >= filters['min_valuation']]
            
            if 'min_catalyst' in filters:
                df = df[df['catalyst_score'] >= filters['min_catalyst']]
            
            if 'risk_levels' in filters and filters['risk_levels']:
                df = df[df['risk'].isin(filters['risk_levels'])]
            
            return df
            
        except Exception as e:
            logger.error(f"Error filtering stocks: {e}")
            return pd.DataFrame()
    
    def get_score_history(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical scores for a stock
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of history to retrieve
        
        Returns:
            DataFrame with historical scores
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM score_history 
                    WHERE ticker = ? 
                    ORDER BY date DESC 
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(ticker.upper(), days))
                return df
        except Exception as e:
            logger.error(f"Error getting score history: {e}")
            return pd.DataFrame()
    
    def add_tag(self, ticker: str, tag: str) -> bool:
        """Add a tag to a stock"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO tags (ticker, tag) VALUES (?, ?)",
                    (ticker.upper(), tag.lower())
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding tag: {e}")
            return False
    
    def remove_tag(self, ticker: str, tag: str) -> bool:
        """Remove a tag from a stock"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM tags WHERE ticker = ? AND tag = ?",
                    (ticker.upper(), tag.lower())
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error removing tag: {e}")
            return False
    
    def get_tags(self, ticker: str) -> List[str]:
        """Get all tags for a stock"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT tag FROM tags WHERE ticker = ?", (ticker.upper(),))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting tags: {e}")
            return []
    
    def get_stocks_by_tag(self, tag: str) -> pd.DataFrame:
        """Get all stocks with a specific tag"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT w.* FROM watchlist w
                    INNER JOIN tags t ON w.ticker = t.ticker
                    WHERE t.tag = ?
                """
                df = pd.read_sql_query(query, conn, params=(tag.lower(),))
                return df
        except Exception as e:
            logger.error(f"Error getting stocks by tag: {e}")
            return pd.DataFrame()
    
    def export_to_csv(self, filepath: str) -> bool:
        """Export watchlist to CSV file"""
        try:
            df = self.get_all_stocks()
            df.to_csv(filepath, index=False)
            logger.info(f"Exported watchlist to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def export_to_json(self, filepath: str) -> bool:
        """Export watchlist to JSON file"""
        try:
            df = self.get_all_stocks()
            df.to_json(filepath, orient='records', indent=2)
            logger.info(f"Exported watchlist to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False
    
    def import_from_csv(self, filepath: str) -> int:
        """
        Import stocks from CSV file
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            Number of stocks imported
        """
        try:
            df = pd.read_csv(filepath)
            count = 0
            
            for _, row in df.iterrows():
                stock_data = row.to_dict()
                if self.add_stock(stock_data):
                    count += 1
            
            logger.info(f"Imported {count} stocks from {filepath}")
            return count
        except Exception as e:
            logger.error(f"Error importing from CSV: {e}")
            return 0
    
    def get_statistics(self) -> Dict:
        """Get watchlist statistics"""
        try:
            df = self.get_all_stocks()
            
            if df.empty:
                return {
                    'total_stocks': 0,
                    'avg_composite_score': 0,
                    'confidence_distribution': {},
                    'sector_distribution': {},
                    'high_confidence_count': 0
                }
            
            stats = {
                'total_stocks': len(df),
                'avg_composite_score': df['composite_score'].mean() if 'composite_score' in df else 0,
                'avg_momentum_score': df['momentum_score'].mean() if 'momentum_score' in df else 0,
                'avg_valuation_score': df['valuation_score'].mean() if 'valuation_score' in df else 0,
                'avg_catalyst_score': df['catalyst_score'].mean() if 'catalyst_score' in df else 0,
                'confidence_distribution': df['confidence_level'].value_counts().to_dict() if 'confidence_level' in df else {},
                'sector_distribution': df['sector'].value_counts().to_dict() if 'sector' in df else {},
                'high_confidence_count': len(df[df['confidence_level'].isin(['HIGH', 'VERY HIGH'])]) if 'confidence_level' in df else 0,
                'last_updated': df['last_updated'].max() if 'last_updated' in df else None
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def clear_watchlist(self) -> bool:
        """Clear all stocks from watchlist (use with caution)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM watchlist")
                cursor.execute("DELETE FROM score_history")
                cursor.execute("DELETE FROM tags")
                conn.commit()
                logger.info("Cleared all stocks from watchlist")
                return True
        except Exception as e:
            logger.error(f"Error clearing watchlist: {e}")
            return False

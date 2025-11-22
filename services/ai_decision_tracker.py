"""
AI Decision Tracker
Tracks AI trading decisions and their outcomes for continuous improvement

Features:
- Log all AI decisions with full context
- Track outcomes (P&L, accuracy, effectiveness)
- Generate performance analytics
- Identify improvement opportunities
- A/B testing support

PHASE 1 IMPLEMENTATION (Simple but Effective)
"""

import os
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
from pathlib import Path


@dataclass
class AIDecisionRecord:
    """Record of a single AI decision"""
    timestamp: str
    trade_id: str
    symbol: str
    asset_type: str  # 'crypto' or 'stock'
    action: str  # 'HOLD', 'CLOSE_NOW', 'TIGHTEN_STOP', etc.
    confidence: float
    reasoning: str
    
    # Position context at decision time
    current_price: float
    entry_price: float
    pnl_pct: float
    hold_time_minutes: float
    
    # AI scoring
    technical_score: float
    trend_score: float
    risk_score: float
    
    # News/sentiment context (if available)
    sentiment_score: Optional[float] = None
    news_count: int = 0
    high_impact_news: bool = False
    
    # Outcome (filled later)
    outcome: Optional[str] = None  # 'PROFITABLE', 'LOSS', 'BREAKEVEN'
    final_pnl_pct: Optional[float] = None
    outcome_quality: Optional[float] = None  # 0-100: how good was this decision
    closed_at: Optional[str] = None
    
    # Decision effectiveness metrics (filled after trade closes)
    was_correct: Optional[bool] = None  # Did this decision improve the outcome?
    improvement_pct: Optional[float] = None  # How much better vs. holding


class AIDecisionTracker:
    """
    Track AI trading decisions and analyze their effectiveness
    Supports continuous learning and improvement
    """
    
    def __init__(self, db_path: str = "data/ai_decisions.db"):
        """
        Initialize AI decision tracker
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        
        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"📊 AI Decision Tracker initialized (DB: {db_path})")
    
    def _init_db(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL,
                reasoning TEXT,
                current_price REAL,
                entry_price REAL,
                pnl_pct REAL,
                hold_time_minutes REAL,
                technical_score REAL,
                trend_score REAL,
                risk_score REAL,
                sentiment_score REAL,
                news_count INTEGER,
                high_impact_news INTEGER,
                outcome TEXT,
                final_pnl_pct REAL,
                outcome_quality REAL,
                closed_at TEXT,
                was_correct INTEGER,
                improvement_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_id ON decisions(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON decisions(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON decisions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action ON decisions(action)")
        
        conn.commit()
        conn.close()
        
        logger.debug("AI Decision Tracker database initialized")
    
    def log_decision(
        self,
        trade_id: str,
        symbol: str,
        asset_type: str,
        action: str,
        confidence: float,
        reasoning: str,
        position_state: Dict,
        ai_scores: Dict,
        sentiment_data: Optional[Dict] = None
    ):
        """
        Log an AI trading decision
        
        Args:
            trade_id: Unique trade identifier
            symbol: Asset symbol (e.g., 'BTC/USD', 'AAPL')
            asset_type: 'crypto' or 'stock'
            action: AI recommended action
            confidence: Confidence score (0-100)
            reasoning: AI reasoning text
            position_state: Dict with current position data
            ai_scores: Dict with technical_score, trend_score, risk_score
            sentiment_data: Optional dict with sentiment analysis
        """
        try:
            record = AIDecisionRecord(
                timestamp=datetime.now().isoformat(),
                trade_id=trade_id,
                symbol=symbol,
                asset_type=asset_type,
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                current_price=position_state.get('current_price', 0.0),
                entry_price=position_state.get('entry_price', 0.0),
                pnl_pct=position_state.get('pnl_pct', 0.0),
                hold_time_minutes=position_state.get('hold_time_minutes', 0.0),
                technical_score=ai_scores.get('technical_score', 0.0),
                trend_score=ai_scores.get('trend_score', 0.0),
                risk_score=ai_scores.get('risk_score', 0.0),
                sentiment_score=sentiment_data.get('sentiment_score') if sentiment_data else None,
                news_count=sentiment_data.get('news_count', 0) if sentiment_data else 0,
                high_impact_news=sentiment_data.get('high_impact', False) if sentiment_data else False
            )
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO decisions (
                    timestamp, trade_id, symbol, asset_type, action, confidence, reasoning,
                    current_price, entry_price, pnl_pct, hold_time_minutes,
                    technical_score, trend_score, risk_score,
                    sentiment_score, news_count, high_impact_news
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp, record.trade_id, record.symbol, record.asset_type,
                record.action, record.confidence, record.reasoning,
                record.current_price, record.entry_price, record.pnl_pct,
                record.hold_time_minutes, record.technical_score, record.trend_score,
                record.risk_score, record.sentiment_score, record.news_count,
                1 if record.high_impact_news else 0
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Logged AI decision: {symbol} - {action} (Confidence: {confidence:.1f}%)")
            
        except Exception as e:
            logger.error("Error logging AI decision: {}", str(e), exc_info=True)
    
    def update_outcome(
        self,
        trade_id: str,
        final_pnl_pct: float,
        outcome: str = None
    ):
        """
        Update decision outcome after trade closes
        
        Args:
            trade_id: Trade identifier
            final_pnl_pct: Final P&L percentage
            outcome: 'PROFITABLE', 'LOSS', or 'BREAKEVEN'
        """
        try:
            # Determine outcome if not provided
            if outcome is None:
                if final_pnl_pct > 0.5:
                    outcome = 'PROFITABLE'
                elif final_pnl_pct < -0.5:
                    outcome = 'LOSS'
                else:
                    outcome = 'BREAKEVEN'
            
            # Calculate outcome quality (how good was this result)
            outcome_quality = self._calculate_outcome_quality(trade_id, final_pnl_pct)
            
            # Evaluate if decision was correct
            was_correct, improvement_pct = self._evaluate_decision_correctness(trade_id, final_pnl_pct)
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE decisions
                SET outcome = ?,
                    final_pnl_pct = ?,
                    outcome_quality = ?,
                    closed_at = ?,
                    was_correct = ?,
                    improvement_pct = ?
                WHERE trade_id = ? AND outcome IS NULL
            """, (
                outcome,
                final_pnl_pct,
                outcome_quality,
                datetime.now().isoformat(),
                1 if was_correct else 0,
                improvement_pct,
                trade_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Updated outcome for {trade_id}: {outcome} ({final_pnl_pct:+.2f}%)")
            
        except Exception as e:
            logger.error("Error updating outcome for {trade_id}: {}", str(e), exc_info=True)
    
    def _calculate_outcome_quality(self, trade_id: str, final_pnl_pct: float) -> float:
        """
        Calculate quality score (0-100) for the outcome
        Considers: final P&L, risk taken, decision timing
        """
        try:
            # Get decision record
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pnl_pct, confidence, action, risk_score
                FROM decisions
                WHERE trade_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (trade_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return 50.0  # Neutral
            
            pnl_at_decision, confidence, action, risk_score = row
            
            # Base score from P&L
            if final_pnl_pct > 5:
                base_score = 90
            elif final_pnl_pct > 2:
                base_score = 75
            elif final_pnl_pct > 0:
                base_score = 60
            elif final_pnl_pct > -2:
                base_score = 40
            elif final_pnl_pct > -5:
                base_score = 25
            else:
                base_score = 10
            
            # Adjust based on decision action
            if action == 'CLOSE_NOW':
                # Good if saved from loss or took good profit
                if pnl_at_decision > 0 and final_pnl_pct > pnl_at_decision * 0.8:
                    base_score += 10  # Took profit at good time
                elif pnl_at_decision < 0 and final_pnl_pct > pnl_at_decision:
                    base_score += 15  # Cut losses effectively
            
            elif action == 'HOLD':
                # Good if position continued to improve
                if final_pnl_pct > pnl_at_decision:
                    base_score += 5
            
            return min(100.0, max(0.0, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating outcome quality: {e}")
            return 50.0  # Neutral default
    
    def _evaluate_decision_correctness(
        self,
        trade_id: str,
        final_pnl_pct: float
    ) -> Tuple[bool, float]:
        """
        Evaluate if the AI decision was correct
        
        Returns:
            (was_correct, improvement_pct)
        """
        try:
            # Get decision context
            conn = sqlite3.connect(self.db_path)
            cursor = cursor.execute("""
                SELECT pnl_pct, action, confidence
                FROM decisions
                WHERE trade_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (trade_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return False, 0.0
            
            pnl_at_decision, action, confidence = row
            
            # Evaluate based on action
            improvement_pct = final_pnl_pct - pnl_at_decision
            
            if action == 'CLOSE_NOW':
                # Correct if final P&L would have been worse by continuing
                was_correct = improvement_pct <= 0 or (pnl_at_decision > 2 and final_pnl_pct > 0)
            
            elif action == 'TIGHTEN_STOP':
                # Correct if it prevented larger loss
                was_correct = improvement_pct >= -2.0
            
            elif action == 'TAKE_PARTIAL':
                # Correct if locked in profits before reversal
                was_correct = pnl_at_decision > 0 and (improvement_pct <= 2.0 or final_pnl_pct > 0)
            
            elif action == 'HOLD':
                # Correct if position continued favorably
                was_correct = improvement_pct >= -1.0
            
            else:
                was_correct = final_pnl_pct > 0
            
            return was_correct, improvement_pct
            
        except Exception as e:
            logger.error(f"Error evaluating decision correctness: {e}")
            return False, 0.0
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """
        Generate performance report for AI decisions
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dict with performance metrics
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all decisions in period
            cursor.execute("""
                SELECT * FROM decisions
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff_date,))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            
            if not rows:
                return {
                    'period_days': days,
                    'total_decisions': 0,
                    'message': 'No decisions in this period'
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(rows, columns=columns)
            
            # Calculate metrics
            total_decisions = len(df)
            closed_decisions = df[df['outcome'].notna()]
            
            report = {
                'period_days': days,
                'total_decisions': total_decisions,
                'closed_decisions': len(closed_decisions),
                'pending_decisions': total_decisions - len(closed_decisions),
                
                # Action breakdown
                'actions_breakdown': df['action'].value_counts().to_dict(),
                
                # Confidence metrics
                'average_confidence': float(df['confidence'].mean()),
                'high_confidence_decisions': int((df['confidence'] >= 80).sum()),
                
                # Outcome metrics (for closed decisions)
                'profitable_decisions': int((closed_decisions['final_pnl_pct'] > 0).sum()) if len(closed_decisions) > 0 else 0,
                'loss_decisions': int((closed_decisions['final_pnl_pct'] < 0).sum()) if len(closed_decisions) > 0 else 0,
                'average_pnl': float(closed_decisions['final_pnl_pct'].mean()) if len(closed_decisions) > 0 else 0.0,
                'win_rate': float((closed_decisions['final_pnl_pct'] > 0).sum() / len(closed_decisions) * 100) if len(closed_decisions) > 0 else 0.0,
                
                # Decision quality
                'average_outcome_quality': float(closed_decisions['outcome_quality'].mean()) if len(closed_decisions) > 0 else 0.0,
                'correct_decisions': int((closed_decisions['was_correct'] == 1).sum()) if len(closed_decisions) > 0 else 0,
                'decision_accuracy': float((closed_decisions['was_correct'] == 1).sum() / len(closed_decisions) * 100) if len(closed_decisions) > 0 else 0.0,
                
                # Sentiment analysis
                'decisions_with_news': int((df['news_count'] > 0).sum()),
                'high_impact_news_decisions': int((df['high_impact_news'] == 1).sum()),
                
                # Best/worst actions
                'best_performing_action': self._get_best_action(df),
                'worst_performing_action': self._get_worst_action(df),
                
                # Recommendations
                'recommendations': self._generate_recommendations(df)
            }
            
            return report
            
        except Exception as e:
            logger.error("Error generating performance report: {}", str(e), exc_info=True)
            return {'error': str(e)}
    
    def _get_best_action(self, df: pd.DataFrame) -> str:
        """Find best performing action type"""
        try:
            closed = df[df['outcome'].notna()]
            if len(closed) == 0:
                return "N/A"
            
            action_performance = closed.groupby('action')['final_pnl_pct'].mean()
            best_action = action_performance.idxmax()
            best_pnl = action_performance.max()
            
            return f"{best_action} ({best_pnl:+.2f}% avg)"
        except:
            return "N/A"
    
    def _get_worst_action(self, df: pd.DataFrame) -> str:
        """Find worst performing action type"""
        try:
            closed = df[df['outcome'].notna()]
            if len(closed) == 0:
                return "N/A"
            
            action_performance = closed.groupby('action')['final_pnl_pct'].mean()
            worst_action = action_performance.idxmin()
            worst_pnl = action_performance.min()
            
            return f"{worst_action} ({worst_pnl:+.2f}% avg)"
        except:
            return "N/A"
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate improvement recommendations based on data"""
        recommendations = []
        
        try:
            closed = df[df['outcome'].notna()]
            
            if len(closed) == 0:
                return ["Insufficient data for recommendations"]
            
            # Check decision accuracy
            if len(closed) >= 10:
                accuracy = (closed['was_correct'] == 1).sum() / len(closed)
                if accuracy < 0.6:
                    recommendations.append(
                        f"Decision accuracy is {accuracy:.1%}. Consider adjusting confidence thresholds or improving technical analysis."
                    )
            
            # Check sentiment utilization
            with_news = (df['news_count'] > 0).sum()
            if with_news / len(df) < 0.5:
                recommendations.append(
                    f"Only {with_news/len(df):.1%} of decisions used news sentiment. Consider increasing news integration."
                )
            
            # Check confidence vs. outcome correlation
            if len(closed) >= 10:
                high_conf = closed[closed['confidence'] >= 80]
                if len(high_conf) > 0:
                    high_conf_accuracy = (high_conf['final_pnl_pct'] > 0).sum() / len(high_conf)
                    if high_conf_accuracy < 0.7:
                        recommendations.append(
                            f"High-confidence decisions only {high_conf_accuracy:.1%} accurate. Model may be overconfident."
                        )
            
            # Check action-specific performance
            action_perf = closed.groupby('action')['final_pnl_pct'].agg(['mean', 'count'])
            for action, stats in action_perf.iterrows():
                if stats['count'] >= 3 and stats['mean'] < -2.0:
                    recommendations.append(
                        f"Action '{action}' averages {stats['mean']:.2f}% P&L. Review decision logic for this action."
                    )
            
            if not recommendations:
                recommendations.append("System performing well! Continue monitoring.")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error analyzing data for recommendations")
        
        return recommendations


# Global instance for easy access
_decision_tracker = None

def get_decision_tracker() -> AIDecisionTracker:
    """Get or create singleton decision tracker instance"""
    global _decision_tracker
    
    if _decision_tracker is None:
        _decision_tracker = AIDecisionTracker()
    
    return _decision_tracker


# Example usage
if __name__ == "__main__":
    tracker = AIDecisionTracker()
    
    # Example: Log a decision
    tracker.log_decision(
        trade_id="BTC_123",
        symbol="BTC/USD",
        asset_type="crypto",
        action="TIGHTEN_STOP",
        confidence=85.0,
        reasoning="Bearish news + overbought RSI",
        position_state={
            'current_price': 46500,
            'entry_price': 45000,
            'pnl_pct': 3.33,
            'hold_time_minutes': 125
        },
        ai_scores={
            'technical_score': 70,
            'trend_score': 65,
            'risk_score': 55
        },
        sentiment_data={
            'sentiment_score': 35.0,
            'news_count': 3,
            'high_impact': True
        }
    )
    
    # Example: Update outcome
    tracker.update_outcome(
        trade_id="BTC_123",
        final_pnl_pct=2.8
    )
    
    # Example: Get performance report
    report = tracker.get_performance_report(days=30)
    print("\n" + "="*80)
    print("AI DECISION PERFORMANCE REPORT")
    print("="*80)
    print(json.dumps(report, indent=2))


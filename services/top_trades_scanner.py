"""
Top Trades Scanner

Scans and ranks top trading opportunities for options and penny stocks
based on multiple criteria including volume, momentum, and scoring.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from loguru import logger
from .penny_stock_analyzer import PennyStockAnalyzer
from services.penny_stock_constants import PENNY_THRESHOLDS, get_price_tier_bonus
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time



@dataclass
class TopTrade:
    """Container for a top trade opportunity"""
    ticker: str
    score: float
    price: float
    change_pct: float
    volume: int
    volume_ratio: float
    reason: str
    trade_type: str  # 'options' or 'penny_stock'
    confidence: str
    risk_level: str


class TopTradesScanner:
    """Scans and identifies top trading opportunities"""
    
    # Popular tickers to scan (can be customized)
    # EXPANDED UNIVERSE: Focus on affordable stocks for small capital accounts
    OPTIONS_UNIVERSE = [
        # Affordable Tech & Growth (Under $50)
        'SOFI', 'HOOD', 'SNAP', 'PINS', 'RBLX', 'DKNG', 'OPEN', 'PLBY',
        'INTC', 'AMD', 'QCOM', 'TXN', 'MU', 'WDC', 'STX',
        # EV & Clean Energy (Under $50)
        'RIVN', 'LCID', 'NKLA', 'FSR', 'RIDE', 'GOEV', 'WKHS',
        'PLUG', 'FCEL', 'BLDP', 'BE', 'RUN', 'ENPH', 'SEDG',
        'CHPT', 'BLNK', 'EVGO', 'QS', 'STEM', 'CLSK',
        # Crypto & Fintech (Under $50)
        'COIN', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF', 'ARBK',
        'SQ', 'PYPL', 'AFRM', 'UPST', 'LC', 'NU', 'PAYO',
        # Telecom (Under $50)
        'NOK', 'ERIC', 'T', 'VZ', 'TMUS', 'LUMN', 'SATS',
        # Meme & High Volume (Under $50)
        'GME', 'AMC', 'BB', 'BBBY', 'EXPR', 'KOSS',
        # Finance (Under $50)
        'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'ALLY',
        # Energy (Under $50)
        'XOM', 'CVX', 'COP', 'SLB', 'HAL', 'OXY', 'DVN',
        'MRO', 'APA', 'FANG', 'CLR', 'MPC', 'VLO', 'PSX',
        # Travel & Hospitality (Under $50)
        'AAL', 'UAL', 'DAL', 'LUV', 'JBLU', 'SAVE', 'ALK',
        'CCL', 'RCL', 'NCLH', 'EXPE', 'LYFT', 'UBER',
        # Cannabis (Under $50)
        'TLRY', 'CGC', 'ACB', 'SNDL', 'HEXO', 'OGI', 'CRON',
        # Chinese ADRs (Under $50)
        'NIO', 'XPEV', 'LI', 'IQ', 'BILI', 'DIDI', 'GRAB',
        # Biotech (Under $50)
        'MRNA', 'BNTX', 'NVAX', 'VXRT', 'INO', 'OCGN', 'SRNE',
        # Large Caps (for reference - will be filtered by price)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
        'PLTR', 'SHOP', 'SNOW', 'NET', 'CRWD', 'DDOG', 'MDB',
        # Indices/ETFs
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV'
    ]
    
    PENNY_STOCK_UNIVERSE = [
        # Popular penny stocks
        'SOFI', 'NIO', 'LCID', 'RIVN', 'PLUG', 'FCEL', 'SNDL', 
        'CLOV', 'WISH', 'SKLZ', 'GSAT', 'BNGO', 'ZOM', 'TXMD', 'IDEX',
        'SIRI', 'NOK', 'BB', 'SENS', 'CLVS', 'OCGN',
        
        # Obscure biotech/pharma pennies (high volatility, breakout potential)
        'NVCR', 'ZLAB', 'CASI', 'FREQ', 'CRBP', 'VXRT', 'MRNA', 'DVAX',
        'EPIX', 'EGAN', 'RGNX', 'PSTV', 'AKBA', 'ARDX', 'SNOA', 'MNKD',
        'TPTX', 'VKTX', 'ALNY', 'BLUE', 'SGMO', 'CRSP', 'EDIT', 'NTLA',
        
        # Mining & resources (commodity plays, breakout on news)
        'GOLD', 'AUY', 'AG', 'CDE', 'HL', 'GPL', 'EXK', 'FSM', 'PAAS',
        'SVM', 'MARA', 'RIOT', 'BITF', 'HUT', 'CLSK', 'ARBK',
        
        # Energy & clean tech (emerging sector momentum)
        'TELL', 'MAXN', 'RUN', 'NOVA', 'SEDG', 'ENPH', 'BE', 'QS', 'BLNK',
        'CHPT', 'EVGO', 'FSR', 'GOEV', 'WKHS', 'RIDE', 'HYLN', 'NKLA',
        
        # Crypto-related (high beta, breakout potential)
        'COIN', 'MSTR', 'SI', 'RIOT', 'MARA', 'BTBT', 'CAN', 'SOS',
        'EBON', 'FTFT', 'GREE', 'BTCM', 'WULF',
        
        # Tech/emerging (obscure growth plays)
        'SOUN', 'BBAI', 'KSCP', 'FRZA', 'VRAR', 'VUZI', 'KOPN', 'AEHR',
        'WOLF', 'MVIS', 'LAZR', 'LIDR', 'OUST', 'VLDR', 'INVZ',
        
        # Cannabis (high volatility, news-driven)
        'TLRY', 'CGC', 'ACB', 'HEXO', 'OGI', 'CRON', 'CURLF', 'GTBIF',
        
        # Shipping/transport (economic plays)
        'TOPS', 'SHIP', 'SBLK', 'STNG', 'DHT', 'FRO', 'EURN', 'NAT',
        
        # Spec plays (high risk/reward)
        'MULN', 'FFIE', 'AMTD', 'HKD', 'GFAI', 'ATAI', 'HOLO', 'IMPP'
    ]
    
    def __init__(self, use_optimizations: bool = True):
        self.penny_analyzer = PennyStockAnalyzer()
        self.use_optimizations = use_optimizations
        
        # Initialize optimizations if enabled
        if self.use_optimizations:
            try:
                from .scanner_integration import create_optimized_scanner
                self.optimized_scanner = create_optimized_scanner()
                logger.info("✅ Optimizations enabled for TopTradesScanner")
            except ImportError:
                self.use_optimizations = False
                logger.warning("⚠️ Optimizations not available, falling back to standard processing")
    
    def scan_top_options_trades(self, top_n: int = 20, use_parallel: bool = True) -> List[TopTrade]:
        """
        Scan for top options trading opportunities
        
        Args:
            top_n: Number of top trades to return
            use_parallel: Use parallel processing for 4-8x speedup (default: True)
        
        Returns:
            List of TopTrade objects sorted by score
        """
        logger.info(f"Scanning for top {top_n} options trades...")
        
        if self.use_optimizations and hasattr(self, 'optimized_scanner'):
            # Use optimized parallel processing
            try:
                logger.info("🚀 Using optimized parallel processing")
                optimized_results = self.optimized_scanner.analyze_options_opportunities_optimized(
                    self.OPTIONS_UNIVERSE, top_n
                )
                
                # Convert to TopTrade objects
                results = []
                for result in optimized_results:
                    trade = TopTrade(
                        ticker=result['ticker'],
                        score=result['score'],
                        price=result['price'],
                        change_pct=result['change_pct'],
                        volume=result.get('volume', 0),
                        volume_ratio=result['volume_ratio'],
                        reason=result['reason'],
                        trade_type='options',
                        confidence=result['confidence'],
                        risk_level=result['risk_level']
                    )
                    results.append(trade)
                
                logger.info(f"✅ Found {len(results)} options opportunities (optimized)")
                return results[:top_n]
                
            except Exception as e:
                logger.error(f"Optimization failed, falling back to standard: {e}")
        
        # Use parallel or sequential processing
        if use_parallel:
            return self._scan_options_trades_parallel(top_n)
        else:
            return self._scan_options_trades_sequential(top_n)
    
    def _scan_options_trades_sequential(self, top_n: int) -> List[TopTrade]:
        """Sequential options scanning (fallback)"""
        logger.info(f"📊 Using sequential processing for {len(self.OPTIONS_UNIVERSE)} options tickers")
        
        results = []
        
        for ticker in self.OPTIONS_UNIVERSE:
            try:
                trade = self._analyze_options_opportunity(ticker)
                if trade and trade.score > 0:
                    results.append(trade)
            except Exception as e:
                logger.error(f"Error analyzing {ticker} for options: {e}")
                continue
        
        # Sort by score and return top N
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Found {len(results)} options opportunities")
        
        return results[:top_n]
    
    def _scan_options_trades_parallel(self, top_n: int, max_workers: int = 8) -> List[TopTrade]:
        """Parallel options scanning using ThreadPoolExecutor (4-8x faster)"""
        logger.info(f"🚀 Using parallel processing ({max_workers} workers) for {len(self.OPTIONS_UNIVERSE)} options tickers")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_ticker = {
                executor.submit(self._analyze_options_opportunity, ticker): ticker
                for ticker in self.OPTIONS_UNIVERSE
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                
                try:
                    trade = future.result(timeout=10)
                    if trade and trade.score > 0:
                        results.append(trade)
                except Exception as e:
                    logger.error(f"Error analyzing {ticker} for options: {e}")
                    continue
                
                # Log progress every 10 tickers
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(self.OPTIONS_UNIVERSE)} tickers analyzed")
        
        # Sort by score and return top N
        results.sort(key=lambda x: x.score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Found {len(results)} options opportunities in {elapsed_time:.2f}s")
        
        return results[:top_n]
    
    def scan_top_penny_stocks(self, top_n: int = 20, use_parallel: bool = True) -> List[TopTrade]:
        """
        Scan for top penny stock opportunities
        
        Args:
            top_n: Number of top trades to return
            use_parallel: Use parallel processing for 4-8x speedup (default: True)
        
        Returns:
            List of TopTrade objects sorted by score
        """
        logger.info(f"Scanning for top {top_n} penny stocks...")
        
        if use_parallel:
            return self._scan_penny_stocks_parallel(top_n)
        else:
            return self._scan_penny_stocks_sequential(top_n)
    
    def _scan_penny_stocks_sequential(self, top_n: int) -> List[TopTrade]:
        """Sequential penny stock scanning (fallback)"""
        logger.info(f"📊 Using sequential processing for {len(self.PENNY_STOCK_UNIVERSE)} penny stocks")
        
        results = []
        
        for ticker in self.PENNY_STOCK_UNIVERSE:
            try:
                trade = self._analyze_penny_stock_opportunity(ticker)
                if trade and trade.score > 0:
                    results.append(trade)
            except Exception as e:
                logger.error(f"Error analyzing {ticker} for penny stocks: {e}")
                continue
        
        # Sort by score and return top N
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Found {len(results)} penny stock opportunities")
        
        return results[:top_n]
    
    def _scan_penny_stocks_parallel(self, top_n: int, max_workers: int = 8) -> List[TopTrade]:
        """Parallel penny stock scanning using ThreadPoolExecutor (4-8x faster)"""
        logger.info(f"🚀 Using parallel processing ({max_workers} workers) for {len(self.PENNY_STOCK_UNIVERSE)} penny stocks")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_ticker = {
                executor.submit(self._analyze_penny_stock_opportunity, ticker): ticker
                for ticker in self.PENNY_STOCK_UNIVERSE
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                
                try:
                    trade = future.result(timeout=10)
                    if trade and trade.score > 0:
                        results.append(trade)
                except Exception as e:
                    logger.error(f"Error analyzing {ticker} for penny stocks: {e}")
                    continue
                
                # Log progress every 10 tickers
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(self.PENNY_STOCK_UNIVERSE)} tickers analyzed")
        
        # Sort by score and return top N
        results.sort(key=lambda x: x.score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Found {len(results)} penny stock opportunities in {elapsed_time:.2f}s")
        
        return results[:top_n]
    
    def _analyze_options_opportunity(self, ticker: str) -> TopTrade:
        """Analyze a stock for options trading potential"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get recent data (extended for better breakout detection)
            hist = stock.history(period="3mo")
            if hist.empty or len(hist) < 10:
                return None
            
            info = stock.info
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price / prev_close - 1) * 100)
            
            # Filter: Only true penny stocks (<$5) and obscure plays
            # Prioritize lower prices with higher breakout potential
            if current_price > 5.0:
                return None  # Skip higher-priced stocks
            
            # Bonus for ultra-low prices (more breakout room)
            ultra_low_bonus = 0
            if current_price < 1.0:
                ultra_low_bonus = 15
            elif current_price < 2.0:
                ultra_low_bonus = 10
            elif current_price < 3.0:
                ultra_low_bonus = 5
            
            # Volume analysis
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Calculate option trading score
            score = 40.0  # Base score (lowered to allow more granular scoring)
            reasons = []
            
            # Volume spike (30 points)
            if volume_ratio > 2.0:
                score += 30
                reasons.append(f"📊 Volume spike ({volume_ratio:.1f}x avg)")
            elif volume_ratio > 1.5:
                score += 20
                reasons.append(f"📈 High volume ({volume_ratio:.1f}x avg)")
            elif volume_ratio > 1.0:
                score += 15
            elif volume_ratio > 0.5:
                score += 5  # Still add some points for reasonable volume
            
            # Price movement (25 points)
            if abs(change_pct) > 5:
                score += 25
                reasons.append(f"🎯 Big move ({change_pct:+.1f}%)")
            elif abs(change_pct) > 3:
                score += 15
                reasons.append(f"↗️ Strong move ({change_pct:+.1f}%)")
            elif abs(change_pct) > 1:
                score += 5
            
            # Liquidity (20 points) - important for options
            if avg_volume > 10_000_000:
                score += 20
                reasons.append("💧 High liquidity")
            elif avg_volume > 5_000_000:
                score += 15
            elif avg_volume > 1_000_000:
                score += 10
            elif avg_volume > 500_000:
                score += 5  # Moderate liquidity still gets points
            else:
                score -= 5  # Smaller penalty for low liquidity
            
            # Volatility (15 points) - calculate from price swings
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            if volatility > 50:
                score += 15
                reasons.append(f"⚡ High volatility ({volatility:.0f}%)")
            elif volatility > 30:
                score += 10
            elif volatility > 20:
                score += 5  # Moderate volatility still acceptable
            elif volatility < 10:
                score -= 5  # Smaller penalty for very low volatility
            
            # Recent trend (10 points)
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
            if current_price > sma_20 * 1.02:
                score += 10
                reasons.append("📈 Above 20-SMA")
            elif current_price < sma_20 * 0.98:
                score += 5
                reasons.append("📉 Below 20-SMA")
            
            # Determine confidence and risk (adjusted for new base score of 40)
            if score >= 75:
                confidence = "VERY HIGH"
                risk = "M"
            elif score >= 60:
                confidence = "HIGH"
                risk = "M"
            elif score >= 45:
                confidence = "MEDIUM"
                risk = "M-H"
            else:
                confidence = "LOW"
                risk = "H"
            
            reason_text = " | ".join(reasons) if reasons else "Standard opportunity"
            
            return TopTrade(
                ticker=ticker,
                score=round(score, 1),
                price=round(current_price, 2),
                change_pct=round(change_pct, 2),
                volume=int(current_volume),
                volume_ratio=round(volume_ratio, 2),
                reason=reason_text,
                trade_type='options',
                confidence=confidence,
                risk_level=risk
            )
            
        except Exception as e:
            logger.error(f"Error analyzing options opportunity for {ticker}: {e}")
            return None
    
    def _detect_breakout_potential(self, hist: pd.DataFrame, current_price: float) -> Tuple[int, List[str]]:
        """
        Detect breakout potential using technical indicators
        Returns: (breakout_score, reasons)
        """
        score = 0
        reasons = []
        
        if len(hist) < 20:
            return score, reasons
        
        closes = hist['Close']
        volumes = hist['Volume']
        
        # 1. Consolidation breakout (20 points)
        recent_high = closes[-20:].max()
        recent_low = closes[-20:].min()
        consolidation_range = (recent_high - recent_low) / recent_low
        
        if consolidation_range < 0.10 and current_price >= recent_high * 0.98:
            score += 20
            reasons.append("💥 Consolidation breakout")
        elif consolidation_range < 0.15:
            score += 10
            reasons.append("🔹 Tight range, coiling")
        
        # 2. Volume surge (25 points)
        avg_volume_20 = volumes[-20:].mean()
        recent_volume_5 = volumes[-5:].mean()
        
        if recent_volume_5 > avg_volume_20 * 2:
            score += 25
            reasons.append(f"📨 Volume surge ({recent_volume_5/avg_volume_20:.1f}x)")
        elif recent_volume_5 > avg_volume_20 * 1.5:
            score += 15
            reasons.append("📈 Rising volume")
        
        # 3. Moving average breakthrough (20 points)
        ma_20 = closes[-20:].mean()
        ma_50 = closes[-50:].mean() if len(closes) >= 50 else ma_20
        
        if current_price > ma_20 and ma_20 > ma_50:
            score += 20
            reasons.append("⬆️ Above MA20 & MA50")
        elif current_price > ma_20:
            score += 10
            reasons.append("✅ Above MA20")
        
        # 4. Bollinger squeeze (15 points)
        std_20 = closes[-20:].std()
        bb_width = (std_20 * 2) / ma_20
        
        if bb_width < 0.10:  # Tight bands
            score += 15
            reasons.append("🔨 BB squeeze detected")
        
        # 5. Recent price momentum (15 points)
        if len(closes) >= 5:
            momentum_5d = (closes.iloc[-1] / closes.iloc[-5] - 1)
            if momentum_5d > 0.10:
                score += 15
                reasons.append(f"🚀 Strong 5D momentum (+{momentum_5d*100:.1f}%)")
            elif momentum_5d > 0.05:
                score += 10
        
        # 6. Support bounce (10 points)
        recent_lows = closes[-10:].nsmallest(3).mean()
        if abs(current_price - recent_lows) / recent_lows < 0.05:
            score += 10
            reasons.append("🟢 Near support level")
        
        return score, reasons
    
    def _analyze_penny_stock_opportunity(self, ticker: str) -> TopTrade:
        """Analyze a penny stock for trading potential with breakout detection"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get recent data
            hist = stock.history(period="3mo")
            if hist.empty or len(hist) < 10:
                return None
            
            info = stock.info
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price / prev_close - 1) * 100)
            
            # Filter: Only true penny stocks (<$5) and obscure plays
            # Prioritize lower prices with higher breakout potential
            if current_price > PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE:
                return None  # Skip higher-priced stocks
            
            # Bonus for ultra-low prices (more breakout room)
            ultra_low_bonus = get_price_tier_bonus(current_price)
            
            # Detect breakout potential
            breakout_score, breakout_reasons = self._detect_breakout_potential(hist, current_price)
            
            # Use penny stock analyzer
            result = self.penny_analyzer.analyze_stock(ticker)
            
            if 'error' in result:
                return None
            
            composite_score = result.get('composite_score', 0) + breakout_score + ultra_low_bonus
            price = result.get('price', 0)
            change_pct = result.get('pct_change', 0)
            volume = result.get('volume', 0)
            avg_volume = result.get('avg_volume', 1)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            # Build reason from scoring components
            reasons = []
            
            # Check for FDA/Healthcare catalyst (highest priority)
            is_healthcare = result.get('is_healthcare', False)
            healthcare_sector = result.get('healthcare_sector', '')
            fda_catalyst = result.get('fda_catalyst', '')
            
            if fda_catalyst:
                reasons.append(f"💊 FDA CATALYST: {fda_catalyst[:50]}")
            elif is_healthcare:
                reasons.append(f"🏥 Healthcare ({healthcare_sector})")
            
            momentum = result.get('momentum_score', 0)
            if momentum >= 70:
                reasons.append(f"🚀 Strong momentum ({momentum:.0f})")
            
            valuation = result.get('valuation_score', 0)
            if valuation >= 70:
                reasons.append(f"💎 Good value ({valuation:.0f})")
            
            catalyst = result.get('catalyst_score', 0)
            if catalyst >= 70:
                reasons.append(f"📰 Active catalyst ({catalyst:.0f})")
            
            if volume_ratio > 2:
                reasons.append(f"📊 Volume spike ({volume_ratio:.1f}x)")
            
            reason_text = " | ".join(reasons) if reasons else result.get('reasoning', 'Standard opportunity')
            
            # Combine reasoning
            base_reason = result.get('reasoning', 'Penny stock opportunity')
            if breakout_reasons:
                breakout_text = " | " + ", ".join(breakout_reasons)
                reason = base_reason + breakout_text
            else:
                reason = base_reason
            
            if ultra_low_bonus > 0:
                reason += f" | 💰 Ultra-low price (${current_price:.3f})"
            
            return TopTrade(
                ticker=ticker,
                score=composite_score,
                price=current_price,
                change_pct=result.get('pct_change', 0),
                volume=result.get('volume', 0),
                volume_ratio=result.get('volume', 0) / result.get('avg_volume', 1) if result.get('avg_volume', 0) > 0 else 0,
                reason=reason,
                trade_type='penny_stock',
                confidence=result.get('confidence_level', 'MEDIUM'),
                risk_level='HIGH' if breakout_score < 30 else 'MEDIUM-HIGH'
            )
            
        except Exception as e:
            logger.error(f"Error analyzing penny stock opportunity for {ticker}: {e}")
            return None
    
    def scan_custom_tickers(self, tickers: List[str], trade_type: str = 'options', top_n: int = 20) -> List[TopTrade]:
        """
        Scan a custom list of tickers
        
        Args:
            tickers: List of ticker symbols
            trade_type: 'options' or 'penny_stock'
            top_n: Number of top trades to return
        
        Returns:
            List of TopTrade objects
        """
        logger.info(f"Scanning {len(tickers)} custom tickers for {trade_type}...")
        
        results = []
        
        for ticker in tickers:
            try:
                if trade_type == 'options':
                    trade = self._analyze_options_opportunity(ticker)
                else:
                    trade = self._analyze_penny_stock_opportunity(ticker)
                
                if trade and trade.score > 0:
                    results.append(trade)
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_n]
    
    def get_quick_insights(self, trades: List[TopTrade]) -> Dict:
        """Generate quick insights from a list of trades"""
        if not trades:
            return {
                'total': 0,
                'avg_score': 0,
                'high_confidence': 0,
                'big_movers': 0,
                'volume_spikes': 0
            }
        
        return {
            'total': len(trades),
            'avg_score': round(sum(t.score for t in trades) / len(trades), 1),
            'high_confidence': len([t for t in trades if t.confidence in ['HIGH', 'VERY HIGH']]),
            'big_movers': len([t for t in trades if abs(t.change_pct) > 3]),
            'volume_spikes': len([t for t in trades if t.volume_ratio > 1.5]),
            'top_ticker': trades[0].ticker if trades else 'N/A',
            'top_score': trades[0].score if trades else 0
        }

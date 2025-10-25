"""Comprehensive stock analysis combining technical, news, and catalysts."""

import logging
from typing import Optional
from models.analysis import StockAnalysis
from analyzers.technical import TechnicalAnalyzer
from analyzers.news import NewsAnalyzer
from utils.caching import get_cached_stock_data

logger = logging.getLogger(__name__)


class ComprehensiveAnalyzer:
    """Combines all analysis into a complete stock evaluation"""
    
    @staticmethod
    def analyze_stock(ticker: str, trading_style: str = "OPTIONS") -> Optional[StockAnalysis]:
        """Perform complete stock analysis"""
        try:
            # Use cached data for better performance
            hist, info = get_cached_stock_data(ticker)
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price / prev_close - 1) * 100)
            
            # Technical indicators
            rsi = TechnicalAnalyzer.calculate_rsi(hist['Close'])
            macd_signal, macd_value = TechnicalAnalyzer.calculate_macd(hist['Close'])
            support, resistance = TechnicalAnalyzer.calculate_support_resistance(hist['Close'])
            iv_rank, iv_percentile = TechnicalAnalyzer.calculate_iv_metrics(ticker)

            # New indicators: EMA8/21, DeMarker, EMA context, Fibonacci targets
            ema8_series = TechnicalAnalyzer.ema(hist['Close'], 8)
            ema21_series = TechnicalAnalyzer.ema(hist['Close'], 21)
            dem_series = TechnicalAnalyzer.demarker(hist, period=14)
            ema_ctx = TechnicalAnalyzer.detect_ema_power_zone_and_reclaim(hist, ema8_series, ema21_series)
            fib_targets = TechnicalAnalyzer.compute_fib_extensions_from_swing(hist)
            
            # Multi-timeframe alignment and sector relative strength
            timeframe_alignment = TechnicalAnalyzer.analyze_timeframe_alignment(ticker)
            sector_rs = TechnicalAnalyzer.calculate_sector_relative_strength(ticker)
            
            # Volume analysis
            current_volume = int(hist['Volume'].iloc[-1])
            avg_volume = int(hist['Volume'].mean())
            
            # Determine trend (base on price change; reinforce with EMA power zone)
            if change_pct > 2:
                trend = "STRONG UPTREND"
            elif change_pct > 0.5:
                trend = "UPTREND"
            elif change_pct < -2:
                trend = "STRONG DOWNTREND"
            elif change_pct < -0.5:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"

            if ema_ctx.get("power_zone") and "DOWNTREND" not in trend:
                # Nudge neutral to uptrend when power zone is active
                if trend == "SIDEWAYS":
                    trend = "UPTREND"
            
            # News and catalysts
            news = NewsAnalyzer.get_stock_news(ticker)
            sentiment_score, sentiment_signals = NewsAnalyzer.analyze_sentiment(news)
            catalysts = NewsAnalyzer.get_catalysts(ticker)
            
            # Earnings information
            earnings_date = None
            earnings_days_away = None
            for catalyst in catalysts:
                if catalyst['type'] == 'Earnings Report':
                    earnings_date = catalyst['date']
                    earnings_days_away = catalyst['days_away']
                    break
            
            # Calculate confidence score (enhanced with timeframe and sector RS)
            confidence_score = ComprehensiveAnalyzer._calculate_confidence(
                rsi, macd_signal, iv_rank, sentiment_score, len(catalysts), earnings_days_away,
                timeframe_alignment=timeframe_alignment, sector_rs=sector_rs
            )
            
            # Generate recommendation based on trading style
            recommendation = ComprehensiveAnalyzer._generate_recommendation(
                rsi, macd_signal, trend, iv_rank, sentiment_score, earnings_days_away, trading_style,
                ema_ctx=ema_ctx, demarker_value=(dem_series.iloc[-1] if not dem_series.empty else None),
                fib_targets=fib_targets
            )
            
            return StockAnalysis(
                ticker=ticker.upper(),
                price=round(current_price, 2),
                change_pct=round(change_pct, 2),
                volume=current_volume,
                avg_volume=avg_volume,
                rsi=rsi,
                macd_signal=macd_signal,
                trend=trend,
                support=support,
                resistance=resistance,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                earnings_date=earnings_date,
                earnings_days_away=earnings_days_away,
                recent_news=news,
                catalysts=catalysts,
                sentiment_score=sentiment_score,
                sentiment_signals=sentiment_signals,
                confidence_score=confidence_score,
                recommendation=recommendation,
                ema8=float(ema8_series.iloc[-1]) if not ema8_series.empty else None,
                ema21=float(ema21_series.iloc[-1]) if not ema21_series.empty else None,
                demarker=float(dem_series.iloc[-1]) if not dem_series.empty else None,
                fib_targets=fib_targets,
                ema_power_zone=bool(ema_ctx.get("power_zone")),
                ema_reclaim=bool(ema_ctx.get("is_reclaim")),
                timeframe_alignment=timeframe_alignment,
                sector_rs=sector_rs
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    @staticmethod
    def _calculate_confidence(rsi: float, macd_signal: str, iv_rank: float, 
                            sentiment: float, catalyst_count: int, 
                            earnings_days: Optional[int],
                            timeframe_alignment: dict = None,
                            sector_rs: dict = None) -> float:
        """Calculate overall confidence score for trading this stock"""
        score = 50  # Base score
        
        # RSI contribution
        if 30 <= rsi <= 70:
            score += 15  # Neutral RSI is good
        elif rsi < 30:
            score += 10  # Oversold can be opportunity
        elif rsi > 70:
            score += 5   # Overbought is risky
        
        # MACD contribution
        if macd_signal in ["BULLISH", "BEARISH"]:
            score += 10
        
        # IV Rank contribution
        if iv_rank > 50:
            score += 15  # High IV is good for selling premium
        elif iv_rank < 30:
            score += 10  # Low IV is good for buying options
        
        # Sentiment contribution
        score += sentiment * 10  # -10 to +10
        
        # Catalyst contribution
        if catalyst_count > 0:
            score += min(catalyst_count * 5, 15)
        
        # Earnings risk
        if earnings_days is not None and earnings_days <= 7:
            score -= 20  # High risk around earnings
        
        # Timeframe alignment bonus (up to +10)
        if timeframe_alignment and timeframe_alignment.get("aligned"):
            alignment_score = timeframe_alignment.get("alignment_score", 0)
            score += min(10, alignment_score / 10)  # Up to +10 for perfect alignment
        
        # Sector relative strength bonus (up to +10)
        if sector_rs:
            rs_score = sector_rs.get("rs_score", 50)
            if rs_score > 60:
                score += min(10, (rs_score - 50) / 5)  # +2 for every 10 points above 50, max +10
            elif rs_score < 40:
                score -= min(10, (50 - rs_score) / 5)  # Penalty for underperformance
        
        return min(100, max(0, score))
    
    @staticmethod
    def _generate_recommendation(rsi: float, macd_signal: str, trend: str,
                                iv_rank: float, sentiment: float,
                                earnings_days: Optional[int], 
                                trading_style: str = "OPTIONS",
                                ema_ctx: dict | None = None,
                                demarker_value: float | None = None,
                                fib_targets: dict | None = None) -> str:
        """Generate trading recommendation based on selected trading style"""
        
        if trading_style == "DAY_TRADE":
            return ComprehensiveAnalyzer._generate_day_trade_recommendation(
                rsi, macd_signal, trend, sentiment, earnings_days, ema_ctx=ema_ctx
            )
        elif trading_style == "SWING_TRADE":
            return ComprehensiveAnalyzer._generate_swing_trade_recommendation(
                rsi, macd_signal, trend, sentiment, earnings_days,
                ema_ctx=ema_ctx, demarker_value=demarker_value, fib_targets=fib_targets
            )
        elif trading_style == "SCALP":
            return ComprehensiveAnalyzer._generate_scalp_recommendation(
                rsi, macd_signal, trend, sentiment
            )
        elif trading_style == "BUY_HOLD":
            return ComprehensiveAnalyzer._generate_buy_hold_recommendation(
                rsi, macd_signal, trend, sentiment, earnings_days
            )
        else:  # OPTIONS
            return ComprehensiveAnalyzer._generate_options_recommendation(
                rsi, macd_signal, trend, iv_rank, sentiment, earnings_days,
                ema_ctx=ema_ctx, demarker_value=demarker_value, fib_targets=fib_targets, current_price=None
            )
    
    @staticmethod
    def _generate_day_trade_recommendation(rsi: float, macd_signal: str, trend: str,
                                          sentiment: float, earnings_days: Optional[int],
                                          ema_ctx: dict | None = None) -> str:
        """Generate day trading recommendation for intraday equity trades"""
        recommendations = []
        
        # Trend-based entry
        if "UPTREND" in trend or "STRONG UPTREND" in trend:
            if rsi < 70:
                recommendations.append("📈 BUY on pullbacks to support levels")
                recommendations.append("Target: Resistance levels for quick profit (0.5-2%)")
            else:
                recommendations.append("⚠️ Overbought - Wait for pullback or avoid")
        elif "DOWNTREND" in trend or "STRONG DOWNTREND" in trend:
            if rsi > 30:
                recommendations.append("📉 SHORT on bounces to resistance")
                recommendations.append("Target: Support levels for quick profit (0.5-2%)")
            else:
                recommendations.append("⚠️ Oversold - Wait for bounce or avoid shorting")
        else:  # SIDEWAYS
            recommendations.append("↔️ Range-bound: BUY near support, SELL near resistance")
            recommendations.append("Use tight stops (0.3-0.5%) for range trading")
        
        # RSI signals
        if rsi < 30:
            recommendations.append("🟢 RSI oversold → Look for bounce/reversal entry")
        elif rsi > 70:
            recommendations.append("🔴 RSI overbought → Look for rejection/reversal short")

        # EMA Power Zone filter
        if ema_ctx and ema_ctx.get("power_zone"):
            recommendations.append("✅ 8>21 EMA Power Zone → Favor long setups")
        
        # MACD confirmation
        if macd_signal == "BULLISH":
            recommendations.append("✅ MACD bullish → Momentum favors longs")
        elif macd_signal == "BEARISH":
            recommendations.append("❌ MACD bearish → Momentum favors shorts")
        
        # Risk management
        recommendations.append("🛡️ Stop Loss: 0.5-1% | Take Profit: 1-3% | Exit by market close")
        
        # Earnings warning
        if earnings_days is not None and earnings_days <= 1:
            recommendations.append("⚠️ EARNINGS TODAY/TOMORROW → Avoid day trading (high volatility risk)")
        
        return "\n".join(recommendations) if recommendations else "Insufficient data for day trade recommendation"
    
    @staticmethod
    def _generate_swing_trade_recommendation(rsi: float, macd_signal: str, trend: str,
                                            sentiment: float, earnings_days: Optional[int],
                                            ema_ctx: dict | None = None,
                                            demarker_value: float | None = None,
                                            fib_targets: dict | None = None) -> str:
        """Generate swing trading recommendation for multi-day equity holds"""
        recommendations = []
        
        # Trend-based strategy
        if "UPTREND" in trend or "STRONG UPTREND" in trend:
            recommendations.append("📈 LONG BIAS: Enter on dips, hold for 3-10 days")
            recommendations.append("Entry: Near support or after consolidation breakout")
            recommendations.append("Target: 5-15% gain to resistance levels")
        elif "DOWNTREND" in trend or "STRONG DOWNTREND" in trend:
            recommendations.append("📉 SHORT BIAS: Enter on rallies, hold for 3-10 days")
            recommendations.append("Entry: Near resistance or after breakdown")
            recommendations.append("Target: 5-15% profit to support levels")
        else:
            recommendations.append("↔️ NEUTRAL: Wait for breakout direction before entering")
        
        # RSI for swing entries
        if rsi < 40 and "UPTREND" in trend:
            recommendations.append("🟢 Good swing entry: RSI pullback in uptrend")
        elif rsi > 60 and "DOWNTREND" in trend:
            recommendations.append("🔴 Good short entry: RSI bounce in downtrend")
        
        # MACD trend confirmation
        if macd_signal == "BULLISH":
            recommendations.append("✅ MACD confirms uptrend → Hold longs, avoid shorts")
        elif macd_signal == "BEARISH":
            recommendations.append("❌ MACD confirms downtrend → Hold shorts, avoid longs")
        
        # EMA power zone / reclaim context
        if ema_ctx:
            if ema_ctx.get("power_zone"):
                recommendations.append("✅ 8>21 EMA and price above both → Power Zone active")
            if ema_ctx.get("is_reclaim"):
                _reasons = ema_ctx.get("reasons", [])
                reasons_list = _reasons if isinstance(_reasons, list) else []
                reasons = "; ".join([str(x) for x in reasons_list])
                recommendations.append(f"✅ EMA Reclaim confirmed ({reasons})")

        # DeMarker precision entries
        if demarker_value is not None:
            if demarker_value <= 0.30 and ("UPTREND" in trend):
                recommendations.append("🟢 DeMarker ≤ 0.30 in uptrend → High-probability pullback entry")
            elif demarker_value >= 0.70 and ("DOWNTREND" in trend):
                recommendations.append("🔴 DeMarker ≥ 0.70 in downtrend → High-probability short entry")

        # Sentiment factor
        if sentiment > 0.3:
            recommendations.append("📰 Positive sentiment → Supports bullish swing trades")
        elif sentiment < -0.3:
            recommendations.append("📰 Negative sentiment → Supports bearish swing trades")
        
        # Fibonacci targets (if available)
        if fib_targets:
            t1 = fib_targets.get("T1_1272")
            t2 = fib_targets.get("T2_1618")
            t3 = fib_targets.get("T3_2618") or fib_targets.get("T3_200")
            fib_lines = []
            if t1:
                fib_lines.append(f"🎯 T1 (127.2%): ${t1:.2f} → Take 25%")
            if t2:
                fib_lines.append(f"🎯 T2 (161.8%): ${t2:.2f} → Take 50%")
            if t3:
                fib_lines.append(f"🎯 T3 (200-261.8%): ${t3:.2f} → Trail remaining")
            if fib_lines:
                recommendations.append("📐 Fibonacci Targets:")
                recommendations.extend(fib_lines)
                recommendations.append("🧭 Move stop to breakeven after T1, trail below 21 EMA thereafter")
        else:
            recommendations.append("🛡️ Stop Loss: 3-5% | Take Profit: 8-15% | Hold time: 3-10 days")
        
        # Earnings consideration
        if earnings_days is not None and earnings_days <= 7:
            recommendations.append("⚠️ EARNINGS SOON → Close position before earnings or use wider stops")
        
        return "\n".join(recommendations) if recommendations else "Insufficient data for swing trade recommendation"
    
    @staticmethod
    def _generate_scalp_recommendation(rsi: float, macd_signal: str, trend: str, sentiment: float) -> str:
        """Generate scalping recommendation for very short-term trades"""
        recommendations = []
        
        recommendations.append("⚡ SCALPING STRATEGY (seconds to minutes):")
        
        # Momentum-based scalping
        if "STRONG UPTREND" in trend:
            recommendations.append("🚀 Strong momentum UP → Scalp long on dips (0.1-0.5% targets)")
            recommendations.append("Entry: Quick pullbacks | Exit: Immediate resistance")
        elif "STRONG DOWNTREND" in trend:
            recommendations.append("💥 Strong momentum DOWN → Scalp short on bounces (0.1-0.5% targets)")
            recommendations.append("Entry: Quick bounces | Exit: Immediate support")
        else:
            recommendations.append("⚠️ Low momentum → Scalping difficult, wait for clear direction")
        
        # RSI for quick reversals
        if rsi < 25:
            recommendations.append("🟢 Extreme oversold → Quick bounce scalp opportunity")
        elif rsi > 75:
            recommendations.append("🔴 Extreme overbought → Quick rejection scalp opportunity")
        
        # Risk management for scalping
        recommendations.append("🛡️ TIGHT STOPS: 0.1-0.3% | Target: 0.2-0.5% | Hold: Seconds to 5 minutes")
        recommendations.append("⚡ Requires: Level 2 data, fast execution, high volume stocks")
        recommendations.append("⚠️ High risk: Only for experienced traders with proper tools")
        
        return "\n".join(recommendations)
    
    @staticmethod
    def _generate_buy_hold_recommendation(rsi: float, macd_signal: str, trend: str,
                                         sentiment: float, earnings_days: Optional[int]) -> str:
        """Generate buy and hold recommendation for long-term investing"""
        recommendations = []
        
        recommendations.append("📊 LONG-TERM INVESTMENT ANALYSIS:")
        
        # Overall trend assessment
        if "UPTREND" in trend or "STRONG UPTREND" in trend:
            recommendations.append("✅ Positive long-term trend → Good for accumulation")
            if rsi < 50:
                recommendations.append("🟢 STRONG BUY: Uptrend + pullback = ideal entry point")
            else:
                recommendations.append("🟡 BUY: Uptrend continues, consider dollar-cost averaging")
        elif "DOWNTREND" in trend or "STRONG DOWNTREND" in trend:
            recommendations.append("⚠️ Negative trend → Wait for reversal or avoid")
            recommendations.append("🔴 HOLD/AVOID: Downtrend not ideal for new positions")
        else:
            recommendations.append("🟡 NEUTRAL: Consolidating, wait for breakout direction")
        
        # Value assessment using RSI
        if rsi < 30:
            recommendations.append("💰 Potentially undervalued (oversold) → Good accumulation zone")
        elif rsi > 70:
            recommendations.append("💸 Potentially overvalued (overbought) → Consider waiting")
        
        # Sentiment for long-term
        if sentiment > 0.3:
            recommendations.append("📰 Strong positive sentiment → Supports long-term bullish case")
        elif sentiment < -0.3:
            recommendations.append("📰 Negative sentiment → Research fundamental concerns")
        
        # Long-term strategy
        recommendations.append("📈 Strategy: Dollar-cost average over time, ignore short-term noise")
        recommendations.append("🎯 Target: 20%+ annual returns | Hold time: 6+ months to years")
        recommendations.append("💡 Consider: Selling covered calls for income if you accumulate shares")
        
        # Earnings note
        if earnings_days is not None and earnings_days <= 14:
            recommendations.append("📅 Earnings soon → Good time to review fundamentals")
        
        return "\n".join(recommendations)
    
    @staticmethod
    def _generate_options_recommendation(rsi: float, macd_signal: str, trend: str,
                                        iv_rank: float, sentiment: float,
                                        earnings_days: Optional[int],
                                        ema_ctx: dict = None,
                                        demarker_value: float = None,
                                        fib_targets: dict = None,
                                        current_price: float = None) -> str:
        """Generate enhanced options trading recommendation with EMA/Fibonacci context"""
        recommendations = []
        
        # EMA Reclaim signal (high-confidence bullish setup)
        if ema_ctx and ema_ctx.get("is_reclaim") and iv_rank is not None:
            if iv_rank > 60:
                # High IV after reclaim: sell puts at key support
                recommendations.append("🔥 EMA RECLAIM + High IV → SELL CASH-SECURED PUTS or BULL PUT SPREAD")
                recommendations.append("   Strike: 5-10% OTM near 21 EMA | DTE: 30-45 days")
            else:
                # Low IV after reclaim: buy calls for directional move
                recommendations.append("🔥 EMA RECLAIM + Low IV → BUY CALL DEBIT SPREAD or LONG CALLS")
                recommendations.append("   Strike: ATM or 1 strike ITM | DTE: 45-60 days (allow time for move)")
        
        # EMA Power Zone (favorable for bullish structures)
        elif ema_ctx and ema_ctx.get("power_zone"):
            recommendations.append("✅ Power Zone Active → Favor bullish structures")
        
        # DeMarker pullback entries for options
        if demarker_value is not None and demarker_value <= 0.30 and "UPTREND" in trend:
            recommendations.append("🎯 DeMarker oversold in uptrend → Time call spreads for bounce")
            recommendations.append("   Ideal for: 0-14 DTE scalps or 30-45 DTE swing calls")
        
        # Fibonacci-based strike and DTE selection
        if fib_targets and current_price:
            t1 = fib_targets.get("T1_1272")
            t2 = fib_targets.get("T2_1618")
            if t1 and t2:
                # Suggest strikes based on Fibonacci targets
                recommendations.append(f"📐 Fibonacci Targets detected:")
                recommendations.append(f"   T1: ${t1:.2f} | T2: ${t2:.2f}")
                if "UPTREND" in trend:
                    recommendations.append(f"   → Long call spread: Buy ATM, Sell @ T1 strike (${t1:.2f})")
                    recommendations.append(f"   → DTE: 30-60 days (time to reach T1/T2)")
                else:
                    recommendations.append(f"   → Use T1/T2 as resistance for put credit spreads or iron condors")
        
        # Standard IV-based strategies
        if iv_rank > 60:
            recommendations.append("📊 High IV (>60) → SELL PREMIUM strategies preferred")
            recommendations.append("   - Cash-secured puts, covered calls, iron condors, credit spreads")
            recommendations.append("   - DTE: 30-45 days (collect theta, avoid extreme gamma risk)")
        elif iv_rank < 40:
            recommendations.append("📊 Low IV (<40) → BUY OPTIONS strategies preferred")
            recommendations.append("   - Long calls/puts, debit spreads")
            recommendations.append("   - DTE: 60-90 days (give time for move, less theta decay)")
        else:
            recommendations.append("📊 Moderate IV → Balanced approach, use defined-risk spreads")
        
        # Trend-based refined strategies
        if "UPTREND" in trend:
            if iv_rank > 50:
                recommendations.append("📈 Uptrend + High IV → Bull put spreads (sell puts 10-15% OTM)")
            else:
                recommendations.append("📈 Uptrend + Low IV → Bull call spreads or long calls (ATM to 1 ITM)")
        elif "DOWNTREND" in trend:
            if iv_rank > 50:
                recommendations.append("📉 Downtrend + High IV → Bear call spreads (sell calls 10-15% OTM)")
            else:
                recommendations.append("📉 Downtrend + Low IV → Bear put spreads or long puts")
        else:
            if iv_rank > 50:
                recommendations.append("↔️ Sideways + High IV → Iron condors or short strangles")
                recommendations.append("   - Sell wings outside expected range (~1 SD)")
        
        # RSI-based entries
        if rsi < 30 and iv_rank < 50:
            recommendations.append("🟢 RSI oversold + Low IV → Buy calls for reversal (30-45 DTE)")
        elif rsi > 70 and iv_rank < 50:
            recommendations.append("🔴 RSI overbought + Low IV → Buy puts for reversal (30-45 DTE)")
        
        # Earnings warning with strategy adjustments
        if earnings_days is not None and earnings_days <= 7:
            recommendations.append("⚠️ EARNINGS IN <7 DAYS:")
            recommendations.append("   - HIGH RISK: IV crush after earnings will hurt long options")
            recommendations.append("   - Consider: Defined-risk spreads, or wait until after earnings")
            recommendations.append("   - Earnings play: Sell premium (iron condors) if IV is elevated")
        elif earnings_days is not None and 7 < earnings_days <= 21:
            recommendations.append("📅 Earnings in 1-3 weeks → Monitor IV expansion, consider pre-earnings premium selling")
        
        # Sentiment overlay
        if sentiment > 0.3:
            recommendations.append("📰 Positive sentiment → Supports bullish strategies")
        elif sentiment < -0.3:
            recommendations.append("📰 Negative sentiment → Supports bearish strategies")
        
        return "\n".join(recommendations) if recommendations else "Insufficient data for options recommendation"

"""Tests that a comprehensive analysis on SOFI completes and all expected attributes are present."""

def test_sofi_attributes_are_accessible():
    """
    Tests that a comprehensive analysis on SOFI completes and all
    expected attributes (including Phase 2) are present and accessible.
    """
    from analyzers.comprehensive import ComprehensiveAnalyzer

    analysis = ComprehensiveAnalyzer.analyze_stock("SOFI", "SWING_TRADE")

    assert analysis is not None, "Analysis should not be None"

    # Verify that all expected attributes exist
    expected_attributes = [
        'ticker', 'price', 'change_pct', 'volume', 'avg_volume', 'rsi',
        'macd_signal', 'trend', 'support', 'resistance', 'iv_rank',
        'iv_percentile', 'earnings_date', 'earnings_days_away',
        'recent_news', 'catalysts', 'sentiment_score', 'sentiment_signals',
        'confidence_score', 'recommendation', 'ema8', 'ema21', 'demarker',
        'fib_targets', 'ema_power_zone', 'ema_reclaim',
        'timeframe_alignment', 'sector_rs'
    ]

    missing_attrs = [attr for attr in expected_attributes if not hasattr(analysis, attr)]

    assert not missing_attrs, f"Missing attributes in analysis object: {', '.join(missing_attrs)}"
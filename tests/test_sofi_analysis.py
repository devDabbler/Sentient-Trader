"""Test that a basic analysis of SOFI completes successfully."""

from analyzers.comprehensive import ComprehensiveAnalyzer

def test_sofi_analysis_completes():
    """Tests that a basic analysis of SOFI completes successfully."""
    analysis = ComprehensiveAnalyzer.analyze_stock("SOFI", "SWING_TRADE")

    assert analysis is not None, "Analysis should not be None"
    assert analysis.ticker == "SOFI"
    assert isinstance(analysis.price, float)
    assert isinstance(analysis.confidence_score, float)
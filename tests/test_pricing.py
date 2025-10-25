import math
from options_pricing import black_scholes_price, binomial_american_price, greeks_finite_difference


def test_black_scholes_zero_time():
    # At expiry price equals intrinsic
    S, K = 100.0, 90.0
    price_call = black_scholes_price(S, K, 0.0, 0.01, 0.2, is_call=True)
    assert math.isclose(price_call, S - K, rel_tol=1e-9)


def test_binomial_american_zero_time():
    S, K = 50.0, 55.0
    price_put = binomial_american_price(S, K, 0.0, 0.01, 0.25, is_call=False)
    assert math.isclose(price_put, max(0.0, K - S), rel_tol=1e-9)


def test_american_vs_european_put_early_exercise():
    # For deep ITM American put, price should be >= BS European (early exercise value)
    S, K = 40.0, 100.0
    T = 0.5
    r = 0.01
    sigma = 0.25
    eur = black_scholes_price(S, K, T, r, sigma, is_call=False)
    am = binomial_american_price(S, K, T, r, sigma, is_call=False, steps=300)
    assert am >= eur - 1e-8


def test_greeks_finite_difference_basic():
    S, K = 100.0, 100.0
    T = 30 / 365.0
    r = 0.01
    sigma = 0.2
    greeks = greeks_finite_difference(black_scholes_price, S, K, T, r, sigma, is_call=True)
    # Sanity checks
    assert 'delta' in greeks and 'vega' in greeks and 'theta' in greeks
    assert abs(greeks['delta']) <= 1.0


def test_low_vol_pricing_near_intrinsic():
    # Extremely low vol should approach intrinsic value (plus small time value)
    S, K = 100.0, 95.0
    T = 10 / 365.0
    r = 0.01
    sigma = 1e-6
    eur = black_scholes_price(S, K, T, r, sigma, is_call=True)
    assert eur >= max(0.0, S - K) - 1e-6


def test_long_dte_behavior():
    # Very long DTE should still return finite numbers and greeks computable
    S, K = 100.0, 100.0
    T = 20.0  # 20 years
    r = 0.02
    sigma = 0.3
    price = black_scholes_price(S, K, T, r, sigma, is_call=True)
    greeks = greeks_finite_difference(black_scholes_price, S, K, T, r, sigma, is_call=True)
    assert price > 0.0
    assert 'delta' in greeks and 'vega' in greeks


def test_binomial_many_steps_convergence():
    # Check that increasing binomial steps stabilizes the price (convergence)
    S, K = 100.0, 100.0
    T = 1.0
    r = 0.01
    sigma = 0.2
    p1 = binomial_american_price(S, K, T, r, sigma, is_call=True, steps=50)
    p2 = binomial_american_price(S, K, T, r, sigma, is_call=True, steps=1000)
    # Prices should not differ wildly; allow small tolerance
    assert abs(p2 - p1) < 1.0


def test_vega_units_per_vol_point():
    # Verify vega scales approximately with a 1% bump
    S, K = 100.0, 100.0
    T = 30 / 365.0
    r = 0.01
    sigma = 0.2
    greeks = greeks_finite_difference(black_scholes_price, S, K, T, r, sigma, is_call=True)
    # numerical check: bump sigma by 1% (0.01) and check price delta approx equals vega reported
    base = black_scholes_price(S, K, T, r, sigma, is_call=True)
    bumped = black_scholes_price(S, K, T, r, sigma + 0.01, is_call=True)
    observed = bumped - base
    # greeks['vega'] is per 1 vol point
    assert abs(observed - greeks['vega']) < 1.0  # allow loose tolerance depending on option

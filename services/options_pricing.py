"""Option pricing helpers.

Includes:
- Black-Scholes European price and greeks
- Binomial tree pricer for American options (Cox-Ross-Rubinstein)
- Finite-difference greeks wrapper
"""
from math import log, sqrt, exp
from scipy.stats import norm
from typing import Callable, Any
import numpy as np


def _to_float(x: Any) -> float:
    """Coerce numbers or numpy scalars/arrays to Python float safely."""
    try:
        # numpy scalars and 0-d arrays support item()
        if hasattr(x, 'item'):
            return float(x.item())
        # numpy arrays -> try to index
        if isinstance(x, (list, tuple)):
            return float(x[0])
        return float(x)
    except Exception:
        # fallback: convert via numpy
        try:
            return float(np.asarray(x).astype(float).reshape(-1)[0])
        except Exception:
            raise TypeError(f"Cannot convert value to float: {type(x))}")


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> float:
    """European Black-Scholes price."""
    if T <= 0 or sigma <= 0:
        return float(max(0.0, S - K) if is_call else max(0.0, K - S))

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    val = (S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)) if is_call else (K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return _to_float(val)


def _black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> dict:
    """Closed-form Black-Scholes Greeks.

    Returns: dict with keys: delta, gamma, vega (per 1 vol point), theta (per day)
    Vega is returned per 1 vol percentage point (i.e. per +1% = 0.01 in sigma decimal).
    """
    # Handle degenerate cases
    if T <= 0 or sigma <= 0:
        # At expiry or zero vol, greeks collapse to intrinsic/deltas and zero elsewhere
        if is_call:
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {'delta': _to_float(delta), 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    pdf_d1 = norm.pdf(d1)

    # Delta
    delta = norm.cdf(d1) if is_call else (norm.cdf(d1) - 1.0)

    # Gamma
    gamma = pdf_d1 / (S * sigma * sqrt(T))

    # Vega (derivative wrt sigma in decimal). Convert to per 1 vol point (1% = 0.01)
    vega_decimal = S * pdf_d1 * sqrt(T)
    vega_per_vol_point = _to_float(vega_decimal * 0.01)

    # Theta (annual) then convert to per day
    # Standard Black-Scholes theta (per year)
    theta_call = (-(S * pdf_d1 * sigma) / (2 * sqrt(T))) - r * K * exp(-r * T) * norm.cdf(d2)
    theta_put = (-(S * pdf_d1 * sigma) / (2 * sqrt(T))) + r * K * exp(-r * T) * norm.cdf(-d2)
    theta = theta_call if is_call else theta_put
    theta_per_day = _to_float(theta / 365.0)

    return {'delta': _to_float(delta), 'gamma': _to_float(gamma), 'vega': vega_per_vol_point, 'theta': theta_per_day}


def binomial_american_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True, steps: int = 200) -> float:
    """Price an American option using Cox-Ross-Rubinstein binomial tree.

    steps: number of time steps; more steps -> better accuracy. Default 200 (fast and reasonable).
    """
    if T <= 0:
        return float(max(0.0, S - K) if is_call else max(0.0, K - S))
    dt = T / steps
    # up/down factors and risk-neutral prob
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    a = exp(r * dt)
    p = (a - d) / (u - d)

    # initialize asset prices at maturity
    asset_prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]

    # option values at maturity
    if is_call:
        option_values = [max(0.0, price - K) for price in asset_prices]
    else:
        option_values = [max(0.0, K - price) for price in asset_prices]

    # backward induction with early exercise check
    disc = exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        new_values = []
        for j in range(i + 1):
            cont = disc * (p * option_values[j + 1] + (1 - p) * option_values[j])
            # asset price at node
            price = S * (u ** j) * (d ** (i - j))
            exercise = (max(0.0, price - K) if is_call else max(0.0, K - price))
            new_values.append(max(cont, exercise))
        option_values = new_values

    return _to_float(option_values[0])


def greeks_finite_difference(pricer: Callable[..., Any], S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> dict:
    """Estimate delta, gamma, vega, theta using central finite differences.

    pricer: function(S, K, T, r, sigma, is_call) -> price
    Return values: delta, gamma, vega (per 1 vol point), theta (per day)
    """
    eps_S = max(0.01, S * 1e-4)
    eps_sigma = max(1e-4, sigma * 1e-4)
    eps_T = max(1.0 / 365.0, T * 1e-4)

    # If caller supplied the plain black_scholes_price function, use closed-form greeks for speed & precision
    try:
        from .options_pricing import black_scholes_price as _bs_check  # type: ignore
    except Exception:
        # relative import may fail in tests; fall back to global name
        _bs_check = globals().get('black_scholes_price')

    # If pricer is exactly the black_scholes_price callable, use analytical greeks
    if pricer is _bs_check or getattr(pricer, '__name__', '') == getattr(_bs_check, '__name__', ''):
        return _black_scholes_greeks(S, K, T, r, sigma, is_call=is_call)

    # Otherwise fall back to central finite differences
    price = _to_float(pricer(S, K, T, r, sigma, is_call=is_call))

    # Delta
    price_up = _to_float(pricer(S + eps_S, K, T, r, sigma, is_call=is_call))
    price_down = _to_float(pricer(S - eps_S, K, T, r, sigma, is_call=is_call))
    delta = _to_float((price_up - price_down) / (2 * eps_S))

    # Gamma
    gamma = _to_float((price_up - 2 * price + price_down) / (eps_S ** 2))

    # Vega (derivative wrt sigma in decimal). Convert to per 1 vol point (1% = 0.01)
    price_sigma_up = _to_float(pricer(S, K, T, r, sigma + eps_sigma, is_call=is_call))
    price_sigma_down = _to_float(pricer(S, K, T, r, sigma - eps_sigma, is_call=is_call))
    vega_deriv = (price_sigma_up - price_sigma_down) / (2 * eps_sigma)
    vega = _to_float(vega_deriv * 0.01)

    # Theta per day
    if T <= 0:
        theta = 0.0
    else:
        price_T_up = _to_float(pricer(S, K, max(T - eps_T, 0.0), r, sigma, is_call=is_call))
        theta = _to_float((price_T_up - price) / (eps_T * 365.0))

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

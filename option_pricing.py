"""
Option Pricing Suite v2.0 - PRODUCTION GRADE
--------------------------------------------------------------------------------
Advanced option pricing models with Greeks and sensitivity analysis:

1) Black-Scholes-Merton (BSM): European options + Greeks (Delta, Gamma, Vega, Theta, Rho)
2) Monte Carlo Simulation: European + American (LSM) with confidence intervals
3) Binomial Trees (CRR): European/American with early exercise
4) Trinomial Trees: Enhanced lattice approach
5) Implied Volatility Calculator: Newton-Raphson + Bisection fallback

Integrated with data_fetcher.py for live market data.

Dependencies:
    pip install numpy scipy pandas yfinance scikit-learn
"""

from __future__ import annotations

import math
import numpy as np
import warnings
import logging
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import brentq, newton
from datetime import datetime
from typing import Tuple, Dict, Optional

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OptionPricing")
warnings.filterwarnings('ignore')

# =============================================================================
# UTILITIES
# =============================================================================

def _validate_inputs(S: float, K: float, r: float, T: float, sigma: float) -> None:
    """Validates option pricing inputs."""
    if S <= 0 or K <= 0:
        raise ValueError(f"S ({S}) and K ({K}) must be > 0")
    if T <= 0:
        raise ValueError(f"T ({T}) must be > 0")
    if sigma <= 0:
        raise ValueError(f"sigma ({sigma}) must be > 0")
    if not (0 <= r <= 1):
        logger.warning(f"⚠️  Risk-free rate r={r:.2%} seems unusual (expected 0-100%)")


def annualized_vol_from_prices(
    prices: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Estimates annualized volatility from price series using log returns.
    """
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if prices.size < 3:
        raise ValueError("Need at least 3 prices to estimate volatility")

    log_returns = np.diff(np.log(prices))
    vol_daily = np.std(log_returns, ddof=1)
    vol_annual = float(vol_daily * math.sqrt(periods_per_year))

    return vol_annual


# =============================================================================
# DATA FETCHER ADAPTER
# =============================================================================

def load_market_inputs_from_datafetcher(
    ticker: str,
    expiry_date: Optional[str] = None,
    strike: Optional[float] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Fetches live market data using FinancialDataFetcher.
    """
    logger.info(f"🔍 Loading market inputs for {ticker}...")

    try:
        from data_fetcher import FinancialDataFetcher
    except ImportError:
        raise ImportError(
            "❌ Could not import 'FinancialDataFetcher' from 'data_fetcher.py'. "
        )

    fetcher = FinancialDataFetcher(ticker=ticker)

    logger.info(f"📊 Fetching live quote for {ticker}...")
    quote = fetcher.fetch_live_quote()

    if not quote or quote.get('price') is None or quote.get('price') <= 0:
        logger.warning("⚠️  Live quote unavailable, fetching recent history...")
        hist_df, _ = fetcher.fetch_ohlcv(ticker, period="5d")
        if not hist_df.empty:
            S = float(hist_df['Close'].iloc[-1])
        else:
            raise ValueError(
                f"❌ CRITICAL: Could not fetch price data for {ticker}. "
            )
    else:
        S = float(quote['price'])

    if S <= 0:
        raise ValueError(f"❌ Invalid spot price: ${S:.2f}")

    logger.info(f"✅ Spot Price (S): ${S:.2f}")

    logger.info(f"📈 Fetching option chain for {ticker}...")
    chain_data = fetcher.fetch_options_chain(expiry_date=expiry_date)

    expiry_str = chain_data.get('expiry')
    if not expiry_str:
        raise ValueError(f"❌ No option expiry dates found for {ticker}.")

    try:
        exp_dt = datetime.strptime(expiry_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"❌ Invalid expiry date format: {expiry_str}")

    days_to_exp = (exp_dt - datetime.now()).days

    if days_to_exp < 1:
        T = 1.0 / 365.0
        logger.warning(f"⚠️  Expiry is very close ({days_to_exp} days). Using T={T:.4f} years")
    else:
        T = float(days_to_exp) / 365.0

    logger.info(f"✅ Expiry: {expiry_str} (T={T:.4f} years, ~{days_to_exp} days)")

    calls = chain_data.get('calls', [])
    if not calls:
        raise ValueError(f"❌ No call options found for {ticker} at expiry {expiry_str}")

    if strike is not None:
        K = float(strike)
        logger.info(f"✅ Using provided strike (K): ${K:.2f}")
    else:
        closest_opt = min(calls, key=lambda x: abs(x['strike'] - S))
        K = float(closest_opt['strike'])
        logger.info(f"✅ Auto-selected ATM strike (K): ${K:.2f}")

    sigma = 0.0
    contract = next((c for c in calls if abs(c['strike'] - K) < 0.01), None)

    if contract and contract.get('impliedVolatility', 0) > 0:
        sigma = float(contract['impliedVolatility'])
        logger.info(f"✅ Implied Volatility (IV): {sigma:.2%}")
    else:
        logger.warning("⚠️  IV not found. Calculating Historical Volatility...")
        df, _ = fetcher.fetch_ohlcv(ticker, period="1y")
        if not df.empty:
            sigma = annualized_vol_from_prices(df['Close'].values)
            logger.info(f"✅ Historical Volatility (1Y): {sigma:.2%}")
        else:
            raise ValueError("❌ Could not calculate volatility")

    if sigma <= 0:
        raise ValueError(f"❌ Invalid volatility: {sigma:.2%}")

    # Fallback to realistic vol limits if Yahoo Finance returns extreme IV (like 400%+)
    # Large IV breaks CRR p computation
    if sigma > 2.0:
        logger.warning(f"⚠️  Volatility seems too high ({sigma:.2%}). Capping at 200%.")
        sigma = 2.0

    r = 0.045
    logger.info(f"✅ Risk-free rate (r): {r:.2%} (fixed)")

    try:
        fundamentals = fetcher.fetch_fundamentals()
        if fundamentals and isinstance(fundamentals, dict):
            q = float(fundamentals.get('dividend_yield', 0) or 0.0)
            logger.info(f"✅ Dividend Yield (q): {q:.2%}")
        else:
            q = 0.0
            logger.warning("⚠️  No dividend data found. Assuming q=0%")
    except Exception as e:
        logger.warning(f"⚠️  Could not fetch fundamentals ({e}). Assuming q=0%")
        q = 0.0

    return S, K, r, T, sigma, q


# =============================================================================
# 1) BLACK-SCHOLES-MERTON (BSM) + GREEKS
# =============================================================================

@dataclass(frozen=True)
class BSMGreeks:
    """Container for Black-Scholes option price and Greeks."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def __str__(self) -> str:
        return (
            f"Price: ${self.price:.4f}\n"
            f"Delta: {self.delta:>8.4f}  Gamma: {self.gamma:>8.6f}\n"
            f"Vega:  {self.vega:>8.4f}  Theta: {self.theta:>8.4f}\n"
            f"Rho:   {self.rho:>8.4f}"
        )


def bsm_d1_d2(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    q: float = 0.0
) -> Tuple[float, float]:
    """Calculates d1 and d2 for Black-Scholes formula."""
    _validate_inputs(S, K, r, T, sigma)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0
) -> float:
    """Black-Scholes European option pricing formula."""
    option_type = option_type.lower()
    d1, d2 = bsm_d1_d2(S, K, r, T, sigma, q)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option_type == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_greeks(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0
) -> BSMGreeks:
    """Calculates Black-Scholes Greeks."""
    option_type = option_type.lower()
    d1, d2 = bsm_d1_d2(S, K, r, T, sigma, q)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    sqrt_T = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    price = black_scholes_price(S, K, r, T, sigma, option_type, q)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)

    gamma = (disc_q * pdf_d1) / (S * sigma * sqrt_T)
    vega = S * disc_q * pdf_d1 * sqrt_T / 100

    if option_type == "call":
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        ) / 365
    else:
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        ) / 365

    if option_type == "call":
        rho = K * T * disc_r * norm.cdf(d2) / 100
    else:
        rho = -K * T * disc_r * norm.cdf(-d2) / 100

    return BSMGreeks(
        price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho
    )


# =============================================================================
# 2) IMPLIED VOLATILITY (IV) CALCULATOR
# =============================================================================

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """Calculates implied volatility using Newton-Raphson with Brent's method fallback."""
    option_type = option_type.lower()

    def objective(sigma: float) -> float:
        try:
            return black_scholes_price(S, K, r, T, sigma, option_type, q) - market_price
        except:
            return 1e10

    def vega_func(sigma: float) -> float:
        try:
            d1, _ = bsm_d1_d2(S, K, r, T, sigma, q)
            return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
        except:
            return 1e-10

    sigma = 0.3
    for _ in range(max_iterations):
        price_error = objective(sigma)
        if abs(price_error) < tolerance: return sigma
        vega = vega_func(sigma)
        if abs(vega) < 1e-10: break
        sigma_new = sigma - price_error / vega
        sigma_new = max(0.001, min(sigma_new, 5.0))
        if abs(sigma_new - sigma) < tolerance: return sigma_new
        sigma = sigma_new

    logger.warning("⚠️  Newton-Raphson failed, using Brent's method...")
    try:
        return brentq(objective, 0.001, 5.0, xtol=tolerance, maxiter=100)
    except:
        logger.error("❌ IV calculation failed. Returning NaN.")
        return float('nan')


# =============================================================================
# 3) MONTE CARLO SIMULATION
# =============================================================================

def simulate_gbm_paths(
    S0: float, r: float, T: float, sigma: float, steps: int, n_paths: int, q: float = 0.0, seed: Optional[int] = 42
) -> np.ndarray:
    _validate_inputs(S0, 1.0, r, T, sigma)
    if steps <= 0 or n_paths <= 0: raise ValueError("steps and n_paths must be > 0")

    rng = np.random.default_rng(seed)
    dt = T / steps
    Z = rng.standard_normal((n_paths, steps))
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt) * Z
    increments = drift + diffusion

    paths = np.empty((n_paths, steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))
    return paths

def mc_european_option_price(
    S0: float, K: float, r: float, T: float, sigma: float, option_type: str = "call", q: float = 0.0, steps: int = 252, n_paths: int = 100_000, seed: Optional[int] = 42, ci_level: float = 0.95
) -> Dict:
    option_type = option_type.lower()
    paths = simulate_gbm_paths(S0, r, T, sigma, steps, n_paths, q, seed)
    ST = paths[:, -1]
    if option_type == "call": payoffs = np.maximum(ST - K, 0.0)
    elif option_type == "put": payoffs = np.maximum(K - ST, 0.0)
    else: raise ValueError("option_type must be 'call' or 'put'")

    disc = math.exp(-r * T)
    discounted_payoffs = disc * payoffs
    price = float(discounted_payoffs.mean())
    std = float(discounted_payoffs.std(ddof=1))
    se = std / math.sqrt(n_paths)

    alpha = 1.0 - ci_level
    z_score = norm.ppf(1.0 - alpha / 2.0)
    return { "price": price, "ci": (price - z_score * se, price + z_score * se), "std_error": se, "std_dev": std, "payoffs": payoffs }

def _poly_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    n = x.shape[0]
    X = np.ones((n, degree + 1), dtype=float)
    for d in range(1, degree + 1): X[:, d] = x**d
    return X

def american_option_lsm(
    S0: float, K: float, r: float, T: float, sigma: float, option_type: str = "put", q: float = 0.0, steps: int = 100, n_paths: int = 100_000, basis_degree: int = 3, seed: Optional[int] = 42
) -> float:
    option_type = option_type.lower()
    paths = simulate_gbm_paths(S0, r, T, sigma, steps, n_paths, q, seed)
    dt = T / steps
    disc = math.exp(-r * dt)

    if option_type == "put": intrinsic = np.maximum(K - paths, 0.0)
    elif option_type == "call": intrinsic = np.maximum(paths - K, 0.0)
    else: raise ValueError("option_type must be 'call' or 'put'")

    cashflow = intrinsic[:, -1].copy()

    for t in range(steps - 1, 0, -1):
        cashflow *= disc
        itm = intrinsic[:, t] > 0.0
        if not np.any(itm): continue

        S_t = paths[itm, t]
        Y = cashflow[itm]
        X = _poly_basis(S_t, degree=basis_degree)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        continuation_value = X @ beta
        exercise_value = intrinsic[itm, t]
        exercise_now = exercise_value > continuation_value

        idx_itm = np.where(itm)[0]
        exercise_idx = idx_itm[exercise_now]
        cashflow[exercise_idx] = exercise_value[exercise_now]

    return float(cashflow.mean() * disc)


# =============================================================================
# 5) BINOMIAL TREE (CRR - Cox-Ross-Rubinstein)
# =============================================================================

def crr_binomial_price(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    steps: int = 200,
    option_type: str = "call",
    american: bool = False,
    q: float = 0.0
) -> float:
    """
    Cox-Ross-Rubinstein Binomial Tree option pricing.
    """
    option_type = option_type.lower()
    _validate_inputs(S, K, r, T, sigma)

    if steps <= 0:
        raise ValueError("steps must be > 0")

    dt = T / steps

    # Check tree stability condition
    # To ensure 0 < p < 1, sigma * sqrt(dt) must be > |(r-q)dt|
    # If not, CRR fails because the up/down movements don't cover the drift.
    drift = (r - q) * dt
    vol = sigma * math.sqrt(dt)

    if abs(drift) >= vol:
        # If stability fails, dynamically increase steps to satisfy condition
        min_steps = math.ceil(((r - q) / sigma)**2 * T)
        # Add a buffer of 10% more steps than the minimum required
        steps = max(steps, int(min_steps * 1.1) + 1)
        logger.warning(f"⚠️ Binomial tree unstable. Auto-increased steps to {steps}.")
        dt = T / steps

    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    growth = math.exp((r - q) * dt)

    p = (growth - d) / (u - d)

    # Final sanity check, fallback to risk neutral generic p
    if not (0.0 < p < 1.0):
        # Extremely small T or extreme parameters can still cause float precision issues
        logger.warning(f"⚠️ Fallback triggered: computed p={p:.4f} is invalid. Clamping to [0.001, 0.999].")
        p = max(0.001, min(0.999, p))

    ST = np.array([S * (u**(steps - i)) * (d**i) for i in range(steps + 1)])

    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    for n in range(steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if american:
            S_n = np.array([S * (u**(n - i)) * (d**i) for i in range(n + 1)])
            intrinsic = np.maximum(S_n - K, 0.0) if option_type == "call" else np.maximum(K - S_n, 0.0)
            V = np.maximum(V, intrinsic)

    return float(V[0])


# =============================================================================
# 6) TRINOMIAL TREE
# =============================================================================

def trinomial_tree_price(
    S: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    steps: int = 100,
    option_type: str = "call",
    q: float = 0.0
) -> float:
    option_type = option_type.lower()
    _validate_inputs(S, K, r, T, sigma)

    if steps <= 0: raise ValueError("steps must be > 0")

    dt = T / steps
    disc = math.exp(-r * dt) # Fix: Was math.exp(-r * T)

    # Trinomial stability check
    # Trinomial stability requires dt < sigma^2 / (r-q)^2
    if (r-q) != 0 and dt >= (sigma**2 / (r-q)**2):
        min_steps = math.ceil(((r-q)**2 * T) / sigma**2)
        steps = max(steps, min_steps + 10)
        dt = T / steps
        disc = math.exp(-r * dt)

    u = math.exp(sigma * math.sqrt(2.0 * dt))
    d = 1.0 / u

    a = math.exp((r - q) * dt / 2.0)
    b = math.exp(sigma * math.sqrt(dt / 2.0))
    c = 1.0 / b

    pu = ((a - c) / (b - c)) ** 2
    pd = ((b - a) / (b - c)) ** 2
    pm = 1.0 - pu - pd

    # Fallback for numerical instability
    if pu < 0 or pd < 0 or pm < 0:
        logger.warning(f"⚠️ Trinomial probabilities invalid. Clamping probabilities.")
        pu = max(0.001, min(0.999, pu))
        pd = max(0.001, min(0.999, pd))
        pm = max(0.001, 1.0 - pu - pd)

    j = np.arange(-steps, steps + 1)
    ST = S * (u ** np.maximum(j, 0)) * (d ** np.maximum(-j, 0))

    if option_type == "call": V = np.maximum(ST - K, 0.0)
    elif option_type == "put": V = np.maximum(K - ST, 0.0)
    else: raise ValueError("option_type must be 'call' or 'put'")

    for n in range(steps - 1, -1, -1):
        V_next = V
        V = np.empty(2 * n + 1, dtype=float)
        for i in range(2 * n + 1):
            V[i] = disc * (pu * V_next[i + 2] + pm * V_next[i + 1] + pd * V_next[i])

    return float(V[0])
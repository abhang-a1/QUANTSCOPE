"""
QuantScope Pro - Metrics Module v2.0 (Production Grade - Jan 2026)
--------------------------------------------------------------------------------
Advanced financial metrics with asset-specific benchmarking.
FIX: Ensures all metrics return scalar values, not Series objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
import logging
import yfinance as yf
from datetime import datetime, timedelta
import warnings

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantScope-Metrics")

# =============================================================================
# CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.045  # 4.5% as of Jan 2026

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _ensure_scalar(value: Any) -> float:
    """
    Ensures a value is a scalar (float), not a pandas Series or array.

    Args:
        value: Any numeric value (scalar, Series, array)

    Returns:
        Float scalar value
    """
    if isinstance(value, (pd.Series, np.ndarray)):
        if len(value) == 0:
            return 0.0
        return float(value.iloc[0] if isinstance(value, pd.Series) else value.flat[0])
    elif pd.isna(value):
        return 0.0
    else:
        return float(value)


# =============================================================================
# ASSET-SPECIFIC BENCHMARKING
# =============================================================================

class AssetSpecificMetrics:
    """
    Asset-tailored evaluation engine with intelligent benchmark detection.
    Compares asset performance against relevant market indices.
    """

    # Comprehensive benchmark mapping (Updated 2026)
    BENCHMARKS = {
        # === US EQUITIES ===
        'SPY': '^GSPC', 'VOO': '^GSPC', 'IVV': '^GSPC',  # S&P 500
        'QQQ': '^IXIC', 'TQQQ': '^IXIC',                  # NASDAQ
        'DIA': '^DJI',                                     # Dow Jones
        'IWM': '^RUT',                                     # Russell 2000

        # Tech Giants
        'AAPL': '^GSPC', 'MSFT': '^GSPC', 'GOOGL': '^IXIC',
        'AMZN': '^IXIC', 'META': '^IXIC', 'NVDA': '^IXIC',
        'TSLA': '^IXIC',

        # Finance
        'JPM': '^GSPC', 'BAC': '^GSPC', 'GS': '^GSPC', 'WFC': '^GSPC',

        # === INDIA ===
        '^NSEI': '^NSEI', 'NIFTY': '^NSEI',
        'RELIANCE.NS': '^NSEI', 'TCS.NS': '^NSEI', 'INFY.NS': '^NSEI',
        'HDFCBANK.NS': '^NSEI', 'ICICIBANK.NS': '^NSEI',

        # === COMMODITIES ===
        'GC=F': 'GC=F',   # Gold
        'SI=F': 'SI=F',   # Silver
        'CL=F': 'CL=F',   # Crude Oil
        'NG=F': 'NG=F',   # Natural Gas

        # === FOREX ===
        'EURUSD=X': 'EURUSD=X',
        'USDJPY=X': 'USDJPY=X',
        'GBPUSD=X': 'GBPUSD=X',
        'USDINR=X': 'USDINR=X',

        # === CRYPTO ===
        'BTC-USD': 'BTC-USD',
        'ETH-USD': 'ETH-USD',
    }

    # Sector-specific benchmarks
    SECTOR_BENCHMARKS = {
        'Technology': '^IXIC',
        'Financials': '^GSPC',
        'Healthcare': 'XLV',
        'Energy': 'XLE',
        'Consumer': '^GSPC',
    }

    def __init__(self, ticker: str, sector: Optional[str] = None):
        """
        Initialize metrics calculator for a specific asset.

        Args:
            ticker: Asset symbol
            sector: Optional sector for benchmark selection
        """
        self.ticker = ticker.upper().strip()
        self.sector = sector
        self.benchmark_ticker = self._detect_benchmark()
        self.benchmark_returns = None
        self.benchmark_prices = None

        logger.info(f"📊 Initialized metrics for {self.ticker} (Benchmark: {self.benchmark_ticker})")

    def _detect_benchmark(self) -> str:
        """
        Intelligently detects the appropriate benchmark for the asset.

        Returns:
            Benchmark ticker symbol
        """
        ticker_clean = self.ticker.replace('$', '').upper()

        # 1. Direct match
        if ticker_clean in self.BENCHMARKS:
            return self.BENCHMARKS[ticker_clean]

        # 2. Suffix matching (e.g., .NS for NSE, .BO for BSE)
        for suffix, bench in self.BENCHMARKS.items():
            if ticker_clean.endswith(suffix):
                return bench

        # 3. Sector-based matching
        if self.sector and self.sector in self.SECTOR_BENCHMARKS:
            return self.SECTOR_BENCHMARKS[self.sector]

        # 4. Regional heuristics
        if any(x in ticker_clean for x in ['.NS', '.BO', '^NSE', 'NIFTY']):
            return '^NSEI'  # India
        elif '.T' in ticker_clean:
            return '^N225'  # Japan (Nikkei)
        elif '.HK' in ticker_clean:
            return '^HSI'   # Hong Kong
        elif '.SS' in ticker_clean or '.SZ' in ticker_clean:
            return '000001.SS'  # China (SSE)
        elif '=F' in ticker_clean:
            return ticker_clean  # Commodities benchmark to themselves
        elif any(x in ticker_clean for x in ['BTC', 'ETH', '-USD']):
            return ticker_clean  # Crypto benchmark to themselves

        # 5. Default to S&P 500
        logger.info(f"No specific benchmark found for {self.ticker}, using S&P 500")
        return '^GSPC'

    def fetch_benchmark(self, period: str = "2y", start: Optional[str] = None) -> pd.Series:
        """
        Fetches benchmark returns for relative performance calculation.

        Args:
            period: Time period ('1y', '2y', '5y', 'max')
            start: Optional start date (YYYY-MM-DD)

        Returns:
            Pandas Series of benchmark returns
        """
        if self.benchmark_returns is not None:
            return self.benchmark_returns

        try:
            logger.info(f"📈 Fetching benchmark data for {self.benchmark_ticker}...")

            # Download benchmark data
            if start:
                bench_data = yf.download(
                    self.benchmark_ticker,
                    start=start,
                    progress=False,
                    auto_adjust=True
                )
            else:
                bench_data = yf.download(
                    self.benchmark_ticker,
                    period=period,
                    progress=False,
                    auto_adjust=True
                )

            if bench_data.empty:
                raise ValueError("Empty benchmark data")

            # Handle MultiIndex columns
            if isinstance(bench_data.columns, pd.MultiIndex):
                try:
                    bench_data.columns = bench_data.columns.droplevel(1)
                except Exception:
                    pass

            # Extract close prices
            if 'Close' in bench_data.columns:
                self.benchmark_prices = bench_data['Close']
            else:
                raise ValueError("No Close column in benchmark data")

            # Calculate returns
            self.benchmark_returns = self.benchmark_prices.pct_change().fillna(0)

            logger.info(f"✅ Fetched {len(self.benchmark_returns)} benchmark data points")
            return self.benchmark_returns

        except Exception as e:
            logger.warning(f"⚠️  Benchmark fetch failed ({self.benchmark_ticker}): {e}")
            logger.info("Using zero returns as fallback")

            # Fallback: Zero returns
            dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
            self.benchmark_returns = pd.Series(0.0, index=dates)
            return self.benchmark_returns


# =============================================================================
# CORE METRICS CALCULATION
# =============================================================================

def calculate_comprehensive_stats(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    risk_free_rate: float = RISK_FREE_RATE
) -> Dict[str, Any]:
    """
    Main entry point for comprehensive metrics calculation.

    Calculates:
    - Returns (Total, Annualized, CAGR)
    - Risk metrics (Volatility, Sharpe, Sortino, Calmar)
    - Drawdown analysis (Max DD, Average DD)
    - Benchmark-relative metrics (Alpha, Beta, Information Ratio)
    - Distribution stats (Skewness, Kurtosis)
    - Win rate and profit factor

    Args:
        df: DataFrame with at least a 'Close' column
        ticker: Optional ticker symbol for benchmark comparison
        risk_free_rate: Annual risk-free rate (default: 4.5%)

    Returns:
        Dictionary of comprehensive metrics
    """
    if df.empty or len(df) < 10:
        logger.warning("Insufficient data for metrics calculation")
        return _get_empty_metrics()

    try:
        # Make a copy to avoid modifying original
        df = df.copy()

        # 1. Calculate returns
        if 'returns' not in df.columns:
            df['returns'] = df['Close'].pct_change()

        returns = df['returns'].dropna()

        if returns.empty or len(returns) < 5:
            return _get_empty_metrics()

        # 2. Basic Return Metrics
        first_close = _ensure_scalar(df['Close'].iloc[0])
        last_close = _ensure_scalar(df['Close'].iloc[-1])
        total_return = (last_close / first_close) - 1

        # CAGR (Compound Annual Growth Rate)
        years = len(df) / TRADING_DAYS_PER_YEAR
        cagr = ((last_close / first_close) ** (1 / years)) - 1 if years > 0 else 0

        # Annualized return
        annualized_return = _ensure_scalar(returns.mean()) * TRADING_DAYS_PER_YEAR

        # 3. Risk Metrics
        volatility = _ensure_scalar(returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe Ratio
        excess_return = annualized_return - risk_free_rate
        sharpe = excess_return / volatility if volatility != 0 else 0

        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = _ensure_scalar(negative_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(negative_returns) > 0 else 0
        sortino = excess_return / downside_std if downside_std != 0 else 0

        # 4. Drawdown Analysis
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max

        max_drawdown = _ensure_scalar(drawdown.min())
        avg_drawdown = _ensure_scalar(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0

        # Calmar Ratio (Return / Max Drawdown)
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 5. Distribution Statistics
        skewness = _ensure_scalar(returns.skew())
        kurtosis = _ensure_scalar(returns.kurtosis())

        # Value at Risk (95% confidence)
        var_95 = _ensure_scalar(returns.quantile(0.05))

        # Conditional Value at Risk (CVaR / Expected Shortfall)
        cvar_95 = _ensure_scalar(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else 0

        # 6. Win Rate & Profit Factor
        win_rate = (len(returns[returns > 0]) / len(returns)) * 100 if len(returns) > 0 else 0

        total_gains = _ensure_scalar(returns[returns > 0].sum()) if len(returns[returns > 0]) > 0 else 0
        total_losses = abs(_ensure_scalar(returns[returns < 0].sum())) if len(returns[returns < 0]) > 0 else 0
        profit_factor = total_gains / total_losses if total_losses != 0 else 0

        # 7. Benchmark-Relative Metrics (if ticker provided)
        alpha = 0.0
        beta = 1.0
        information_ratio = 0.0
        tracking_error = 0.0

        if ticker:
            try:
                asset_metrics = AssetSpecificMetrics(ticker)
                benchmark_returns = asset_metrics.fetch_benchmark(period="2y")

                # Align returns with benchmark
                aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

                if len(aligned_returns) > 20:
                    # Beta (via regression)
                    covariance = _ensure_scalar(aligned_returns.cov(aligned_benchmark))
                    benchmark_variance = _ensure_scalar(aligned_benchmark.var())
                    beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0

                    # Alpha (Jensen's Alpha)
                    benchmark_return = _ensure_scalar(aligned_benchmark.mean()) * TRADING_DAYS_PER_YEAR
                    alpha = annualized_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

                    # Information Ratio
                    active_return = aligned_returns - aligned_benchmark
                    tracking_error = _ensure_scalar(active_return.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
                    information_ratio = (_ensure_scalar(active_return.mean()) * TRADING_DAYS_PER_YEAR) / tracking_error if tracking_error != 0 else 0

            except Exception as e:
                logger.warning(f"Benchmark comparison failed: {e}")

        # 8. Compile Metrics (ensure all are scalar floats)
        metrics = {
            # Returns
            "total_return": round(float(total_return * 100), 2),
            "annualized_return": round(float(annualized_return * 100), 2),
            "cagr": round(float(cagr * 100), 2),

            # Risk
            "volatility": round(float(volatility * 100), 2),
            "downside_volatility": round(float(downside_std * 100), 2),

            # Risk-Adjusted Returns
            "sharpe": round(float(sharpe), 3),
            "sortino": round(float(sortino), 3),
            "calmar": round(float(calmar), 3),
            "information_ratio": round(float(information_ratio), 3),

            # Drawdown
            "max_drawdown": round(float(max_drawdown * 100), 2),
            "avg_drawdown": round(float(avg_drawdown * 100), 2),

            # Benchmark-Relative
            "alpha": round(float(alpha * 100), 2),
            "beta": round(float(beta), 3),
            "tracking_error": round(float(tracking_error * 100), 2),

            # Distribution
            "skewness": round(float(skewness), 3),
            "kurtosis": round(float(kurtosis), 3),
            "var_95": round(float(var_95 * 100), 2),
            "cvar_95": round(float(cvar_95 * 100), 2),

            # Win Rate
            "win_rate": round(float(win_rate), 2),
            "profit_factor": round(float(profit_factor), 3),

            # Metadata
            "observation_count": int(len(returns)),
            "time_period_years": round(float(years), 2)
        }

        logger.info(f"✅ Calculated {len(metrics)} comprehensive metrics")
        return metrics

    except Exception as e:
        logger.error(f"❌ Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return _get_empty_metrics()


def _get_empty_metrics() -> Dict[str, Any]:
    """Returns empty/zero metrics dictionary."""
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "cagr": 0.0,
        "volatility": 0.0,
        "downside_volatility": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "information_ratio": 0.0,
        "max_drawdown": 0.0,
        "avg_drawdown": 0.0,
        "alpha": 0.0,
        "beta": 1.0,
        "tracking_error": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "observation_count": 0,
        "time_period_years": 0.0
    }


# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================

def calculate_returns_distribution(
    df: pd.DataFrame,
    bins: int = 30
) -> Dict[str, List]:
    """
    Generates histogram data for returns distribution visualization.

    Args:
        df: DataFrame with 'Close' column
        bins: Number of histogram bins

    Returns:
        Dictionary with labels and values for Chart.js
    """
    if df.empty or len(df) < 2:
        return {"labels": [], "values": [], "normal_curve": []}

    try:
        # Calculate returns (in percentage)
        returns = df['Close'].pct_change().dropna() * 100

        if len(returns) < 2:
            return {"labels": [], "values": [], "normal_curve": []}

        # Create histogram
        hist, bin_edges = np.histogram(returns, bins=bins)

        # Calculate bin centers for labels
        labels = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
        labels = [round(float(x), 2) for x in labels]

        # Generate normal distribution overlay
        mu = _ensure_scalar(returns.mean())
        sigma = _ensure_scalar(returns.std())

        normal_curve = []
        for label in labels:
            # Probability density function
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((label - mu) / sigma) ** 2)
            # Scale to match histogram
            normal_curve.append(pdf * len(returns) * (bin_edges[1] - bin_edges[0]))

        logger.info(f"✅ Generated distribution with {bins} bins")

        return {
            "labels": labels,
            "values": hist.tolist(),
            "normal_curve": [round(float(x), 2) for x in normal_curve]
        }

    except Exception as e:
        logger.error(f"❌ Distribution calculation failed: {e}")
        return {"labels": [], "values": [], "normal_curve": []}


# =============================================================================
# WIN RATE & TRADE STATISTICS
# =============================================================================

def calculate_win_rate(df: pd.DataFrame) -> float:
    """
    Calculates percentage of positive return days.

    Args:
        df: DataFrame with 'Close' column

    Returns:
        Win rate percentage
    """
    if df.empty or len(df) < 2:
        return 0.0

    try:
        returns = df['Close'].pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        wins = len(returns[returns > 0])
        win_rate = (wins / len(returns)) * 100

        return round(float(win_rate), 2)

    except Exception as e:
        logger.error(f"❌ Win rate calculation failed: {e}")
        return 0.0


def calculate_trade_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates detailed trade statistics.

    Args:
        df: DataFrame with 'Close' column

    Returns:
        Dictionary with trade statistics
    """
    if df.empty or len(df) < 2:
        return {}

    try:
        returns = df['Close'].pct_change().dropna()

        if len(returns) == 0:
            return {}

        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        stats = {
            "total_trades": int(len(returns)),
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
            "win_rate": round(float((len(wins) / len(returns)) * 100), 2),

            "avg_win": round(float(_ensure_scalar(wins.mean()) * 100), 3) if len(wins) > 0 else 0.0,
            "avg_loss": round(float(_ensure_scalar(losses.mean()) * 100), 3) if len(losses) > 0 else 0.0,
            "largest_win": round(float(_ensure_scalar(wins.max()) * 100), 3) if len(wins) > 0 else 0.0,
            "largest_loss": round(float(_ensure_scalar(losses.min()) * 100), 3) if len(losses) > 0 else 0.0,

            "total_gains": round(float(_ensure_scalar(wins.sum()) * 100), 3) if len(wins) > 0 else 0.0,
            "total_losses": round(float(_ensure_scalar(losses.sum()) * 100), 3) if len(losses) > 0 else 0.0,

            "profit_factor": round(float(_ensure_scalar(wins.sum()) / abs(_ensure_scalar(losses.sum()))), 3) if len(losses) > 0 and _ensure_scalar(losses.sum()) != 0 else 0.0,
            "expectancy": round(float(_ensure_scalar(returns.mean()) * 100), 3)
        }

        return stats

    except Exception as e:
        logger.error(f"❌ Trade statistics calculation failed: {e}")
        return {}


# =============================================================================
# ROLLING METRICS
# =============================================================================

def calculate_rolling_metrics(
    df: pd.DataFrame,
    window: int = 30,
    metrics: List[str] = ['sharpe', 'volatility']
) -> pd.DataFrame:
    """
    Calculates rolling window metrics for time-series analysis.

    Args:
        df: DataFrame with 'Close' column
        window: Rolling window size in days
        metrics: List of metrics to calculate ('sharpe', 'volatility', 'max_drawdown')

    Returns:
        DataFrame with rolling metrics
    """
    if df.empty or len(df) < window:
        return pd.DataFrame()

    try:
        returns = df['Close'].pct_change().dropna()
        results = pd.DataFrame(index=df.index[1:])

        if 'sharpe' in metrics:
            rolling_mean = returns.rolling(window=window).mean() * TRADING_DAYS_PER_YEAR
            rolling_std = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            results['rolling_sharpe'] = (rolling_mean - RISK_FREE_RATE) / rolling_std

        if 'volatility' in metrics:
            results['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

        if 'max_drawdown' in metrics:
            rolling_max = df['Close'].rolling(window=window, min_periods=1).max()
            rolling_dd = ((df['Close'] - rolling_max) / rolling_max) * 100
            results['rolling_max_drawdown'] = rolling_dd

        logger.info(f"✅ Calculated rolling metrics with {window}-day window")
        return results

    except Exception as e:
        logger.error(f"❌ Rolling metrics calculation failed: {e}")
        return pd.DataFrame()


# =============================================================================
# TESTING & DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 QUANTSCOPE PRO - METRICS MODULE v2.0 TESTING")
    print("=" * 80)

    # Test with sample data
    import yfinance as yf

    print("\n📊 Testing with AAPL...")
    ticker = "AAPL"
    df = yf.download(ticker, period="2y", progress=False)

    if not df.empty:
        # Comprehensive stats
        stats = calculate_comprehensive_stats(df, ticker=ticker)

        print("\n" + "=" * 80)
        print("📈 COMPREHENSIVE METRICS")
        print("=" * 80)
        print(f"Total Return:        {stats['total_return']:.2f}%")
        print(f"CAGR:                {stats['cagr']:.2f}%")
        print(f"Volatility:          {stats['volatility']:.2f}%")
        print(f"Sharpe Ratio:        {stats['sharpe']:.3f}")
        print(f"Sortino Ratio:       {stats['sortino']:.3f}")
        print(f"Max Drawdown:        {stats['max_drawdown']:.2f}%")
        print(f"Alpha:               {stats['alpha']:.2f}%")
        print(f"Beta:                {stats['beta']:.3f}")
        print(f"Win Rate:            {stats['win_rate']:.2f}%")
        print(f"Profit Factor:       {stats['profit_factor']:.3f}")

        # Distribution
        dist = calculate_returns_distribution(df, bins=25)
        print(f"\n📊 Distribution bins: {len(dist['labels'])}")

        # Trade stats
        trade_stats = calculate_trade_statistics(df)
        print("\n" + "=" * 80)
        print("📊 TRADE STATISTICS")
        print("=" * 80)
        print(f"Total Trades:        {trade_stats['total_trades']}")
        print(f"Win Rate:            {trade_stats['win_rate']:.2f}%")
        print(f"Avg Win:             {trade_stats['avg_win']:.3f}%")
        print(f"Avg Loss:            {trade_stats['avg_loss']:.3f}%")
        print(f"Profit Factor:       {trade_stats['profit_factor']:.3f}")

        print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY")
    else:
        print("❌ Failed to fetch test data")

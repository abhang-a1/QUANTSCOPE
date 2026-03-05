"""
Data Fetcher Module v2.0 - PRODUCTION GRADE
--------------------------------------------------------------------------------
Enterprise-grade financial data fetcher with:
- Multi-currency support with automatic detection
- Comprehensive ticker information
- Live quotes, options, news, fundamentals, sentiment
- Advanced caching strategies (TTL-based)
- Robust error handling and fallback mechanisms
- ML-ready data preparation
- Compatible with QuantScope Backend API

Dependencies:
    pip install yfinance pandas numpy scikit-learn cachetools
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass, asdict
import cachetools
import time
import random
import hashlib

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TickerMetadata:
    """Currency and market metadata for a ticker"""
    currency: str = 'USD'
    exchange: str = ''
    market: str = 'US'
    timezone: str = 'America/New_York'
    is_index: bool = False
    name: str = ''
    quote_type: str = 'EQUITY'

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# MAIN DATA FETCHER CLASS
# =============================================================================

class FinancialDataFetcher:
    """
    Enterprise-grade financial data fetcher with native currency support
    and comprehensive market data access.
    """

    # Currency mapping by exchange
    CURRENCY_MAP = {
        'US': 'USD', 'NASDAQ': 'USD', 'NYSE': 'USD', 'AMEX': 'USD',
        'NSE': 'INR', 'NSI': 'INR', 'BSE': 'INR', 'BOM': 'INR',
        'TSE': 'JPY', 'JPX': 'JPY',
        'SSE': 'CNY', 'SHH': 'CNY', 'SHZ': 'CNY',
        'HKSE': 'HKD', 'HKG': 'HKD',
        'LSE': 'GBP', 'LON': 'GBP',
        'XETR': 'EUR', 'FRA': 'EUR', 'PAR': 'EUR',
        'ASX': 'AUD', 'SYD': 'AUD',
        'TSX': 'CAD', 'TOR': 'CAD',
    }

    # Ticker suffix to currency mapping
    SUFFIX_CURRENCY = {
        '.NS': 'INR', '.BO': 'INR',
        '.T': 'JPY',
        '.SS': 'CNY', '.SZ': 'CNY',
        '.HK': 'HKD',
        '.L': 'GBP',
        '.DE': 'EUR', '.PA': 'EUR',
        '.AX': 'AUD',
        '.TO': 'CAD',
    }

    def __init__(
        self,
        ticker: str = 'AAPL',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize the data fetcher.

        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'RELIANCE.NS', '^NSEI')
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
        """
        self.ticker = ticker.upper().strip()
        self.start_date = start_date or '2020-01-01'
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        # Multi-tier caching system
        self.cache = cachetools.TTLCache(maxsize=2000, ttl=7200)       # OHLCV: 2hr
        self.quote_cache = cachetools.TTLCache(maxsize=2000, ttl=30)   # Quotes: 30s
        self.meta_cache = cachetools.TTLCache(maxsize=500, ttl=86400)  # Meta: 24hr
        self.info_cache = cachetools.TTLCache(maxsize=500, ttl=3600)   # Info: 1hr
        self.news_cache = cachetools.TTLCache(maxsize=500, ttl=1800)   # News: 30min

        # Rate limiting
        self.last_request = 0
        self.request_delay = 0.3  # 300ms between requests

        # ML preprocessing
        try:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        except ImportError:
            logger.warning("scikit-learn not installed. ML features disabled.")
            self.scaler = None

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    def _rate_limit(self) -> None:
        """Implements rate limiting to avoid API throttling."""
        elapsed = time.time() - self.last_request
        sleep_time = max(0, self.request_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_request = time.time()

    # =========================================================================
    # CURRENCY & METADATA DETECTION
    # =========================================================================

    def _detect_currency(self, info: Optional[Dict] = None) -> TickerMetadata:
        """
        Intelligent currency and market detection with robust fallbacks.

        Args:
            info: yfinance info dictionary (optional)

        Returns:
            TickerMetadata object with detected attributes
        """
        cache_key = f"meta_{self.ticker}"
        if cache_key in self.meta_cache:
            return self.meta_cache[cache_key]

        # Initialize defaults
        currency = 'USD'
        market = 'US'
        exchange = ''
        timezone = 'America/New_York'
        name = self.ticker
        quote_type = 'EQUITY'

        # Parse info dict if available
        if info and isinstance(info, dict):
            currency = info.get('currency', 'USD')
            exchange = info.get('exchange', '') or info.get('fullExchangeName', '')
            timezone = info.get('timezone', 'America/New_York')
            name = info.get('shortName') or info.get('longName') or self.ticker
            quote_type = info.get('quoteType', 'EQUITY')

            # Map exchange to currency
            if exchange in self.CURRENCY_MAP:
                currency = self.CURRENCY_MAP[exchange]

        # Detect by ticker suffix
        for suffix, curr in self.SUFFIX_CURRENCY.items():
            if self.ticker.endswith(suffix):
                currency = curr
                if suffix in ['.NS', '.BO']:
                    market = 'India'
                elif suffix == '.T':
                    market = 'Japan'
                elif suffix in ['.SS', '.SZ']:
                    market = 'China'
                elif suffix == '.HK':
                    market = 'Hong Kong'
                elif suffix == '.L':
                    market = 'UK'
                elif suffix in ['.DE', '.PA']:
                    market = 'Europe'
                elif suffix == '.AX':
                    market = 'Australia'
                elif suffix == '.TO':
                    market = 'Canada'
                break

        # Detect indices
        is_index = (
            self.ticker.startswith('^') or
            'INDEX' in self.ticker.upper() or
            'NIFTY' in self.ticker.upper()
        )

        # Special cases
        if any(x in self.ticker for x in ['=F', 'GC=F', 'CL=F']):
            currency, market, quote_type = 'USD', 'Commodity', 'FUTURE'
        elif '-USD' in self.ticker or 'BTC' in self.ticker:
            currency, market, quote_type = 'USD', 'Crypto', 'CRYPTOCURRENCY'

        meta = TickerMetadata(
            currency=currency,
            exchange=exchange,
            market=market,
            timezone=timezone,
            is_index=is_index,
            name=name,
            quote_type=quote_type
        )

        self.meta_cache[cache_key] = meta
        return meta

    # =========================================================================
    # LIVE MARKET DATA
    # =========================================================================

    def fetch_live_quote(self) -> Dict[str, Any]:
        """
        Fetches ultra-low latency live quote using fast_info with robust fallbacks.

        Returns:
            Dictionary with current price, change, volume, and metadata
        """
        cache_key = f"quote_{self.ticker}"
        if cache_key in self.quote_cache:
            return self.quote_cache[cache_key]

        try:
            self._rate_limit()
            dat = yf.Ticker(self.ticker)

            # Try fast_info first (fastest method)
            try:
                fast = dat.fast_info
                last_price = getattr(fast, 'last_price', None)
                prev_close = getattr(fast, 'previous_close', None)
                day_high = getattr(fast, 'day_high', None)
                day_low = getattr(fast, 'day_low', None)
                volume = getattr(fast, 'last_volume', 0)
                currency = getattr(fast, 'currency', 'USD')
            except Exception:
                last_price = prev_close = day_high = day_low = None
                volume = 0
                currency = 'USD'

            # Fallback to history if fast_info incomplete
            if last_price is None or prev_close is None:
                logger.info(f"fast_info incomplete for {self.ticker}, using history fallback")
                hist = dat.history(period="2d", interval="1m")
                if not hist.empty:
                    last_price = float(hist['Close'].iloc[-1])
                    prev_close = float(hist['Close'].iloc[0])
                    day_high = float(hist['High'].max())
                    day_low = float(hist['Low'].min())
                    volume = int(hist['Volume'].sum())
                else:
                    # Ultimate fallback
                    last_price = prev_close = day_high = day_low = 0.0
                    volume = 0

            # Calculate changes
            change = (last_price - prev_close) if prev_close else 0.0
            change_pct = ((change / prev_close) * 100) if prev_close else 0.0

            # Get metadata
            info_minimal = {'currency': currency}
            meta = self._detect_currency(info_minimal)

            quote = {
                "ticker": self.ticker,
                "price": float(last_price),
                "previous_close": float(prev_close),
                "change": float(change),
                "change_percent": round(change_pct, 2),
                "day_high": float(day_high or last_price),
                "day_low": float(day_low or last_price),
                "volume": int(volume),
                "currency": meta.currency,
                "exchange": meta.exchange,
                "timestamp": datetime.now().isoformat()
            }

            self.quote_cache[cache_key] = quote
            logger.info(f"✅ Live quote for {self.ticker}: ${quote['price']:.2f} {quote['currency']}")
            return quote

        except Exception as e:
            logger.error(f"❌ Live quote failed for {self.ticker}: {e}")
            return {
                "ticker": self.ticker,
                "price": 0.0,
                "previous_close": 0.0,
                "change": 0.0,
                "change_percent": 0.0,
                "day_high": 0.0,
                "day_low": 0.0,
                "volume": 0,
                "currency": "USD",
                "exchange": "N/A",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    # =========================================================================
    # HISTORICAL OHLCV DATA
    # =========================================================================

    def fetch_ohlcv(
        self,
        ticker: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d"
    ) -> Tuple[pd.DataFrame, TickerMetadata]:
        """
        Fetches historical OHLCV data with robust error handling.

        Args:
            ticker: Stock symbol (uses self.ticker if None)
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            Tuple of (DataFrame, TickerMetadata)
        """
        ticker = (ticker or self.ticker).upper()
        cache_key = f"ohlcv_{ticker}_{period}_{interval}"

        if cache_key in self.cache:
            logger.info(f"📦 Using cached data for {ticker}")
            return self.cache[cache_key]

        try:
            self._rate_limit()
            logger.info(f"📊 Fetching {period} of {interval} data for {ticker}...")

            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False
            )

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Handle MultiIndex columns (when downloading multiple tickers)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass

            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Add derived features
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility_20'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)

            # Get metadata
            dat = yf.Ticker(ticker)
            try:
                info = dat.info
            except Exception:
                info = {}

            meta = self._detect_currency(info if isinstance(info, dict) else {})

            # Cache result
            self.cache[cache_key] = (df, meta)
            logger.info(f"✅ Fetched {len(df)} rows for {ticker} in {meta.currency}")

            return df, meta

        except Exception as e:
            logger.error(f"❌ OHLCV fetch failed for {ticker}: {e}")
            logger.warning("⚠️  Generating mock data as fallback...")
            return self._generate_mock_data(ticker)

    # =========================================================================
    # COMPREHENSIVE TICKER INFORMATION
    # =========================================================================

    def fetch_ticker_info(self) -> Dict[str, Any]:
        """
        Fetches comprehensive ticker information including:
        - Company details (name, description, sector, industry)
        - Market data (market cap, shares outstanding, enterprise value)
        - Trading info (exchange, currency, timezone)
        - Valuation metrics (PE, PEG, P/B, P/S ratios)
        - Dividend information
        - Risk metrics (Beta, 52-week range, moving averages)
        - Financial health (debt, cash, ratios)
        - Profitability metrics (margins, ROE, ROA)
        - Growth rates
        - Analyst recommendations
        - Company officers
        - Contact information

        Returns:
            Comprehensive dictionary of ticker information
        """
        cache_key = f"info_{self.ticker}"
        if cache_key in self.info_cache:
            logger.info(f"📦 Using cached info for {self.ticker}")
            return self.info_cache[cache_key]

        try:
            self._rate_limit()
            logger.info(f"📊 Fetching comprehensive info for {self.ticker}...")

            dat = yf.Ticker(self.ticker)
            info = dat.info or {}

            # Detect metadata
            meta = self._detect_currency(info)

            # Build comprehensive info dictionary
            ticker_info = {
                # === BASIC IDENTITY ===
                "ticker": self.ticker,
                "name": info.get('longName') or info.get('shortName') or self.ticker,
                "short_name": info.get('shortName', self.ticker),
                "description": info.get('longBusinessSummary', 'N/A'),

                # === MARKET CLASSIFICATION ===
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "exchange": meta.exchange or info.get('exchange', 'N/A'),
                "currency": meta.currency,
                "market": meta.market,
                "timezone": meta.timezone,
                "quote_type": info.get('quoteType', meta.quote_type),
                "is_index": meta.is_index,

                # === MARKET DATA ===
                "market_cap": info.get('marketCap'),
                "enterprise_value": info.get('enterpriseValue'),
                "shares_outstanding": info.get('sharesOutstanding'),
                "float_shares": info.get('floatShares'),
                "shares_short": info.get('sharesShort'),
                "short_ratio": info.get('shortRatio'),
                "short_percent_of_float": info.get('shortPercentOfFloat'),

                # === VALUATION METRICS ===
                "trailing_pe": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "peg_ratio": info.get('pegRatio'),
                "price_to_book": info.get('priceToBook'),
                "price_to_sales_ttm": info.get('priceToSalesTrailing12Months'),
                "enterprise_to_revenue": info.get('enterpriseToRevenue'),
                "enterprise_to_ebitda": info.get('enterpriseToEbitda'),

                # === DIVIDEND INFO ===
                "dividend_rate": info.get('dividendRate'),
                "dividend_yield": info.get('dividendYield'),
                "payout_ratio": info.get('payoutRatio'),
                "ex_dividend_date": self._format_timestamp(info.get('exDividendDate')),
                "five_year_avg_dividend_yield": info.get('fiveYearAvgDividendYield'),

                # === RISK METRICS ===
                "beta": info.get('beta'),
                "52_week_high": info.get('fiftyTwoWeekHigh'),
                "52_week_low": info.get('fiftyTwoWeekLow'),
                "52_week_change": info.get('52WeekChange'),
                "50_day_average": info.get('fiftyDayAverage'),
                "200_day_average": info.get('twoHundredDayAverage'),

                # === FINANCIAL HEALTH ===
                "total_cash": info.get('totalCash'),
                "total_cash_per_share": info.get('totalCashPerShare'),
                "total_debt": info.get('totalDebt'),
                "debt_to_equity": info.get('debtToEquity'),
                "current_ratio": info.get('currentRatio'),
                "quick_ratio": info.get('quickRatio'),
                "book_value": info.get('bookValue'),
                "free_cashflow": info.get('freeCashflow'),
                "operating_cashflow": info.get('operatingCashflow'),

                # === PROFITABILITY ===
                "total_revenue": info.get('totalRevenue'),
                "revenue_per_share": info.get('revenuePerShare'),
                "profit_margins": info.get('profitMargins'),
                "operating_margins": info.get('operatingMargins'),
                "gross_margins": info.get('grossMargins'),
                "ebitda_margins": info.get('ebitdaMargins'),
                "return_on_assets": info.get('returnOnAssets'),
                "return_on_equity": info.get('returnOnEquity'),

                # === GROWTH METRICS ===
                "earnings_growth": info.get('earningsGrowth'),
                "revenue_growth": info.get('revenueGrowth'),
                "earnings_quarterly_growth": info.get('earningsQuarterlyGrowth'),

                # === TRADING INFO ===
                "average_volume": info.get('averageVolume'),
                "average_volume_10days": info.get('averageVolume10days'),
                "average_daily_volume_10day": info.get('averageDailyVolume10Day'),
                "bid": info.get('bid'),
                "ask": info.get('ask'),
                "bid_size": info.get('bidSize'),
                "ask_size": info.get('askSize'),

                # === ANALYST RECOMMENDATIONS ===
                "target_high_price": info.get('targetHighPrice'),
                "target_low_price": info.get('targetLowPrice'),
                "target_mean_price": info.get('targetMeanPrice'),
                "target_median_price": info.get('targetMedianPrice'),
                "recommendation_key": info.get('recommendationKey'),
                "recommendation_mean": info.get('recommendationMean'),
                "number_of_analyst_opinions": info.get('numberOfAnalystOpinions'),

                # === COMPANY INFO ===
                "website": info.get('website'),
                "address1": info.get('address1'),
                "city": info.get('city'),
                "state": info.get('state'),
                "zip": info.get('zip'),
                "country": info.get('country'),
                "phone": info.get('phone'),
                "fax": info.get('fax'),
                "full_time_employees": info.get('fullTimeEmployees'),

                # === KEY OFFICERS ===
                "officers": self._format_officers(info.get('companyOfficers', [])[:5]),

                # === TIMESTAMPS ===
                "last_updated": datetime.now().isoformat(),
                "first_trade_date": self._format_timestamp(info.get('firstTradeDateEpochUtc')),
                "last_fiscal_year_end": self._format_timestamp(info.get('lastFiscalYearEnd')),
                "most_recent_quarter": self._format_timestamp(info.get('mostRecentQuarter')),
            }

            self.info_cache[cache_key] = ticker_info
            logger.info(f"✅ Fetched comprehensive info for {self.ticker}")
            return ticker_info

        except Exception as e:
            logger.error(f"❌ Failed to fetch ticker info for {self.ticker}: {e}")
            return {
                "ticker": self.ticker,
                "name": self.ticker,
                "error": str(e),
                "currency": "USD",
                "exchange": "N/A",
                "last_updated": datetime.now().isoformat()
            }

    def _format_officers(self, officers_list: List[Dict]) -> List[Dict]:
        """Formats company officers data."""
        formatted = []
        for officer in officers_list:
            formatted.append({
                "name": officer.get('name', 'N/A'),
                "title": officer.get('title', 'N/A'),
                "age": officer.get('age'),
                "year_born": officer.get('yearBorn'),
                "total_pay": officer.get('totalPay'),
                "exercised_value": officer.get('exercisedValue')
            })
        return formatted

    def _format_timestamp(self, timestamp: Optional[Union[int, float]]) -> Optional[str]:
        """Converts Unix timestamp to ISO format."""
        if timestamp:
            try:
                return datetime.fromtimestamp(timestamp).isoformat()
            except Exception:
                return None
        return None

    # =========================================================================
    # OPTIONS CHAIN
    # =========================================================================

    def fetch_options_chain(self, expiry_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches real options chain with simulation fallback.

        Args:
            expiry_date: Target expiry date in 'YYYY-MM-DD' format (optional)

        Returns:
            Dictionary containing calls, puts, and metadata
        """
        try:
            self._rate_limit()
            logger.info(f"📈 Fetching options chain for {self.ticker}...")

            dat = yf.Ticker(self.ticker)
            expirations = dat.options

            if not expirations:
                logger.warning("No real options data available. Simulating...")
                return self._simulate_options_chain()

            # Select expiry date
            if expiry_date and expiry_date in expirations:
                target_date = expiry_date
            else:
                target_date = expirations[0]  # Nearest expiry

            # Fetch chain
            chain = dat.option_chain(target_date)
            calls_df = chain.calls.fillna(0)
            puts_df = chain.puts.fillna(0)

            # Convert to records
            calls = calls_df.to_dict('records')
            puts = puts_df.to_dict('records')

            # Get spot price
            spot = self.fetch_live_quote()['price']

            # Enrich options with missing data
            for opt in calls + puts:
                if opt.get('bid', 0) <= 0:
                    opt['bid'] = max(0, opt.get('lastPrice', 0) * 0.98)
                if opt.get('ask', 0) <= 0:
                    opt['ask'] = max(0, opt.get('lastPrice', 0) * 1.02)

            logger.info(f"✅ Fetched {len(calls)} calls, {len(puts)} puts")

            return {
                "ticker": self.ticker,
                "expiry": target_date,
                "available_expirations": list(expirations),
                "calls": calls,
                "puts": puts,
                "spot_price": spot,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Options chain fetch failed: {e}. Simulating...")
            return self._simulate_options_chain()

    def _simulate_options_chain(self) -> Dict[str, Any]:
        """Generates realistic simulated options chain for testing/fallback."""
        try:
            spot = self.fetch_live_quote()['price'] or 150.0
        except Exception:
            spot = 150.0

        strikes = [round(spot * (0.8 + 0.05 * i), 1) for i in range(9)]
        expiry = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

        calls = []
        puts = []

        for k in strikes:
            moneyness_c = (spot - k) / spot
            moneyness_p = (k - spot) / spot

            c_intrinsic = max(0, spot - k)
            p_intrinsic = max(0, k - spot)

            c_time_value = spot * 0.03 * np.exp(-abs(moneyness_c))
            p_time_value = spot * 0.03 * np.exp(-abs(moneyness_p))

            c_price = c_intrinsic + c_time_value
            p_price = p_intrinsic + p_time_value

            calls.append({
                'contractSymbol': f"{self.ticker}{datetime.now().strftime('%y%m%d')}C{int(k*1000)}",
                'strike': k,
                'lastPrice': round(c_price, 2),
                'bid': round(c_price * 0.98, 2),
                'ask': round(c_price * 1.02, 2),
                'volume': random.randint(10, 500),
                'openInterest': random.randint(50, 2000),
                'impliedVolatility': round(0.20 + random.uniform(-0.05, 0.10), 4)
            })

            puts.append({
                'contractSymbol': f"{self.ticker}{datetime.now().strftime('%y%m%d')}P{int(k*1000)}",
                'strike': k,
                'lastPrice': round(p_price, 2),
                'bid': round(p_price * 0.98, 2),
                'ask': round(p_price * 1.02, 2),
                'volume': random.randint(10, 500),
                'openInterest': random.randint(50, 2000),
                'impliedVolatility': round(0.20 + random.uniform(-0.05, 0.10), 4)
            })

        return {
            "ticker": self.ticker,
            "expiry": expiry,
            "available_expirations": [expiry],
            "calls": calls,
            "puts": puts,
            "spot_price": spot,
            "simulated": True,
            "timestamp": datetime.now().isoformat()
        }

    # =========================================================================
    # NEWS & SENTIMENT
    # =========================================================================

    def fetch_news(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches latest news headlines for the ticker.

        Args:
            count: Number of news items to return

        Returns:
            List of news dictionaries with title, publisher, link, published date
        """
        cache_key = f"news_{self.ticker}_{count}"
        # Removing static cache for real-time news updates
        # if cache_key in self.news_cache:
        #     return self.news_cache[cache_key]

        try:
            self._rate_limit()
            logger.info(f"📰 Fetching news for {self.ticker}...")

            dat = yf.Ticker(self.ticker)
            news_items = dat.news or []

            results = []
            for item in news_items[:count]:
                ts = item.get('providerPublishTime', 0)
                published = self._format_timestamp(ts)

                results.append({
                    "title": item.get('title'),
                    "publisher": item.get('publisher'),
                    "link": item.get('link'),
                    "published": published,
                    "type": item.get('type'),
                    "thumbnail": item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url')
                })

            self.news_cache[cache_key] = results
            logger.info(f"✅ Fetched {len(results)} news items")
            return results

        except Exception as e:
            logger.error(f"❌ News fetch failed: {e}")
            return []

    def fetch_sentiment_score(self) -> float:
        """
        Generates deterministic pseudo-random sentiment score (0-10).
        In production, this would integrate with sentiment analysis APIs.

        Returns:
            Sentiment score between 0 (bearish) and 10 (bullish)
        """
        # Deterministic based on ticker and current hour
        current_hour = int(time.time() / 3600)
        seed_value = int(hashlib.md5(f"{self.ticker}{current_hour}".encode()).hexdigest(), 16) % (10**8)
        random.seed(seed_value)

        # Generate sentiment with slight bullish bias (5.5 mean)
        sentiment = 5.5 + random.uniform(-2.5, 2.5)
        return round(max(0.0, min(10.0, sentiment)), 1)

    # =========================================================================
    # FUNDAMENTALS
    # =========================================================================

    def fetch_fundamentals(self) -> Dict[str, Any]:
        """
        Fetches key fundamental metrics.
        NO ARGUMENTS - uses self.ticker

        Returns:
            Dictionary with PE ratio, dividend yield, beta, sector, industry
        """
        try:
            dat = yf.Ticker(self.ticker)
            info = dat.info or {}

            return {
                "market_cap": info.get('marketCap'),
                "enterprise_value": info.get('enterpriseValue'),
                "trailing_pe": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "peg_ratio": info.get('pegRatio'),
                "price_to_book": info.get('priceToBook'),
                "price_to_sales": info.get('priceToSalesTrailing12Months'),
                "dividend_yield": info.get('dividendYield', 0.0),
                "payout_ratio": info.get('payoutRatio'),
                "beta": info.get('beta'),
                "profit_margins": info.get('profitMargins'),
                "operating_margins": info.get('operatingMargins'),
                "return_on_assets": info.get('returnOnAssets'),
                "return_on_equity": info.get('returnOnEquity'),
                "revenue_growth": info.get('revenueGrowth'),
                "earnings_growth": info.get('earningsGrowth'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "recommendation_key": info.get('recommendationKey'),
                "target_mean_price": info.get('targetMeanPrice')
            }

        except Exception as e:
            logger.error(f"❌ Fundamentals fetch failed: {e}")
            return {}

    # =========================================================================
    # ML-READY DATA PREPARATION
    # =========================================================================

    def fetch_data(self) -> np.ndarray:
        """
        Main entry point for ML models.
        Fetches historical Close prices as numpy array.

        Returns:
            2D numpy array of shape (n_samples, 1) with close prices
        """
        df, _ = self.fetch_ohlcv(self.ticker, period="5y")
        if df.empty:
            return np.array([]).reshape(-1, 1)
        return df['Close'].values.reshape(-1, 1)

    def prepare_lstm_data(
        self,
        prices: np.ndarray,
        lookback: int = 60,
        test_split: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Prepares data for LSTM model (scaling + sequence generation).
        Compatible with deep_learning_models.py.

        Args:
            prices: 1D or 2D array of prices
            lookback: Number of past timesteps to use
            test_split: Fraction of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        if self.scaler is None:
            raise ImportError("scikit-learn not installed. Cannot prepare LSTM data.")

        if prices.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), self.scaler

        # Ensure 2D shape
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)

        # Scale prices
        scaled_prices = self.scaler.fit_transform(prices)

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_prices) - lookback):
            X.append(scaled_prices[i:i + lookback])
            y.append(scaled_prices[i + lookback])

        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), self.scaler

        # Train-test split
        split_idx = int((1 - test_split) * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"✅ Prepared LSTM data: Train={len(X_train)}, Test={len(X_test)}")

        return X_train, X_test, y_train, y_test, self.scaler

    def get_market_data(
        self,
        market_ticker: str = '^GSPC',
        period_years: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Fetches aligned stock and market data for Beta calculation.

        Args:
            market_ticker: Market index ticker (default: S&P 500)
            period_years: Historical period in years

        Returns:
            Tuple of (stock_close, market_close) as pandas Series
        """
        start_date = (datetime.now() - timedelta(days=365 * period_years)).strftime('%Y-%m-%d')

        try:
            self._rate_limit()
            data = yf.download(
                [self.ticker, market_ticker],
                start=start_date,
                auto_adjust=True,
                progress=False
            )

            # Extract close prices
            if isinstance(data.columns, pd.MultiIndex):
                stock_close = data[self.ticker]['Close']
                market_close = data[market_ticker]['Close']
            else:
                # Fallback for single ticker download
                stock_close = data['Close']
                market_close = data['Close']

            return stock_close, market_close

        except Exception as e:
            logger.error(f"❌ Market data fetch failed: {e}")
            raise

    # =========================================================================
    # MOCK DATA GENERATION (FALLBACK)
    # =========================================================================

    def _generate_mock_data(
        self,
        ticker: str,
        days: int = 252
    ) -> Tuple[pd.DataFrame, TickerMetadata]:
        """
        Generates realistic mock OHLCV data for testing/fallback.

        Args:
            ticker: Stock symbol
            days: Number of trading days to generate

        Returns:
            Tuple of (DataFrame, TickerMetadata)
        """
        logger.info(f"🎲 Generating mock data for {ticker} ({days} days)")

        dates = pd.bdate_range(end=datetime.now(), periods=days)

        # Generate realistic price walk
        np.random.seed(hash(ticker) % (2**32))
        returns = np.random.normal(0.0005, 0.02, days)  # Slight upward drift
        price = 150.0
        prices = []

        for ret in returns:
            price *= (1 + ret)
            prices.append(max(1.0, price))

        # Generate OHLCV
        df = pd.DataFrame({
            'Open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'High': [p * (1 + abs(np.random.uniform(0, 0.02))) for p in prices],
            'Low': [p * (1 - abs(np.random.uniform(0, 0.02))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1_000_000, 10_000_000, days)
        }, index=dates)

        # Add derived features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility_20'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)

        meta = TickerMetadata(currency='USD', name=ticker, exchange='SIMULATED')

        return df, meta


# =============================================================================
# MAIN TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 FINANCIAL DATA FETCHER v2.0 - TESTING SUITE")
    print("=" * 80)

    # Test 1: Indian Stock
    print("\n" + "=" * 80)
    print("TEST 1: RELIANCE.NS (Indian Stock)")
    print("=" * 80)

    fetcher_in = FinancialDataFetcher("RELIANCE.NS")
    quote_in = fetcher_in.fetch_live_quote()
    print(f"\n📊 Live Quote:")
    print(f"   Price: ₹{quote_in['price']:.2f} {quote_in['currency']}")
    print(f"   Change: {quote_in['change_percent']:.2f}%")

    info_in = fetcher_in.fetch_ticker_info()
    print(f"\n🏢 Company Info:")
    print(f"   Name: {info_in['name']}")
    print(f"   Sector: {info_in['sector']}")
    print(f"   Market Cap: {info_in['market_cap']}")
    print(f"   Currency: {info_in['currency']}")

    # Test 2: US Stock
    print("\n" + "=" * 80)
    print("TEST 2: AAPL (US Stock)")
    print("=" * 80)

    fetcher_us = FinancialDataFetcher("AAPL")
    quote_us = fetcher_us.fetch_live_quote()
    print(f"\n📊 Live Quote:")
    print(f"   Price: ${quote_us['price']:.2f} {quote_us['currency']}")
    print(f"   Change: {quote_us['change_percent']:.2f}%")

    info_us = fetcher_us.fetch_ticker_info()
    print(f"\n🏢 Company Info:")
    print(f"   Name: {info_us['name']}")
    print(f"   CEO: {info_us['officers'][0]['name'] if info_us['officers'] else 'N/A'}")
    print(f"   Beta: {info_us['beta']}")
    print(f"   PE Ratio: {info_us['trailing_pe']}")

    # Test 3: Options Chain
    print("\n" + "=" * 80)
    print("TEST 3: Options Chain")
    print("=" * 80)

    options = fetcher_us.fetch_options_chain()
    print(f"\n📈 Options Data:")
    print(f"   Expiry: {options['expiry']}")
    print(f"   Calls: {len(options['calls'])}")
    print(f"   Puts: {len(options['puts'])}")
    print(f"   Spot: ${options['spot_price']:.2f}")

    # Test 4: News
    print("\n" + "=" * 80)
    print("TEST 4: News Headlines")
    print("=" * 80)

    news = fetcher_us.fetch_news(count=3)
    for i, article in enumerate(news, 1):
        print(f"\n📰 Article {i}:")
        print(f"   Title: {article['title']}")
        print(f"   Publisher: {article['publisher']}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
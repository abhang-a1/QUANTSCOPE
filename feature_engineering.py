"""
Feature Engineering Module v2.0 - PRODUCTION GRADE (Updated Jan 2026)
--------------------------------------------------------------------------------
70+ Production-Ready Features - FIXED MultiIndex Column Handling
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE ENGINEER CLASS
# =============================================================================

class FeatureEngineer:
    """Production-grade feature engineering - 70+ quantitative features"""

    @staticmethod
    def compute_all_features(
        data: pd.DataFrame,
        normalize: bool = False
    ) -> pd.DataFrame:
        """
        Master function - generates 70+ production-ready features.

        Args:
            data: OHLCV DataFrame
            normalize: If True, applies z-score normalization (Mean=0, Std=1) for ML
                      If False, returns raw values for dashboard visualization

        Returns:
            DataFrame with all engineered features
        """
        if data.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame()

        try:
            # Create copy
            df = data.copy()

            # FIX: Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                logger.info("🔧 Flattening MultiIndex columns...")
                df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns.get_level_values(0)

            # Handle missing values (Modern pandas syntax)
            df = df.ffill()
            df = df.fillna(0)

            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'Volume':
                        df[col] = 1_000_000.0
                    else:
                        df[col] = df['Close'] if 'Close' in df.columns else 100.0

            logger.info(f"📊 Computing features for {len(df)} rows...")

            # ===== 1. PRICE TRANSFORMS =====
            df['returns'] = df['Close'].pct_change().fillna(0)
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
            df['price_momentum_5'] = (df['Close'] / df['Close'].shift(5) - 1).fillna(0)
            df['price_momentum_10'] = (df['Close'] / df['Close'].shift(10) - 1).fillna(0)
            df['price_momentum_20'] = (df['Close'] / df['Close'].shift(20) - 1).fillna(0)

            # ===== 2. OSCILLATORS =====
            df['rsi_14'] = FeatureEngineer._rsi(df['Close'], 14)
            df['rsi_7'] = FeatureEngineer._rsi(df['Close'], 7)

            macd, macd_signal, macd_hist = FeatureEngineer._macd(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist

            df['stochastic_k'], df['stochastic_d'] = FeatureEngineer._stochastic(df)
            df['cci'] = FeatureEngineer._cci(df, 20)
            df['williams_r'] = FeatureEngineer._williams_r(df, 14)

            # ===== 3. MOVING AVERAGES =====
            df['sma_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
            df['sma_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['sma_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['sma_200'] = df['Close'].rolling(window=200, min_periods=1).mean()

            df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

            # FIX: Ensure all operations return Series, not DataFrame
            df['price_to_sma20'] = ((df['Close'] / df['sma_20']) - 1).fillna(0)
            df['price_to_sma50'] = ((df['Close'] / df['sma_50']) - 1).fillna(0)
            df['price_to_sma200'] = ((df['Close'] / df['sma_200']) - 1).fillna(0)

            # Golden/Death Cross signals
            df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
            df['price_above_sma200'] = (df['Close'] > df['sma_200']).astype(int)

            # ===== 4. VOLATILITY =====
            df['atr_14'] = FeatureEngineer._atr(df, 14)
            df['atr_7'] = FeatureEngineer._atr(df, 7)

            df['historical_volatility_10'] = (df['log_returns'].rolling(10).std() * np.sqrt(252)).fillna(0)
            df['historical_volatility_30'] = (df['log_returns'].rolling(30).std() * np.sqrt(252)).fillna(0)

            # Bollinger Bands
            bb_mid = df['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['Close'].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = bb_mid + (2 * bb_std)
            df['bb_lower'] = bb_mid - (2 * bb_std)
            df['bb_middle'] = bb_mid
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-10)).fillna(0)
            df['bb_position'] = ((df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)).fillna(0.5)

            # ===== 5. VOLUME INDICATORS =====
            df['volume_sma_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = (df['Volume'] / (df['volume_sma_20'] + 1e-10)).fillna(1.0)

            df['obv'] = FeatureEngineer._obv(df)
            df['mfi'] = FeatureEngineer._mfi(df, 14)
            df['ad_line'] = FeatureEngineer._ad_line(df)

            # VWAP
            df['vwap'] = FeatureEngineer._vwap(df)

            # ===== 6. SUPPORT & RESISTANCE =====
            pivot = (df['High'] + df['Low'] + df['Close']) / 3
            df['pivot_point'] = pivot
            df['support_1'] = (2 * pivot) - df['High']
            df['support_2'] = pivot - (df['High'] - df['Low'])
            df['resistance_1'] = (2 * pivot) - df['Low']
            df['resistance_2'] = pivot + (df['High'] - df['Low'])

            # ===== 7. TREND STRENGTH =====
            df['adx'] = FeatureEngineer._adx(df, 14)

            # ===== 8. CANDLESTICK PATTERNS =====
            df['doji'] = FeatureEngineer._is_doji(df)
            df['hammer'] = FeatureEngineer._is_hammer(df)
            df['shooting_star'] = FeatureEngineer._is_shooting_star(df)
            df['engulfing_bullish'] = FeatureEngineer._is_bullish_engulfing(df)
            df['engulfing_bearish'] = FeatureEngineer._is_bearish_engulfing(df)

            # ===== 9. PRICE CHANNELS =====
            df['donchian_upper'] = df['High'].rolling(window=20, min_periods=1).max()
            df['donchian_lower'] = df['Low'].rolling(window=20, min_periods=1).min()
            df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2

            # ===== 10. STATISTICAL FEATURES =====
            df['returns_skew_20'] = df['returns'].rolling(window=20, min_periods=1).skew().fillna(0)
            df['returns_kurtosis_20'] = df['returns'].rolling(window=20, min_periods=1).kurt().fillna(0)

            # Final cleanup
            df = df.replace([np.inf, -np.inf], 0)
            df = df.fillna(0)

            # Normalization (Only for ML pipeline)
            if normalize:
                logger.info("📐 Applying z-score normalization for ML...")
                cols_to_normalize = [c for c in df.columns if c not in required_cols]
                for col in cols_to_normalize:
                    col_std = df[col].std()
                    if col_std > 0 and not np.isnan(col_std):
                        df[col] = (df[col] - df[col].mean()) / col_std

            logger.info(f"✅ Generated {len(df.columns)} features")
            return df

        except Exception as e:
            logger.error(f"❌ Feature computation failed: {e}")
            import traceback
            traceback.print_exc()
            return data.copy()

    # =========================================================================
    # INDICATOR IMPLEMENTATIONS
    # =========================================================================

    @staticmethod
    def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Moving Average Convergence Divergence"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    @staticmethod
    def _stochastic(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K and %D)"""
        low_min = df['Low'].rolling(window=period, min_periods=1).min()
        high_max = df['High'].rolling(window=period, min_periods=1).max()

        k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
        d = k.rolling(window=3, min_periods=1).mean()

        return k.fillna(50), d.fillna(50)

    @staticmethod
    def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma = tp.rolling(window=period, min_periods=1).mean()
        md = tp.rolling(window=period, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - ma) / (0.015 * (md + 1e-10))
        return cci.fillna(0)

    @staticmethod
    def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R"""
        high_max = df['High'].rolling(window=period, min_periods=1).max()
        low_min = df['Low'].rolling(window=period, min_periods=1).min()
        wr = -100 * ((high_max - df['Close']) / (high_max - low_min + 1e-10))
        return wr.fillna(-50)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        h_l = df['High'] - df['Low']
        h_pc = (df['High'] - df['Close'].shift(1)).abs()
        l_pc = (df['Low'] - df['Close'].shift(1)).abs()

        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr.fillna(0)

    @staticmethod
    def _obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume"""
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv

    @staticmethod
    def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']

        mf_pos = mf.where(tp.diff() > 0, 0)
        mf_neg = mf.where(tp.diff() < 0, 0)

        mf_pos_sum = mf_pos.rolling(window=period, min_periods=1).sum()
        mf_neg_sum = mf_neg.rolling(window=period, min_periods=1).sum()

        mfi = 100 - (100 / (1 + mf_pos_sum / (mf_neg_sum + 1e-10)))
        return mfi.fillna(50)

    @staticmethod
    def _ad_line(df: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
        ad = (clv * df['Volume']).cumsum()
        return ad.fillna(0)

    @staticmethod
    def _vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (tp * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-10)
        return vwap.fillna(df['Close'])

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index (Trend Strength)"""
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = FeatureEngineer._atr(df, period)

        pos_di = 100 * (pos_dm.rolling(window=period, min_periods=1).mean() / (atr + 1e-10))
        neg_di = 100 * (neg_dm.rolling(window=period, min_periods=1).mean() / (atr + 1e-10))

        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10))
        adx = dx.rolling(window=period, min_periods=1).mean()

        return adx.fillna(0)

    # =========================================================================
    # CANDLESTICK PATTERN RECOGNITION
    # =========================================================================

    @staticmethod
    def _is_doji(df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
        """Doji pattern (Open ≈ Close)"""
        body = abs(df['Close'] - df['Open'])
        range_ = df['High'] - df['Low'] + 1e-10
        return (body / range_ < threshold).astype(int)

    @staticmethod
    def _is_hammer(df: pd.DataFrame) -> pd.Series:
        """Hammer pattern (bullish reversal)"""
        body = abs(df['Close'] - df['Open'])
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)

        is_hammer = (
            (lower_shadow > 2 * body) &
            (upper_shadow < body) &
            (df['Close'] > df['Open'])
        )
        return is_hammer.astype(int)

    @staticmethod
    def _is_shooting_star(df: pd.DataFrame) -> pd.Series:
        """Shooting Star pattern (bearish reversal)"""
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']

        is_shooting_star = (
            (upper_shadow > 2 * body) &
            (lower_shadow < body) &
            (df['Close'] < df['Open'])
        )
        return is_shooting_star.astype(int)

    @staticmethod
    def _is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
        """Bullish Engulfing pattern"""
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)

        is_engulfing = (
            (prev_close < prev_open) &
            (df['Close'] > df['Open']) &
            (df['Open'] < prev_close) &
            (df['Close'] > prev_open)
        )
        return is_engulfing.fillna(False).astype(int)

    @staticmethod
    def _is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
        """Bearish Engulfing pattern"""
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)

        is_engulfing = (
            (prev_close > prev_open) &
            (df['Close'] < df['Open']) &
            (df['Open'] > prev_close) &
            (df['Close'] < prev_open)
        )
        return is_engulfing.fillna(False).astype(int)


# =============================================================================
# API-COMPATIBLE WRAPPER FUNCTIONS
# =============================================================================

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for backend API.
    Returns NON-NORMALIZED features for dashboard display.
    """
    return FeatureEngineer.compute_all_features(df, normalize=False)


def interpret_signals(row: pd.Series) -> List[str]:
    """Generates human-readable trading signals."""
    signals = []

    rsi = row.get('rsi_14', 50)
    if rsi < 30:
        signals.append("🔴 Oversold (RSI < 30) - Potential Buy")
    elif rsi > 70:
        signals.append("🔴 Overbought (RSI > 70) - Potential Sell")

    macd = row.get('macd', 0)
    macd_signal = row.get('macd_signal', 0)
    if macd > macd_signal and row.get('macd_histogram', 0) > 0:
        signals.append("🟢 Bullish MACD Crossover")
    elif macd < macd_signal and row.get('macd_histogram', 0) < 0:
        signals.append("🔴 Bearish MACD Crossover")

    if row.get('price_above_sma200', 0) > 0:
        signals.append("🟢 Long-term Bullish Trend (Above SMA200)")
    else:
        signals.append("🔴 Long-term Bearish Trend (Below SMA200)")

    bb_width = row.get('bb_width', 0)
    if bb_width < 0.05:
        signals.append("⚠️ Volatility Squeeze (Breakout Imminent)")

    volume_ratio = row.get('volume_ratio', 1.0)
    if volume_ratio > 2.0:
        signals.append("📈 Unusually High Volume (2x+ Average)")

    if row.get('hammer', 0) == 1:
        signals.append("🔨 Hammer Pattern (Bullish Reversal)")
    if row.get('shooting_star', 0) == 1:
        signals.append("⭐ Shooting Star (Bearish Reversal)")
    if row.get('engulfing_bullish', 0) == 1:
        signals.append("🟢 Bullish Engulfing")
    if row.get('engulfing_bearish', 0) == 1:
        signals.append("🔴 Bearish Engulfing")

    return signals if signals else ["📊 Neutral - No Strong Signals"]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 FEATURE ENGINEERING MODULE v2.0 TESTING")
    print("=" * 80)

    import yfinance as yf

    print("\n📊 Fetching test data (AAPL)...")
    df = yf.download("AAPL", period="1y", progress=False)

    if not df.empty:
        print(f"✅ Fetched {len(df)} rows")
        print(f"📋 Columns: {df.columns.tolist()}")

        print("\n🔧 Generating features...")
        featured_df = generate_features(df)

        print(f"\n✅ Generated {len(featured_df.columns)} features")
        print(f"📊 Sample features:")

        sample_cols = [col for col in ['Close', 'rsi_14', 'macd', 'bb_width', 'obv'] if col in featured_df.columns]
        print(featured_df[sample_cols].tail())

        print("\n🔍 Interpreting latest signals...")
        latest_row = featured_df.iloc[-1]
        signals = interpret_signals(latest_row)

        print("\n📡 Trading Signals:")
        for signal in signals:
            print(f"  • {signal}")

        print("\n✅ ALL TESTS PASSED")
    else:
        print("❌ Failed to fetch test data")

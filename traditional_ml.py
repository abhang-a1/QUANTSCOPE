"""
Traditional ML Models (Beta, SVR, Trees) - QUANT PRODUCTION GRADE v2.0
Optimized for maximum accuracy using Feature Engineering, GridSearch & Ensemble Methods.
Integrated with data_fetcher.py for live market data.

Dependencies:
    pip install numpy pandas scikit-learn xgboost lightgbm yfinance
"""

import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime
from typing import Tuple, Dict, Optional

# Sklearn & Stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

# Import Data Fetcher
try:
    from data_fetcher import FinancialDataFetcher
    FETCHER_AVAILABLE = True
except ImportError:
    FETCHER_AVAILABLE = False
    import yfinance as yf

# Safe imports for boosting
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantML")

# ==============================================================================
# 1. LINEAR REGRESSION: BETA, ALPHA & MARKET CORRELATION (CAPM MODEL)
# ==============================================================================
def calculate_beta_and_correlation(
    stock_ticker: str = 'NVDA',
    market_ticker: str = '^GSPC',
    period: str = '2y'
) -> Tuple[float, float, float]:
    """
    Calculates Beta (Systematic Risk), Alpha (Excess Return), and R² using CAPM.

    Returns:
        beta: Market sensitivity coefficient
        alpha: Annualized excess return
        r_squared: Model fit quality
    """
    logger.info(f"📊 Calculating Beta for {stock_ticker} vs {market_ticker}...")

    try:
        # Fetch data using data_fetcher if available
        if FETCHER_AVAILABLE:
            fetcher_stock = FinancialDataFetcher(ticker=stock_ticker)
            fetcher_market = FinancialDataFetcher(ticker=market_ticker)

            df_stock, _ = fetcher_stock.fetch_ohlcv(stock_ticker, period=period)
            df_market, _ = fetcher_market.fetch_ohlcv(market_ticker, period=period)

            stock_close = df_stock['Close']
            market_close = df_market['Close']
        else:
            # Fallback to yfinance
            import yfinance as yf
            data = yf.download(
                [stock_ticker, market_ticker],
                period=period,
                progress=False,
                group_by='ticker',
                auto_adjust=True
            )

            if isinstance(data.columns, pd.MultiIndex):
                stock_close = data[stock_ticker]['Close']
                market_close = data[market_ticker]['Close']
            else:
                stock_close = data['Close']
                market_close = data['Close']

        # Log Returns (Stationary transformation)
        stock_ret = np.log(stock_close / stock_close.shift(1)).dropna()
        market_ret = np.log(market_close / market_close.shift(1)).dropna()

        # Alignment
        data_ret = pd.concat([stock_ret, market_ret], axis=1).dropna()
        data_ret.columns = ['Stock', 'Market']

        if len(data_ret) < 50:
            raise ValueError("Insufficient data for regression (need at least 50 days)")

        X = data_ret['Market'].values.reshape(-1, 1)
        y = data_ret['Stock'].values.reshape(-1, 1)

        # OLS Regression
        model = LinearRegression()
        model.fit(X, y)

        beta = float(model.coef_[0][0])
        alpha_daily = float(model.intercept_[0])
        r_squared = float(model.score(X, y))

        # Correlation
        correlation = np.corrcoef(X.flatten(), y.flatten())[0, 1]

        # Annualized Alpha (252 trading days)
        alpha_annual = (1 + alpha_daily) ** 252 - 1

        logger.info(f"✅ Beta: {beta:.4f} | Alpha (Annual): {alpha_annual:.2%} | R²: {r_squared:.4f}")

        # Risk interpretation
        if beta > 1.2:
            risk_profile = "High Volatility (Aggressive)"
        elif beta > 0.8:
            risk_profile = "Market-like (Moderate)"
        elif beta > 0.0:
            risk_profile = "Defensive (Low Volatility)"
        else:
            risk_profile = "Inverse/Hedging"

        logger.info(f"📈 Risk Profile: {risk_profile}")

        return beta, alpha_annual, r_squared

    except Exception as e:
        logger.error(f"❌ Beta calculation failed: {e}")
        return 1.0, 0.0, 0.0


# ==============================================================================
# 2. ADVANCED FEATURE ENGINEERING (70+ FEATURES)
# ==============================================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 70+ technical indicators for ML model training.

    Categories:
    - Momentum: RSI, Stochastic, ROC
    - Trend: MACD, Moving Averages
    - Volatility: ATR, Bollinger Bands
    - Volume: OBV, Volume Ratios
    - Lagged Features: 1-20 day lags
    """
    df = df.copy()

    # Ensure Close is 1D Series
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

    # Handle OHLCV columns
    high = df['High'].squeeze() if 'High' in df.columns else close
    low = df['Low'].squeeze() if 'Low' in df.columns else close
    volume = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(1, index=df.index)

    # ===== MOMENTUM INDICATORS =====

    # 1. RSI (Multiple Periods)
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    # 2. Stochastic Oscillator
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((close - low_14) / (high_14 - low_14 + 1e-10))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()

    # 3. Rate of Change (ROC)
    for period in [5, 10, 20]:
        df[f'ROC_{period}'] = ((close - close.shift(period)) / (close.shift(period) + 1e-10)) * 100

    # ===== TREND INDICATORS =====

    # 4. MACD (Multiple Settings)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 5. Moving Averages (Multiple Periods)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = close.rolling(window=period).mean()
        df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()

    # 6. Moving Average Convergence
    df['SMA_Cross_5_20'] = df['SMA_5'] - df['SMA_20']
    df['SMA_Cross_20_50'] = df['SMA_20'] - df['SMA_50']

    # ===== VOLATILITY INDICATORS =====

    # 7. Bollinger Bands (Multiple Periods)
    for period in [10, 20, 30]:
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        df[f'BB_Upper_{period}'] = sma + (2 * std)
        df[f'BB_Lower_{period}'] = sma - (2 * std)
        df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / (sma + 1e-10)
        df[f'BB_Position_{period}'] = (close - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'] + 1e-10)

    # 8. ATR (Average True Range)
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    df['ATR_Ratio'] = df['ATR_14'] / (close + 1e-10)

    # ===== VOLUME INDICATORS =====

    # 9. Volume Features
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / (df['Volume_SMA_20'] + 1e-10)

    # 10. OBV (On-Balance Volume)
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['OBV'] = obv
    df['OBV_EMA'] = obv.ewm(span=20, adjust=False).mean()

    # ===== RETURNS & VOLATILITY =====

    # 11. Log Returns (Multiple Horizons)
    df['Log_Ret'] = np.log(close / close.shift(1))
    for period in [5, 10, 20]:
        df[f'Return_{period}d'] = (close - close.shift(period)) / (close.shift(period) + 1e-10)

    # 12. Historical Volatility
    for period in [10, 20, 30]:
        df[f'Volatility_{period}'] = df['Log_Ret'].rolling(window=period).std() * np.sqrt(252)

    # ===== LAGGED FEATURES (CRITICAL FOR TIME SERIES ML) =====

    # 13. Price & Return Lags
    for lag in [1, 2, 3, 5, 7, 10, 14, 20]:
        df[f'Lag_Return_{lag}'] = df['Log_Ret'].shift(lag)
        df[f'Lag_Close_{lag}'] = close.shift(lag)

    # 14. Rolling Statistics
    for window in [5, 10, 20]:
        df[f'Roll_Mean_{window}'] = close.rolling(window=window).mean()
        df[f'Roll_Std_{window}'] = close.rolling(window=window).std()
        df[f'Roll_Min_{window}'] = close.rolling(window=window).min()
        df[f'Roll_Max_{window}'] = close.rolling(window=window).max()

    # ===== TIME-BASED FEATURES =====

    # 15. Seasonal Features
    if isinstance(df.index, pd.DatetimeIndex):
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter

    # Drop NaN rows
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)

    logger.info(f"✅ Generated {len(df.columns)} features from raw OHLCV data")
    logger.info(f"   Dropped {initial_rows - final_rows} rows with NaN values")

    return df


# ==============================================================================
# 3. DATA PREPARATION WITH ROBUST ERROR HANDLING
# ==============================================================================
def prepare_tabular_data_from_fetcher(
    ticker: str = 'NVDA',
    period: str = '5y',
    target_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fetches data, adds features, and prepares Train/Test splits.

    Args:
        ticker: Stock symbol
        period: Historical period
        target_horizon: Days ahead to predict (default: 1)

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"🔧 Preparing ML pipeline for {ticker}...")

    # 1. Fetch Data
    try:
        if FETCHER_AVAILABLE:
            fetcher = FinancialDataFetcher(ticker=ticker)
            df, _ = fetcher.fetch_ohlcv(ticker, period=period)
        else:
            import yfinance as yf
            df = yf.download(ticker, period=period, progress=False)
    except Exception as e:
        logger.error(f"❌ Data fetch failed: {e}")
        raise

    if df.empty or len(df) < 200:
        raise ValueError(f"Insufficient data for {ticker}. Need at least 200 days.")

    # Store Close before feature engineering
    close_price = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

    # 2. Feature Engineering
    df = add_technical_indicators(df)

    # 3. Define Features (X) - Select most important features
    # Exclude non-predictive columns
    exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'Dividends', 'Stock Splits']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Ensure we have numeric features only
    df_numeric = df[feature_cols].select_dtypes(include=[np.number])

    X = df_numeric.values

    # 4. Define Target (y) - Next Day's Close Price
    y = df['Close'].values if 'Close' in df.columns else close_price.loc[df.index].values

    # Shift target: X[t] predicts y[t+target_horizon]
    X = X[:-target_horizon]
    y = y[target_horizon:]

    if len(X) < 100:
        raise ValueError("After feature engineering, insufficient data remains.")

    # 5. Time Series Split (No Shuffle! Maintains temporal order)
    split_idx = int(0.85 * len(X))  # 85% train, 15% test

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"✅ Data prepared: Train={len(X_train)}, Test={len(X_test)}, Features={X.shape[1]}")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# 4. OPTIMIZED SVR WITH GRIDSEARCH
# ==============================================================================
def train_svr(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fast_mode: bool = True
) -> np.ndarray:
    """
    SVR with GridSearch Hyperparameter Tuning.
    Uses RobustScaler to handle outliers better.

    Args:
        fast_mode: If True, uses reduced grid for speed
    """
    logger.info("🤖 Training SVR with GridSearch...")

    # Robust Scaling (better for financial data with outliers)
    scaler_X = RobustScaler()
    scaler_y = StandardScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=3)

    # Grid Search Parameters
    if fast_mode:
        param_grid = {
            'C': [50, 100],
            'gamma': ['scale', 0.01],
            'epsilon': [0.1]
        }
    else:
        param_grid = {
            'C': [10, 50, 100, 500, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.05, 0.1, 0.2]
        }

    grid = GridSearchCV(
        SVR(kernel='rbf'),
        param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train_sc, y_train_sc)

    best_svr = grid.best_estimator_
    logger.info(f"   ✅ Best SVR Params: {grid.best_params_}")
    logger.info(f"   ✅ Best CV Score: {-grid.best_score_:.4f}")

    # Predict
    pred_sc = best_svr.predict(X_test_sc)
    pred = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    logger.info(f"   📊 SVR - RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

    return pred


# ==============================================================================
# 5. ENSEMBLE TREES (RF, XGBoost, LightGBM) + VOTING ENSEMBLE
# ==============================================================================
def train_trees(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Trains Random Forest, XGBoost, LightGBM, and creates a Voting Ensemble.
    """
    logger.info("🌲 Training Ensemble Trees...")
    preds = {}
    models = []

    # 1. Random Forest (Robust Baseline)
    logger.info("   Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    preds['RandomForest'] = rf.predict(X_test)
    models.append(('rf', rf))

    rmse_rf = np.sqrt(mean_squared_error(y_test, preds['RandomForest']))
    r2_rf = r2_score(y_test, preds['RandomForest'])
    logger.info(f"   ✅ RF - RMSE: {rmse_rf:.2f} | R²: {r2_rf:.4f}")

    # 2. XGBoost (High Performance)
    if XGBOOST_AVAILABLE:
        logger.info("   Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        preds['XGBoost'] = xgb.predict(X_test)
        models.append(('xgb', xgb))

        rmse_xgb = np.sqrt(mean_squared_error(y_test, preds['XGBoost']))
        r2_xgb = r2_score(y_test, preds['XGBoost'])
        logger.info(f"   ✅ XGB - RMSE: {rmse_xgb:.2f} | R²: {r2_xgb:.4f}")

    # 3. LightGBM (Fastest)
    if LIGHTGBM_AVAILABLE:
        logger.info("   Training LightGBM...")
        lgbm = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=50,
            max_depth=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)
        preds['LightGBM'] = lgbm.predict(X_test)
        models.append(('lgbm', lgbm))

        rmse_lgb = np.sqrt(mean_squared_error(y_test, preds['LightGBM']))
        r2_lgb = r2_score(y_test, preds['LightGBM'])
        logger.info(f"   ✅ LGB - RMSE: {rmse_lgb:.2f} | R²: {r2_lgb:.4f}")

    # 4. Gradient Boosting (Sklearn)
    logger.info("   Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    preds['GradientBoosting'] = gb.predict(X_test)

    rmse_gb = np.sqrt(mean_squared_error(y_test, preds['GradientBoosting']))
    r2_gb = r2_score(y_test, preds['GradientBoosting'])
    logger.info(f"   ✅ GB - RMSE: {rmse_gb:.2f} | R²: {r2_gb:.4f}")

    # 5. Voting Ensemble (Average of all models)
    if len(preds) >= 2:
        ensemble_pred = np.mean(list(preds.values()), axis=0)
        preds['Ensemble'] = ensemble_pred

        rmse_ens = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        r2_ens = r2_score(y_test, ensemble_pred)
        logger.info(f"   ✅ ENSEMBLE - RMSE: {rmse_ens:.2f} | R²: {r2_ens:.4f}")

    return preds


# ==============================================================================
# MAIN TEST & DEMO
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 QuantScope ML Pipeline - Production Grade")
    print("=" * 80)

    ticker = 'AAPL'  # Change to any ticker

    # 1. Calculate Beta & Market Correlation
    print("\n📊 CAPM Analysis")
    print("-" * 80)
    beta, alpha, r2 = calculate_beta_and_correlation(ticker, '^GSPC', period='2y')

    # 2. Train ML Models
    print("\n🤖 Machine Learning Pipeline")
    print("-" * 80)

    try:
        X_train, X_test, y_train, y_test = prepare_tabular_data_from_fetcher(ticker)

        # SVR
        svr_pred = train_svr(X_train, X_test, y_train, y_test, fast_mode=True)

        # Trees
        tree_preds = train_trees(X_train, X_test, y_train, y_test)

        # Final Summary
        print("\n" + "=" * 80)
        print("📈 MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Model':<20} {'RMSE':<15} {'MAE':<15} {'R²':<10}")
        print("-" * 80)

        rmse_svr = np.sqrt(mean_squared_error(y_test, svr_pred))
        mae_svr = mean_absolute_error(y_test, svr_pred)
        r2_svr = r2_score(y_test, svr_pred)
        print(f"{'SVR':<20} {rmse_svr:<15.2f} {mae_svr:<15.2f} {r2_svr:<10.4f}")

        for name, pred in tree_preds.items():
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            print(f"{name:<20} {rmse:<15.2f} {mae:<15.2f} {r2:<10.4f}")

        print("=" * 80)
        print("✅ Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

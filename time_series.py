"""
Time Series Analysis Module
1. Moving Averages: SMA, EMA, MACD
2. ARIMA: Auto-Regressive Integrated Moving Average + Stationarity Tests
3. GARCH: GJR-GARCH Volatility Forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
import warnings

# Import your local DataFetcher
try:
    from data_fetcher import FinancialDataFetcher
except ImportError:
    print("Warning: data_fetcher.py not found. Make sure it is in the same directory.")

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. MOVING AVERAGES & MOMENTUM (SMA, EMA, MACD)
# ==============================================================================
class TechnicalIndicators:
    @staticmethod
    def calculate_ma_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
        """
        Calculates SMA, EMA, and MACD.
        Returns DataFrame with new columns.
        """
        df = df.copy()

        # SMA & EMA
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=slow, adjust=False).mean()

        # MACD
        df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']

        return df

    @staticmethod
    def plot_indicators(df, ticker):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Price & MA
        ax1.plot(df.index, df['Close'], label='Close', color='black', alpha=0.6)
        ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='blue', linestyle='--')
        ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='orange', linestyle='--')
        ax1.set_title(f"{ticker} - Moving Averages")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MACD
        ax2.plot(df.index, df['MACD_Line'], label='MACD Line', color='blue')
        ax2.plot(df.index, df['MACD_Signal'], label='Signal', color='red')
        ax2.bar(df.index, df['MACD_Hist'], label='Histogram', color='gray', alpha=0.5)
        ax2.set_title("MACD")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.show()


# ==============================================================================
# 2. ARIMA (Auto-Regressive Integrated Moving Average)
# ==============================================================================
class ArimaForecaster:
    @staticmethod
    def check_stationarity(timeseries):
        """
        Performs Augmented Dickey-Fuller test.
        Returns: is_stationary (bool), d (suggested differencing order)
        """
        result = adfuller(timeseries.dropna())
        p_value = result[1]
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("✓ Series is Stationary")
            return True, 0
        else:
            print("⚠ Series is NOT Stationary (Differencing recommended)")
            return False, 1

    @staticmethod
    def fit_predict_arima(series, order=(5, 1, 0), forecast_steps=10):
        """
        Fits ARIMA model and forecasts future values.
        """
        print(f"\n--- Fitting ARIMA{order} ---")
        model = ARIMA(series, order=order)
        model_fit = model.fit()

        # Forecast
        forecast_res = model_fit.get_forecast(steps=forecast_steps)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        return forecast_mean, conf_int, model_fit


# ==============================================================================
# 3. GARCH (Volatility Forecasting)
# ==============================================================================
class GarchVolatility:
    @staticmethod
    def fit_gjr_garch(returns):
        """
        Fits GJR-GARCH(1,1) model to capture asymmetry (leverage effect).
        Returns: Model Result, Conditional Volatility
        """
        print("\n--- Fitting GJR-GARCH(1,1) ---")
        # Scale returns to percentage for better convergence
        scaled_returns = returns * 100

        # GJR-GARCH specification: p=1, q=1, o=1 (asymmetric term)
        # 'o=1' enables the GJR asymmetric term [web:3][web:111]
        model = arch_model(scaled_returns, vol='Garch', p=1, o=1, q=1, dist='Normal')
        res = model.fit(disp='off')

        print(res.summary())
        return res, res.conditional_volatility

    @staticmethod
    def forecast_volatility(garch_result, horizon=5):
        """
        Forecasts future volatility.
        """
        # Forecast variance
        forecasts = garch_result.forecast(horizon=horizon, reindex=False)

        # Convert variance to standard deviation (volatility)
        vol_forecast = np.sqrt(forecasts.variance.values[-1, :])
        return vol_forecast


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_analysis(ticker="NVDA"):
    # 1. FETCH DATA using your local data_fetcher
    print(f"Fetching data for {ticker}...")
    fetcher = FinancialDataFetcher(ticker=ticker)

    # We need a DataFrame for indicators, not just numpy array
    df, meta = fetcher.fetch_ohlcv(ticker, period="2y")

    if df.empty:
        print("Error: No data fetched.")
        return

    # --------------------------------------------------------------------------
    # A. MOVING AVERAGES & MACD
    # --------------------------------------------------------------------------
    print("\n[A] Calculating Technical Indicators...")
    ti = TechnicalIndicators()
    df_tech = ti.calculate_ma_macd(df)
    ti.plot_indicators(df_tech, ticker)

    # --------------------------------------------------------------------------
    # B. ARIMA (Price Forecasting)
    # --------------------------------------------------------------------------
    print("\n[B] ARIMA Price Forecasting...")
    price_series = df['Close']

    # Check Stationarity
    arima_tool = ArimaForecaster()
    is_stationary, d = arima_tool.check_stationarity(price_series)

    # Fit ARIMA (Using (5,1,0) as a robust default for daily stocks)
    # If stationary, d=0, else d=1
    arima_order = (5, d, 0)
    forecast, conf, _ = arima_tool.fit_predict_arima(price_series, order=arima_order, forecast_steps=10)

    print(f"\nNext 5 Days Price Forecast: {forecast[:5].values}")

    # Plot ARIMA
    plt.figure(figsize=(10, 5))
    plt.plot(price_series.index[-50:], price_series.values[-50:], label='Actual')

    # Create forecast dates
    last_date = price_series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='B')

    plt.plot(forecast_dates, forecast, label='Forecast', color='red')
    plt.fill_between(forecast_dates, conf.iloc[:, 0], conf.iloc[:, 1], color='pink', alpha=0.3)
    plt.title(f"ARIMA{arima_order} Price Forecast")
    plt.legend()
    plt.show()

    # --------------------------------------------------------------------------
    # C. GARCH (Volatility Forecasting)
    # --------------------------------------------------------------------------
    print("\n[C] GJR-GARCH Volatility Forecasting...")
    # GARCH requires Returns, not Prices
    returns = df['Close'].pct_change().dropna()

    garch_tool = GarchVolatility()
    garch_res, cond_vol = garch_tool.fit_gjr_garch(returns)

    # Forecast next 5 days volatility
    vol_pred = garch_tool.forecast_volatility(garch_res, horizon=5)
    print(f"\nNext 5 Days Volatility Forecast (%): {vol_pred}")

    # Plot Volatility
    plt.figure(figsize=(10, 5))
    # Plot last 6 months of historical volatility
    plt.plot(returns.index[-126:], cond_vol[-126:], label='Historical Volatility (In-Sample)', color='blue')

    # Plot forecast
    plt.plot(forecast_dates[:5], vol_pred, label='Forecast Volatility', color='orange', marker='o')
    plt.title(f"GJR-GARCH Volatility Model: {ticker}")
    plt.ylabel("Volatility (%)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_analysis("JPM")

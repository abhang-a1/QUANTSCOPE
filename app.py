"""
QuantScope Pro - Institutional AI Trading Analytics Backend v2.1 (Fixed & Restored)
Main Flask API with Authentication - Frontend files in static/ folder
Fixes & Updates:
- Restored and stabilized Ensemble Model logic in /api/predict/
- Safely handles NaN values in features and quotes to prevent JSON serialization errors
- Fully integrates the Options updates (Call/Put) and Forecast overlays
- Handles missing index.html gracefully
Updated: February 24, 2026
"""

import os
import sys
import logging
import traceback
import warnings
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, send_from_directory, session, redirect
from flask_cors import CORS

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("QuantScope")

# ==============================================================================
# IMPORT PATH
# ==============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# CORE DATA FETCHER (CRITICAL)
# ==============================================================================
data_fetcher_available = False
FinancialDataFetcher = None

try:
    from data_fetcher import FinancialDataFetcher

    data_fetcher_available = True
    logger.info("✅ data_fetcher imported successfully")
except ImportError as e:
    logger.error(f"❌ CRITICAL: data_fetcher.py not found: {e}")

# ==============================================================================
# OPTIONAL MODULES
# ==============================================================================
modules = {
    "options": None,
    "timeseries": None,
    "ml": None,
    "deeplearning": None,
    "features": None,
    "metrics": None,
}

try:
    import option_pricing
    modules["options"] = option_pricing
    logger.info("✅ option_pricing imported")
except ImportError as e:
    logger.warning(f"⚠️ option_pricing not loaded: {e}")

try:
    import time_series
    modules["timeseries"] = time_series
    logger.info("✅ time_series imported")
except ImportError as e:
    logger.warning(f"⚠️ time_series not loaded: {e}")

try:
    import traditional_ml
    modules["ml"] = traditional_ml
    logger.info("✅ traditional_ml imported")
except ImportError as e:
    logger.warning(f"⚠️ traditional_ml not loaded: {e}")

try:
    import deep_learning_models
    modules["deeplearning"] = deep_learning_models
    logger.info("✅ deep_learning_models imported")
except ImportError as e:
    logger.warning(f"⚠️ deep_learning_models not loaded: {e}")

try:
    import feature_engineering
    modules["features"] = feature_engineering
    logger.info("✅ feature_engineering imported")
except ImportError as e:
    logger.warning(f"⚠️ feature_engineering not loaded: {e}")

try:
    import metrics
    modules["metrics"] = metrics
    logger.info("✅ metrics imported")
except ImportError as e:
    logger.warning(f"⚠️ metrics not loaded: {e}")

# ==============================================================================
# AUTH
# ==============================================================================
auth_available = False
auth_bp = None

try:
    from auth_routes import auth_bp, login_required

    auth_available = True
    logger.info("✅ auth_routes imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Authentication not available: {e}")

    # Dummy decorator if auth not available
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)

        return decorated_function


# ==============================================================================
# FLASK CONFIG
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/static",
    template_folder=TEMPLATE_DIR,
)

# Session configuration
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "quantscope-2026-change-this-in-production-please"
)
app.config["SESSION_COOKIE_SECURE"] = False  # True in production with HTTPS
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

if auth_available and auth_bp:
    app.register_blueprint(auth_bp)
    logger.info("✅ Authentication routes registered")

# CORS
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Cache-Control"] = "public, max-age=300"
    return response


# ==============================================================================
# DECORATORS
# ==============================================================================
def require_ticker(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ticker = request.args.get("ticker", "").upper().strip()
        if not ticker:
            return jsonify({"error": "ticker parameter required", "example": f"{request.path}?ticker=AAPL"}), 400
        kwargs["ticker"] = ticker
        return f(*args, **kwargs)

    return decorated_function


def safe_api_call(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"API Error in {f.__name__}: {traceback.format_exc()}")
            return jsonify({"error": str(e), "type": type(e).__name__, "timestamp": datetime.now().isoformat()}), 500

    return decorated_function

def safe_float(val, default=0.0):
    """Safely cast value to float, defaulting to 0.0 if NaN or None to prevent JSON errors."""
    if val is None or pd.isna(val):
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default)


# ==============================================================================
# FRONTEND ROUTES
# ==============================================================================
@app.route("/")
def index():
    """Root route - Redirect to login if not authenticated."""
    if auth_available:
        if not session.get("token"):
            return redirect("/login")

    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return jsonify({
            "error": "Frontend missing",
            "message": "Please move your HTML file to static/index.html",
            "current_directory": os.getcwd(),
            "static_directory": STATIC_DIR
        }), 404

    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """Serve main dashboard HTML."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return jsonify({"error": "Frontend missing", "message": "Please move your HTML file to static/index.html", "static_dir": STATIC_DIR}), 404

    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/style.css")
def serve_css():
    try:
        return send_from_directory(STATIC_DIR, "style.css", mimetype="text/css")
    except Exception:
        return "/* style.css not found */", 404, {"Content-Type": "text/css"}


@app.route("/script.js")
def serve_js():
    try:
        return send_from_directory(STATIC_DIR, "script.js", mimetype="application/javascript")
    except Exception:
        return "// script.js not found", 404, {"Content-Type": "application/javascript"}


@app.route("/static/css/<path:filename>")
def serve_static_css(filename):
    try:
        return send_from_directory(os.path.join(STATIC_DIR, "css"), filename)
    except Exception:
        return jsonify({"error": f"CSS file not found: {filename}"}), 404


@app.route("/static/js/<path:filename>")
def serve_static_js(filename):
    try:
        return send_from_directory(os.path.join(STATIC_DIR, "js"), filename)
    except Exception:
        return jsonify({"error": f"JS file not found: {filename}"}), 404


@app.route("/static/<path:filename>")
def serve_static_files(filename):
    try:
        return send_from_directory(STATIC_DIR, filename)
    except Exception:
        return jsonify({"error": f"File not found: {filename}"}), 404


# ==============================================================================
# API ROUTES
# ==============================================================================
@app.route("/api/status/", methods=["GET"])
def get_status():
    static_files = os.listdir(STATIC_DIR) if os.path.exists(STATIC_DIR) else []
    return jsonify(
        {
            "status": "online",
            "version": "2026.02.24",
            "timestamp": datetime.now().isoformat(),
            "authentication": {
                "enabled": auth_available,
                "logged_in": bool(session.get("token")) if auth_available else None,
            },
            "modules": {
                "data_fetcher": data_fetcher_available,
                "options": modules["options"] is not None,
                "timeseries": modules["timeseries"] is not None,
                "ml": modules["ml"] is not None,
                "deeplearning": modules["deeplearning"] is not None,
                "features": modules["features"] is not None,
                "metrics": modules["metrics"] is not None,
                "auth": auth_available,
            },
            "frontend": {
                "static_dir": STATIC_DIR,
                "files": static_files,
                "index_html": "index.html" in static_files,
            },
        }
    )


@app.route("/api/ticker-info/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_ticker_info(ticker):
    if not data_fetcher_available:
        return jsonify({"error": "Data fetcher unavailable"}), 503

    fetcher = FinancialDataFetcher(ticker=ticker)
    info = fetcher.fetch_ticker_info()

    if not info or "error" in info:
        return jsonify({"error": "Failed to fetch ticker information", "ticker": ticker}), 404

    news = fetcher.fetch_news(count=5)
    logger.info(f"✅ Ticker info: {ticker} - {info.get('name', 'N/A')}")

    return jsonify({"success": True, "ticker": ticker, "data": {**info, "news": news}, "timestamp": datetime.now().isoformat()})


@app.route("/api/quote/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_live_quote(ticker):
    if not data_fetcher_available:
        return jsonify({"error": "Data fetcher unavailable"}), 503

    fetcher = FinancialDataFetcher(ticker=ticker)
    quote = fetcher.fetch_live_quote()
    sentiment = fetcher.fetch_sentiment_score()

    if not quote or quote.get("price", 0) == 0:
        return jsonify({"error": "Failed to fetch live quote"}), 404

    try:
        hist_df, meta = fetcher.fetch_ohlcv(ticker, period="3mo")
        currency = meta.currency if meta else quote.get("currency", "USD")
        h_max = hist_df["High"].max() if not hist_df.empty else pd.NA
        l_min = hist_df["Low"].min() if not hist_df.empty else pd.NA
        high = float(h_max) if not pd.isna(h_max) else quote["price"] * 1.05
        low = float(l_min) if not pd.isna(l_min) else quote["price"] * 0.95
    except Exception:
        currency = quote.get("currency", "USD")
        high = quote["price"] * 1.05
        low = quote["price"] * 0.95

    price = float(quote["price"])
    direction = 1 if sentiment > 6 else -1 if sentiment < 4 else 0
    predicted_price = price * (1 + (direction * 0.005))

    return jsonify(
        {
            "success": True,
            "ticker": ticker,
            "data": {
                "currentPrice": float(price),
                "previousClose": safe_float(quote.get("previous_close", price)),
                "change": safe_float(quote.get("change", 0)),
                "changePercent": safe_float(quote.get("change_percent", 0)),
                "dayHigh": safe_float(quote.get("day_high", high)),
                "dayLow": safe_float(quote.get("day_low", low)),
                "volume": int(safe_float(quote.get("volume", 0))),
                "currency": currency,
                "sentiment": safe_float(sentiment),
                "predictedPrice": round(float(predicted_price), 2),
                "high52Week": float(high),
                "low52Week": float(low),
            },
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/history/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_history(ticker):
    if not data_fetcher_available:
        return jsonify({"error": "Data fetcher unavailable"}), 503

    period = request.args.get("period", "1y")
    interval = request.args.get("interval", "1d")

    fetcher = FinancialDataFetcher(ticker=ticker)
    df, meta = fetcher.fetch_ohlcv(ticker, period=period, interval=interval)

    if df is None or df.empty:
        return jsonify({"error": "No historical data available"}), 404

    df_reset = df.reset_index()
    if "Date" in df_reset.columns:
        dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in df_reset["Date"]]
    else:
        dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in df_reset.index]

    return jsonify(
        {
            "success": True,
            "ticker": ticker,
            "data": {
                "period": period,
                "interval": interval,
                "currency": meta.currency if meta else "USD",
                "count": len(df_reset),
                "dates": dates,
                "open": df_reset["Open"].tolist() if "Open" in df_reset.columns else [],
                "high": df_reset["High"].tolist() if "High" in df_reset.columns else [],
                "low": df_reset["Low"].tolist() if "Low" in df_reset.columns else [],
                "close": df_reset["Close"].tolist() if "Close" in df_reset.columns else [],
                "volume": df_reset["Volume"].tolist() if "Volume" in df_reset.columns else [],
            },
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/metrics/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_metrics(ticker):
    """Get advanced performance metrics (includes returns distribution)."""
    if not modules["metrics"]:
        return jsonify({"error": "Metrics module not available"}), 503
    if not data_fetcher_available:
        return jsonify({"error": "Data fetcher unavailable"}), 503

    fetcher = FinancialDataFetcher(ticker=ticker)
    df, _ = fetcher.fetch_ohlcv(ticker, period="2y")

    if df is None or df.empty or len(df) < 50:
        return jsonify({"error": "Insufficient data for metrics"}), 400

    stats = modules["metrics"].calculate_comprehensive_stats(df, ticker=ticker)
    trade_stats = modules["metrics"].calculate_trade_statistics(df)
    distribution = modules["metrics"].calculate_returns_distribution(df, bins=30)

    logger.info(f"✅ Metrics: {ticker} (Sharpe={safe_float(stats.get('sharpe', 0)):.2f})")

    return jsonify(
        {
            "success": True,
            "ticker": ticker,
            "data": {
                "statistics": stats,
                "tradeStatistics": trade_stats,
                "distribution": distribution,
            },
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/features/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_features(ticker):
    if not modules["features"]:
        return jsonify({"error": "Features module not available"}), 503
    if not data_fetcher_available:
        return jsonify({"error": "Data fetcher unavailable"}), 503

    fetcher = FinancialDataFetcher(ticker=ticker)
    df, _ = fetcher.fetch_ohlcv(ticker, period="6mo")

    if df is None or df.empty:
        return jsonify({"error": "No data available"}), 404

    enriched_df = modules["features"].generate_features(df)
    if enriched_df is None or enriched_df.empty:
        return jsonify({"error": "Feature generation failed"}), 500

    latest = enriched_df.iloc[-1]
    signals = modules["features"].interpret_signals(latest)

    history_cols = [
        "bb_upper", "bb_lower", "bb_middle", "rsi_14",
        "macd", "macd_signal", "macd_histogram", "sma_20", "sma_50",
    ]

    hist_out = {}
    for col in history_cols:
        if col in enriched_df.columns:
            hist_out[col] = [
                round(float(v), 4) if not pd.isna(v) else None for v in enriched_df[col].tail(126)
            ]

    df_reset = enriched_df.reset_index()
    date_col = "Date" if "Date" in df_reset.columns else "index"
    hist_dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in df_reset[date_col].tail(126)]

    hist_data = {
        "dates": hist_dates,
        **hist_out,
        "close": [round(float(v), 2) for v in enriched_df["Close"].tail(126)],
    }

    return jsonify(
        {
            "success": True,
            "ticker": ticker,
            "data": {
                "latest": {
                    "rsi_14": safe_float(latest.get("rsi_14"), 50.0),
                    "macd": safe_float(latest.get("macd"), 0.0),
                    "macd_signal": safe_float(latest.get("macd_signal"), 0.0),
                    "atr_14": safe_float(latest.get("atr_14"), 0.0),
                    "adx": safe_float(latest.get("adx"), 0.0),
                    "obv": safe_float(latest.get("obv"), 0.0),
                    "mfi": safe_float(latest.get("mfi"), 50.0),
                },
                "signals": signals,
                "history": hist_data,
            },
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/options/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_options(ticker):
    """Get options pricing + greeks for both CALL and PUT."""
    if not modules["options"]:
        return jsonify({"error": "Options module not available"}), 503

    try:
        S, K, r, T, sigma, q = modules["options"].load_market_inputs_from_datafetcher(ticker=ticker)

        bs_call = modules["options"].black_scholes_greeks(S, K, r, T, sigma, "call", q)
        bs_put = modules["options"].black_scholes_greeks(S, K, r, T, sigma, "put", q)

        mc_call = modules["options"].mc_european_option_price(S, K, r, T, sigma, "call", q, steps=252, n_paths=50000)
        mc_put = modules["options"].mc_european_option_price(S, K, r, T, sigma, "put", q, steps=252, n_paths=50000)

        binom_call = modules["options"].crr_binomial_price(S, K, r, T, sigma, steps=200, option_type="call", american=False, q=q)
        binom_put = modules["options"].crr_binomial_price(S, K, r, T, sigma, steps=200, option_type="put", american=False, q=q)

        logger.info(f"✅ Options: {ticker} (Call=${bs_call.price:.2f}, Put=${bs_put.price:.2f})")

        return jsonify(
            {
                "success": True,
                "ticker": ticker,
                "data": {
                    "inputs": {
                        "spot_price": round(float(S), 2),
                        "strike_price": round(float(K), 2),
                        "volatility": round(float(sigma) * 100, 2),
                        "time_to_expiry_years": round(float(T), 4),
                        "risk_free_rate": round(float(r) * 100, 2),
                        "dividend_yield": round(float(q) * 100, 2),
                    },
                    "blackscholes": {
                        "call": {
                            "price": round(float(bs_call.price), 2),
                            "delta": round(float(bs_call.delta), 4),
                            "gamma": round(float(bs_call.gamma), 6),
                            "vega": round(float(bs_call.vega), 4),
                            "theta": round(float(bs_call.theta), 4),
                            "rho": round(float(bs_call.rho), 4),
                        },
                        "put": {
                            "price": round(float(bs_put.price), 2),
                            "delta": round(float(bs_put.delta), 4),
                            "gamma": round(float(bs_put.gamma), 6),
                            "vega": round(float(bs_put.vega), 4),
                            "theta": round(float(bs_put.theta), 4),
                            "rho": round(float(bs_put.rho), 4),
                        },
                    },
                    "montecarlo": {
                        "call": {
                            "price": round(float(mc_call["price"]), 2),
                            "confidence_interval_95": [
                                round(float(mc_call["ci"][0]), 2),
                                round(float(mc_call["ci"][1]), 2),
                            ],
                        },
                        "put": {
                            "price": round(float(mc_put["price"]), 2),
                            "confidence_interval_95": [
                                round(float(mc_put["ci"][0]), 2),
                                round(float(mc_put["ci"][1]), 2),
                            ],
                        },
                    },
                    "binomial": {
                        "european_call": round(float(binom_call), 2),
                        "european_put": round(float(binom_put), 2),
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Options error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/forecast/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_forecast(ticker):
    """Simple ARIMA forecast endpoint (10 steps) + historical overlay."""
    if not modules["timeseries"]:
        return jsonify({"error": "Time series module not available"}), 503
    if not data_fetcher_available:
        return jsonify({"error": "Data fetcher unavailable"}), 503

    fetcher = FinancialDataFetcher(ticker=ticker)
    df, _ = fetcher.fetch_ohlcv(ticker, period="2y")

    if df is None or df.empty or len(df) < 100:
        return jsonify({"error": "Insufficient data for forecasting"}), 400

    try:
        arima = modules["timeseries"].ArimaForecaster()
        prices = df["Close"].dropna()
        _, d = arima.check_stationarity(prices)
        forecast, conf_int, _ = arima.fit_predict_arima(prices, order=(5, d, 0), forecast_steps=10)

        last_date = df.index[-1]
        fc_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=10)
        forecast_dates = [d.strftime("%Y-%m-%d") for d in fc_dates]

        hist_closes = df["Close"].tail(60).tolist()
        hist_dates = [str(d.date()) for d in df.index[-60:]]

        logger.info(f"✅ Forecast: {ticker}")

        return jsonify(
            {
                "success": True,
                "ticker": ticker,
                "data": {
                    "historical": {
                        "dates": hist_dates,
                        "prices": [round(float(p), 2) for p in hist_closes],
                    },
                    "arima": {
                        "dates": forecast_dates,
                        "prices": [round(float(p), 2) for p in forecast.tolist()[:10]],
                        "lower": [round(float(x), 2) for x in conf_int.iloc[:10, 0].tolist()],
                        "upper": [round(float(x), 2) for x in conf_int.iloc[:10, 1].tolist()],
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Forecast error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def predict_future_price(ticker):
    """Predict future prices based on selected timeframe (PROTECTED)."""
    if not data_fetcher_available:
        return jsonify({"error": "Data fetcher unavailable"}), 503

    period = request.args.get("period", "1y").lower()
    model_type = request.args.get("model", "auto").lower()

    forecast_map = {
        "1m": 22, "3m": 65, "6m": 130,
        "1y": 252, "2y": 504, "5y": 1260,
    }

    if period not in forecast_map:
        return jsonify({"error": f"Invalid period: {period}", "valid_periods": list(forecast_map.keys())}), 400

    forecast_days = forecast_map[period]

    if model_type == "auto":
        if forecast_days <= 65:
            model_type = "arima"
        elif forecast_days <= 252:
            model_type = "lstm"
        else:
            model_type = "ensemble"

    logger.info(f"🎯 Prediction request: {ticker} for {period} using {model_type}")

    try:
        fetcher = FinancialDataFetcher(ticker=ticker)
        if forecast_days <= 130:
            train_period = "2y"
        elif forecast_days <= 504:
            train_period = "5y"
        else:
            train_period = "max"

        df, meta = fetcher.fetch_ohlcv(ticker, period=train_period)
        if df is None or df.empty or len(df) < 100:
            return jsonify({"error": "Insufficient historical data"}), 400

        current_price = float(df["Close"].iloc[-1])
        last_date = df.index[-1]

        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        forecast_dates_str = [d.strftime("%Y-%m-%d") for d in forecast_dates]

        predictions = []
        conf_lower = []
        conf_upper = []
        model_metrics = {}

        if model_type == "arima":
            if not modules["timeseries"]:
                return jsonify({"error": "Time series module not available"}), 503

            arima_tool = modules["timeseries"].ArimaForecaster()
            prices = df["Close"].dropna()
            _, d = arima_tool.check_stationarity(prices)
            forecast, conf_int, model_fit = arima_tool.fit_predict_arima(
                prices, order=(5, d, 0), forecast_steps=forecast_days
            )

            predictions = forecast.tolist()
            conf_lower = conf_int.iloc[:, 0].tolist()
            conf_upper = conf_int.iloc[:, 1].tolist()

            model_metrics = {
                "model": "ARIMA(5," + str(d) + ",0)",
                "aic": float(model_fit.aic),
                "bic": float(model_fit.bic),
                "confidence_level": 0.95,
            }

        elif model_type == "lstm":
            if not modules["deeplearning"]:
                return jsonify({"error": "Deep learning module not available"}), 503

            lookback = 60
            pipeline = modules["deeplearning"].DeepLearningDataPipeline(
                ticker=ticker, lookback=lookback, forecast_horizon=1, use_multivariate=False,
            )

            (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, _, _ = pipeline.get_data()

            if len(X_train) < 100:
                return jsonify({"error": "Insufficient data for LSTM"}), 400

            input_shape = (X_train.shape[1], X_train.shape[2])
            lstm_model = modules["deeplearning"].build_lstm_model(input_shape)

            lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)

            predictions = []
            last_sequence = X_test[-1:].copy()

            for _ in range(forecast_days):
                pred_scaled = lstm_model.predict(last_sequence, verbose=0)[0, 0]
                pred_price = scaler.inverse_transform(np.array([[pred_scaled] + [0] * (scaler.scale_.shape[0] - 1)]))[0, 0]
                predictions.append(float(pred_price))
                new_point = np.array([[pred_scaled]])
                last_sequence = np.concatenate([last_sequence[:, 1:, :], new_point.reshape(1, 1, 1)], axis=1)

            test_pred = lstm_model.predict(X_test, verbose=0)
            test_actual = scaler.inverse_transform(
                np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaler.scale_.shape[0] - 1))], axis=1)
            )[:, 0]
            test_pred_prices = scaler.inverse_transform(
                np.concatenate([test_pred, np.zeros((test_pred.shape[0], scaler.scale_.shape[0] - 1))], axis=1)
            )[:, 0]

            rmse = float(np.sqrt(np.mean((test_pred_prices - test_actual) ** 2)))
            std_error = rmse

            conf_lower = [p - 1.96 * std_error for p in predictions]
            conf_upper = [p + 1.96 * std_error for p in predictions]

            model_metrics = {
                "model": "LSTM", "lookback_days": lookback, "rmse": round(rmse, 2), "confidence_level": 0.95,
            }

        elif model_type == "ensemble":
            if not modules["timeseries"] or not modules["deeplearning"]:
                return jsonify({"error": "Required modules for ensemble not available"}), 503

            arima_pred = None
            lstm_pred = None

            # 1. Run ARIMA
            try:
                arima_tool = modules["timeseries"].ArimaForecaster()
                prices = df["Close"].dropna()
                _, d = arima_tool.check_stationarity(prices)
                forecast_arima, _, _ = arima_tool.fit_predict_arima(prices, order=(5, d, 0), forecast_steps=forecast_days)
                arima_pred = forecast_arima.tolist()
            except Exception as e:
                logger.warning(f"ARIMA failed in ensemble: {e}")

            # 2. Run LSTM
            try:
                lookback = 60
                pipeline = modules["deeplearning"].DeepLearningDataPipeline(
                    ticker=ticker, lookback=lookback, forecast_horizon=1, use_multivariate=False
                )
                (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, _, _ = pipeline.get_data()

                if len(X_train) >= 100:
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    lstm_model = modules["deeplearning"].build_lstm_model(input_shape)
                    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=0)

                    lstm_pred = []
                    last_sequence = X_test[-1:].copy()

                    for _ in range(forecast_days):
                        pred_scaled = lstm_model.predict(last_sequence, verbose=0)[0, 0]
                        pred_price = scaler.inverse_transform(np.array([[pred_scaled] + [0] * (scaler.scale_.shape[0] - 1)]))[0, 0]
                        lstm_pred.append(float(pred_price))
                        new_point = np.array([[pred_scaled]])
                        last_sequence = np.concatenate([last_sequence[:, 1:, :], new_point.reshape(1, 1, 1)], axis=1)
            except Exception as e:
                logger.warning(f"LSTM failed in ensemble: {e}")

            # 3. Combine
            if arima_pred and lstm_pred:
                predictions = [0.3 * a + 0.7 * l for a, l in zip(arima_pred, lstm_pred)]
                std_ensemble = float(np.std([abs(a - l) for a, l in zip(arima_pred, lstm_pred)]))
                conf_lower = [p - 2.0 * std_ensemble for p in predictions]
                conf_upper = [p + 2.0 * std_ensemble for p in predictions]
                model_metrics = {"model": "Ensemble (30% ARIMA + 70% LSTM)", "confidence_level": 0.95}
            elif lstm_pred:
                predictions = lstm_pred
                std_error = float(np.std(predictions) * 0.1)
                conf_lower = [p - 1.96 * std_error for p in predictions]
                conf_upper = [p + 1.96 * std_error for p in predictions]
                model_metrics = {"model": "LSTM Only (ARIMA failed)", "confidence_level": 0.95}
            elif arima_pred:
                predictions = arima_pred
                std_error = float(np.std(predictions) * 0.1)
                conf_lower = [p - 1.96 * std_error for p in predictions]
                conf_upper = [p + 1.96 * std_error for p in predictions]
                model_metrics = {"model": "ARIMA Only (LSTM failed)", "confidence_level": 0.95}
            else:
                return jsonify({"error": "Ensemble models failed completely"}), 500

        else:
            return jsonify({
                "error": f"Invalid model: {model_type}",
                "valid_models": ["arima", "lstm", "ensemble", "auto"]
            }), 400

        predicted_end_price = predictions[-1]
        price_change = predicted_end_price - current_price
        price_change_percent = (price_change / current_price) * 100

        logger.info(f"✅ Prediction: {ticker} {period} - Current: ${current_price:.2f} → Predicted: ${predicted_end_price:.2f} ({price_change_percent:+.2f}%)")

        return jsonify(
            {
                "success": True,
                "ticker": ticker,
                "data": {
                    "period": period,
                    "forecast_days": forecast_days,
                    "model": model_metrics,
                    "current": {
                        "price": round(current_price, 2),
                        "date": last_date.strftime("%Y-%m-%d"),
                        "currency": meta.currency if meta else "USD",
                    },
                    "prediction": {
                        "end_price": round(predicted_end_price, 2),
                        "price_change": round(price_change, 2),
                        "price_change_percent": round(price_change_percent, 2),
                        "direction": "bullish" if price_change > 0 else "bearish",
                    },
                    "forecast": {
                        "dates": forecast_dates_str,
                        "prices": [round(float(p), 2) for p in predictions],
                        "confidence_lower": [round(float(c), 2) for c in conf_lower] if conf_lower else None,
                        "confidence_upper": [round(float(c), 2) for c in conf_upper] if conf_upper else None,
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/api/ml-models/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_ml_models(ticker):
    """Get traditional ML model predictions (PROTECTED)."""
    if not modules["ml"]:
        return jsonify({"error": "ML module not available"}), 503

    try:
        beta, alpha, r_squared = modules["ml"].calculate_beta_and_correlation(ticker, "^GSPC", period="2y")
        X_train, X_test, y_train, y_test = modules["ml"].prepare_tabular_data_from_fetcher(ticker)

        if len(y_test) == 0:
            return jsonify({"error": "Insufficient data for ML models"}), 400

        svr_pred = modules["ml"].train_svr(X_train, X_test, y_train, y_test)
        tree_preds = modules["ml"].train_trees(X_train, X_test, y_train, y_test)

        logger.info(f"✅ ML Models: {ticker} (Beta={safe_float(beta):.3f})")

        return jsonify(
            {
                "success": True,
                "ticker": ticker,
                "data": {
                    "market_analysis": {
                        "beta": round(float(beta), 4),
                        "alpha": round(float(alpha), 6),
                        "r_squared": round(float(r_squared), 4),
                    },
                    "predictions": {
                        "current_price": round(float(y_test[-1]), 2),
                        "svr": round(float(svr_pred[-1]), 2),
                        "random_forest": round(float(tree_preds.get("RandomForest", [0])[-1]), 2),
                        "xgboost": round(float(tree_preds.get("XGBoost", [0])[-1]), 2) if "XGBoost" in tree_preds else None,
                        "lightgbm": round(float(tree_preds.get("LightGBM", [0])[-1]), 2) if "LightGBM" in tree_preds else None,
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"ML error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/deep-learning/", methods=["GET"])
@login_required
@require_ticker
@safe_api_call
def get_deep_learning(ticker):
    """Get deep learning predictions (PROTECTED)."""
    if not modules["deeplearning"]:
        return jsonify({"error": "Deep learning module not available"}), 503
    if not data_fetcher_available:
        return jsonify({"error": "Data fetcher unavailable"}), 503

    try:
        lookback = 60
        pipeline = modules["deeplearning"].DeepLearningDataPipeline(ticker, lookback)
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, test_dates, _ = pipeline.get_data()

        if len(X_test) == 0:
            return jsonify({"error": "Insufficient data for deep learning"}), 400

        input_shape = (X_train.shape[1], X_train.shape[2])
        lstm_model = modules["deeplearning"].build_lstm_model(input_shape)
        lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            verbose=0,
        )

        predictions = lstm_model.predict(X_test, verbose=0)
        pred_prices = scaler.inverse_transform(
            np.concatenate([predictions, np.zeros((predictions.shape[0], scaler.scale_.shape[0] - 1))], axis=1)
        )[:, 0]

        actual_prices = scaler.inverse_transform(
            np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaler.scale_.shape[0] - 1))], axis=1)
        )[:, 0]

        rmse = float(np.sqrt(np.mean((pred_prices - actual_prices) ** 2)))
        mae = float(np.mean(np.abs(pred_prices - actual_prices)))
        mape = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100)

        logger.info(f"✅ Deep Learning: {ticker} (RMSE={rmse:.2f})")

        return jsonify(
            {
                "success": True,
                "ticker": ticker,
                "data": {
                    "model": "LSTM",
                    "lookback_days": lookback,
                    "metrics": {
                        "rmse": round(rmse, 2),
                        "mae": round(mae, 2),
                        "mape": round(mape, 2),
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Deep Learning error: {e}")
        return jsonify({"error": str(e)}), 400


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found", "path": request.path, "method": request.method}), 404


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({"error": "Internal server error", "details": str(error)}), 500


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🚀 QuantScope AI - Professional Trading Platform v2.1")
    logger.info("=" * 80)
    logger.info("Version: 2026.02.24")
    logger.info(f"Base Directory: {BASE_DIR}")
    logger.info(f"Static Directory: {STATIC_DIR}")
    logger.info(f"Template Directory: {TEMPLATE_DIR}")
    logger.info(f"Authentication: {'✅ ENABLED' if auth_available else '❌ DISABLED'}")
    logger.info("=" * 80)

    if os.path.exists(STATIC_DIR):
        static_files = os.listdir(STATIC_DIR)
        logger.info("📁 Frontend Files (in static/):")
        logger.info(f"  {'✅' if 'index.html' in static_files else '❌'} index.html")
        logger.info(f"  {'✅' if 'style.css' in static_files else '❌'} style.css")
        logger.info(f"  {'✅' if 'script.js' in static_files else '❌'} script.js")

    if auth_available and os.path.exists(TEMPLATE_DIR):
        template_files = os.listdir(TEMPLATE_DIR)
        logger.info("📁 Template Files (in templates/):")
        logger.info(f"  {'✅' if 'login.html' in template_files else '❌'} login.html")
        logger.info(f"  {'✅' if 'signup.html' in template_files else '❌'} signup.html")

    logger.info("=" * 80)
    logger.info("📦 Backend Modules:")
    logger.info(f"  {'✅' if data_fetcher_available else '❌'} data_fetcher (CRITICAL)")
    for module_name, module_obj in modules.items():
        status = "✅" if module_obj else "❌"
        logger.info(f"  {status} {module_name}")
    logger.info("=" * 80)

    PORT = int(os.environ.get("PORT", 8000))
    HOST = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"🌐 Server starting on http://{HOST}:{PORT}")
    logger.info(f"📊 Dashboard: http://localhost:{PORT}/")
    if auth_available:
        logger.info(f"🔐 Login: http://localhost:{PORT}/login")
        logger.info(f"📝 Signup: http://localhost:{PORT}/signup")
        logger.info("=" * 80)
        logger.info("ℹ️  Note: Dashboard visible to all, but analysis requires login")
    logger.info("=" * 80)
    logger.info("💡 Press CTRL+C to stop the server")
    logger.info("=" * 80)

    app.run(debug=False, host=HOST, port=PORT, use_reloader=False, threaded=True)
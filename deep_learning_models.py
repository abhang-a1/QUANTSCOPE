"""
Deep Learning Financial Forecasting Suite v2.0 - PRODUCTION GRADE
--------------------------------------------------------------------------------
Advanced architectures for time series prediction:
1. RNN (Recurrent Neural Network): Basic sequence memory
2. LSTM (Long Short-Term Memory): Gates to handle long-term dependencies
3. GRU (Gated Recurrent Unit): Efficient simplified LSTM
4. Transformer: Multi-head self-attention for global context
5. CNN-LSTM Hybrid: Convolutional feature extraction + LSTM temporal modeling
6. Bidirectional LSTM: Forward + backward sequence processing

Dependencies:
    pip install tensorflow numpy pandas scikit-learn matplotlib yfinance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import warnings
from typing import Tuple, Dict, List
from datetime import datetime

# Sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Custom Data Fetcher
try:
    from data_fetcher import FinancialDataFetcher
    FETCHER_AVAILABLE = True
except ImportError:
    FETCHER_AVAILABLE = False
    import yfinance as yf
    warnings.warn("FinancialDataFetcher not found. Using yfinance fallback.")

# Configure Logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepLearningForecaster")

# GPU Configuration (if available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"✅ GPU Acceleration Enabled: {len(gpus)} GPU(s) detected")
    except RuntimeError as e:
        logger.warning(f"GPU configuration error: {e}")
else:
    logger.info("🖥️  Running on CPU")


# =============================================================================
# 1. ADVANCED DATA PIPELINE WITH MULTI-FEATURE SUPPORT
# =============================================================================

class DeepLearningDataPipeline:
    """
    Enhanced data pipeline with:
    - Multi-feature support (OHLCV + Technical Indicators)
    - Multiple scaling strategies
    - Sequence generation with stride options
    - Train/Val/Test split
    """

    def __init__(
        self,
        ticker: str,
        lookback: int = 60,
        forecast_horizon: int = 1,
        scaler_type: str = 'minmax',
        use_multivariate: bool = False
    ):
        """
        Args:
            ticker: Stock symbol
            lookback: Number of historical days to use
            forecast_horizon: Days ahead to predict
            scaler_type: 'minmax', 'standard', or 'robust'
            use_multivariate: If True, uses OHLCV instead of just Close
        """
        self.ticker = ticker
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.use_multivariate = use_multivariate

        # Select Scaler
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_type}")

        if FETCHER_AVAILABLE:
            self.fetcher = FinancialDataFetcher(ticker=ticker)

    def fetch_data(self, period: str = "5y") -> pd.DataFrame:
        """Fetches historical market data."""
        logger.info(f"📊 Fetching {period} of data for {self.ticker}...")

        try:
            if FETCHER_AVAILABLE:
                df, _ = self.fetcher.fetch_ohlcv(self.ticker, period=period)
            else:
                df = yf.download(self.ticker, period=period, progress=False)
        except Exception as e:
            logger.error(f"❌ Data fetch failed: {e}")
            raise

        if df.empty or len(df) < self.lookback + 50:
            raise ValueError(f"Insufficient data for {self.ticker}")

        return df

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds technical indicators as features."""
        df = df.copy()

        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2

        # Bollinger Bands
        df['SMA_20'] = close.rolling(window=20).mean()
        df['BB_Width'] = (close.rolling(window=20).std() * 2) / (df['SMA_20'] + 1e-10)

        # Volume Indicators
        if 'Volume' in df.columns:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-10)

        df.dropna(inplace=True)
        return df

    def create_sequences(
        self,
        data: np.ndarray,
        lookback: int,
        forecast_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sliding window sequences.

        Args:
            data: Scaled data array
            lookback: Past time steps
            forecast_horizon: Future steps to predict

        Returns:
            X: Input sequences [samples, lookback, features]
            y: Target values [samples, forecast_horizon]
        """
        X, y = [], []

        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i - lookback:i])
            # For multi-step, take average or last value
            if forecast_horizon == 1:
                y.append(data[i, 0])  # Predict Close price (first column)
            else:
                y.append(data[i:i + forecast_horizon, 0])

        return np.array(X), np.array(y)

    def get_data(self) -> Tuple:
        """
        Main pipeline: Fetch → Feature Engineering → Scale → Sequences → Split

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, test_dates, feature_names
        """
        # 1. Fetch Data
        df = self.fetch_data()

        # 2. Feature Engineering
        if self.use_multivariate:
            df = self.add_technical_features(df)
            feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'BB_Width']
            feature_cols = [col for col in feature_cols if col in df.columns]
            data = df[feature_cols].values
        else:
            data = df['Close'].values.reshape(-1, 1)
            feature_cols = ['Close']

        # 3. Scale Data
        scaled_data = self.scaler.fit_transform(data)

        # 4. Create Sequences
        X, y = self.create_sequences(scaled_data, self.lookback, self.forecast_horizon)

        logger.info(f"✅ Generated {len(X)} sequences with shape {X.shape}")

        # 5. Split: 70% Train, 15% Val, 15% Test
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

        # Store dates for visualization
        test_start_idx = self.lookback + train_size + val_size
        test_dates = df.index[test_start_idx:test_start_idx + len(X_test)]

        logger.info(f"📦 Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), self.scaler, test_dates, feature_cols


# =============================================================================
# 2. ENHANCED MODEL FACTORY
# =============================================================================

def build_rnn_model(input_shape: Tuple) -> keras.Model:
    """
    ✅ RNN: Simple Recurrent Neural Network
    Best for: Short-term patterns, quick training
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(64, return_sequences=True, activation='tanh'),
        layers.Dropout(0.3),
        layers.SimpleRNN(64, return_sequences=False, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="RNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',  # More robust to outliers than MSE
        metrics=['mae']
    )
    return model


def build_lstm_model(input_shape: Tuple) -> keras.Model:
    """
    ✅ LSTM: Long Short-Term Memory
    Best for: Long-term dependencies, industry standard
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(100, return_sequences=True, activation='tanh'),
        layers.Dropout(0.3),
        layers.LSTM(100, return_sequences=True, activation='tanh'),
        layers.Dropout(0.3),
        layers.LSTM(50, return_sequences=False, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model


def build_gru_model(input_shape: Tuple) -> keras.Model:
    """
    ✅ GRU: Gated Recurrent Unit
    Best for: Faster training, similar performance to LSTM
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(100, return_sequences=True, activation='tanh'),
        layers.Dropout(0.3),
        layers.GRU(100, return_sequences=True, activation='tanh'),
        layers.Dropout(0.3),
        layers.GRU(50, return_sequences=False, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="GRU")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model


def build_bidirectional_lstm_model(input_shape: Tuple) -> keras.Model:
    """
    ✅ Bidirectional LSTM: Processes sequence forward AND backward
    Best for: Capturing patterns from both past and future context
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(100, return_sequences=True, activation='tanh')),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(50, return_sequences=False, activation='tanh')),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="BiLSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model


def build_cnn_lstm_model(input_shape: Tuple) -> keras.Model:
    """
    ✅ CNN-LSTM Hybrid: Convolutional layers extract features, LSTM models temporal
    Best for: Complex patterns with local + global dependencies
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(100, return_sequences=True, activation='tanh'),
        layers.Dropout(0.3),
        layers.LSTM(50, return_sequences=False, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ], name="CNN_LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model


def build_transformer_model(input_shape: Tuple, head_size: int = 128, num_heads: int = 4, ff_dim: int = 128) -> keras.Model:
    """
    ✅ Transformer: Multi-head self-attention mechanism
    Best for: Capturing long-range dependencies, parallel processing
    """
    inputs = layers.Input(shape=input_shape)

    # Multi-Head Attention
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=0.2
    )(inputs, inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed-Forward Network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=input_shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x + res

    # Second Attention Block (Stacked Transformer)
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=0.2
    )(x, x)
    x = layers.Dropout(0.2)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Global Pooling + Dense Layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='huber',
        metrics=['mae']
    )
    return model


# =============================================================================
# 3. TRAINING ENGINE WITH ADVANCED CALLBACKS
# =============================================================================

def train_and_evaluate(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    epochs: int = 100,
    batch_size: int = 32
) -> Dict:
    """
    Trains model with early stopping and learning rate scheduling.
    """
    logger.info(f"\n🚀 Training {model.name}...")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Predict on Test Set
    predictions = model.predict(X_test, verbose=0)

    # Inverse Transform (back to original scale)
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)

    # Only inverse transform the first column (Close price)
    predictions_actual = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], scaler.scale_.shape[0] - 1))], axis=1)
    )[:, 0]

    y_test_actual = scaler.inverse_transform(
        np.concatenate([y_test, np.zeros((y_test.shape[0], scaler.scale_.shape[0] - 1))], axis=1)
    )[:, 0]

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    r2 = r2_score(y_test_actual, predictions_actual)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_actual - predictions_actual) / (y_test_actual + 1e-10))) * 100

    logger.info(f"✅ {model.name} - RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")

    return {
        "model_name": model.name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "predictions": predictions_actual,
        "actuals": y_test_actual,
        "history": history.history
    }


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_results(results: List[Dict], test_dates: pd.DatetimeIndex, ticker: str):
    """Enhanced visualization with subplots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot 1: Predictions vs Actuals
    ax1.plot(test_dates, results[0]['actuals'], label="Actual Price", color='black', linewidth=2.5, alpha=0.8)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
    for i, res in enumerate(results):
        ax1.plot(
            test_dates,
            res['predictions'],
            label=f"{res['model_name']} (RMSE={res['rmse']:.2f})",
            color=colors[i % len(colors)],
            alpha=0.7,
            linewidth=1.5
        )

    ax1.set_title(f"{ticker} Price Prediction - Deep Learning Model Comparison", fontsize=18, fontweight='bold')
    ax1.set_ylabel("Price ($)", fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Prediction Errors
    for i, res in enumerate(results):
        errors = res['predictions'] - res['actuals']
        ax2.plot(test_dates, errors, label=res['model_name'], color=colors[i % len(colors)], alpha=0.6)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title("Prediction Errors", fontsize=14)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Error ($)", fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Deep Learning Financial Forecaster v2.0 - Production Grade")
    print("=" * 80)

    # Configuration
    TICKER = "^NSEI"
    LOOKBACK = 120
    FORECAST_HORIZON = 1
    EPOCHS = 100
    BATCH_SIZE = 32
    USE_MULTIVARIATE = False  # Set to True for multi-feature input

    # 1. Initialize Pipeline
    pipeline = DeepLearningDataPipeline(
        ticker=TICKER,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON,
        scaler_type='minmax',
        use_multivariate=USE_MULTIVARIATE
    )

    # 2. Prepare Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, test_dates, features = pipeline.get_data()

    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"📐 Input Shape: {input_shape} | Features: {features}")

    # 3. Build Models
    models = [
        build_rnn_model(input_shape),
        build_lstm_model(input_shape),
        build_gru_model(input_shape),
        build_bidirectional_lstm_model(input_shape),
        build_cnn_lstm_model(input_shape),
        build_transformer_model(input_shape)
    ]

    results = []

    # 4. Train & Evaluate
    for model in models:
        try:
            res = train_and_evaluate(
                model, X_train, y_train, X_val, y_val, X_test, y_test, scaler,
                epochs=EPOCHS, batch_size=BATCH_SIZE
            )
            results.append(res)
        except Exception as e:
            logger.error(f"❌ {model.name} failed: {e}")

    # 5. Visualize Results
    if results:
        plot_results(results, test_dates, TICKER)

        # 6. Leaderboard
        print("\n" + "=" * 80)
        print(f"📊 FINAL LEADERBOARD ({TICKER})")
        print("=" * 80)
        print(f"{'Model':<20} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10} | {'MAPE':<10}")
        print("-" * 80)
        for res in sorted(results, key=lambda x: x['rmse']):
            print(f"{res['model_name']:<20} | {res['rmse']:<10.2f} | {res['mae']:<10.2f} | {res['r2']:<10.4f} | {res['mape']:<10.2f}%")
        print("=" * 80)

        # Best Model
        best_model = min(results, key=lambda x: x['rmse'])
        logger.info(f"🏆 Best Model: {best_model['model_name']} with RMSE={best_model['rmse']:.2f}")

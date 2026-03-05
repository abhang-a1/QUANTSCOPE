import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.getenv('DEBUG', False)
    
    # Data settings
    DEFAULT_HISTORICAL_DAYS = 252  # 1 year of trading days
    MAX_HISTORICAL_DAYS = 1000
    DEFAULT_PREDICTION_DAYS = 5
    MAX_PREDICTION_DAYS = 30
    
    # Model settings
    LSTM_SEQUENCE_LENGTH = 60
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    XGBOOST_N_ESTIMATORS = 200
    XGBOOST_MAX_DEPTH = 6
    XGBOOST_LEARNING_RATE = 0.05
    
    # Feature engineering
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    VOLATILITY_WINDOW = 20
    
    # API settings
    API_TIMEOUT = 300  # 5 minutes
    CACHE_TIMEOUT = 3600  # 1 hour
    
    # File paths
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_cache')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

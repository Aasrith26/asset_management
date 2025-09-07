# 4-ASSET BACKEND CONFIGURATION
# =============================
# Configuration settings for the complete backend system

import os
from typing import Dict, Any
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"  
    PRODUCTION = "production"

class BackendConfig:
    """Main configuration for 4-asset backend"""
    
    # Environment
    ENVIRONMENT = Environment(os.getenv("ENVIRONMENT", "development"))
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_RELOAD = ENVIRONMENT == Environment.DEVELOPMENT
    
    # Timeout Settings
    DEFAULT_ANALYSIS_TIMEOUT = int(os.getenv("ANALYSIS_TIMEOUT_MINUTES", 15))
    MAX_ANALYSIS_TIMEOUT = int(os.getenv("MAX_ANALYSIS_TIMEOUT_MINUTES", 30))
    
    # Asset-Specific Timeouts
    ASSET_TIMEOUTS = {
        "NIFTY50": int(os.getenv("NIFTY_TIMEOUT_SECONDS", 300)),      # 5 minutes
        "GOLD": int(os.getenv("GOLD_TIMEOUT_SECONDS", 400)),          # 6.5 minutes
        "BITCOIN": int(os.getenv("BITCOIN_TIMEOUT_SECONDS", 180)),    # 3 minutes
        "REIT": int(os.getenv("REIT_TIMEOUT_SECONDS", 600))           # 10 minutes
    }
    
    # Data Sources
    TRADING_ECONOMICS_API_KEY = os.getenv("TE_API_KEY", "guest:guest")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    
    # File Storage
    RESULTS_DIRECTORY = os.getenv("RESULTS_DIR", "analysis_results")
    LOG_DIRECTORY = os.getenv("LOG_DIR", "logs")
    
    # Database (if needed in future)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sentiment_analysis.db")
    
    # Redis (for caching, if needed)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 300))  # 5 minutes
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 3600))  # 1 hour
    
    # Security
    API_KEY_HEADER = "X-API-Key"
    ALLOWED_API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    
    # Portfolio Settings
    DEFAULT_PORTFOLIO_WEIGHTS = {
        "NIFTY50": 0.40,   # 40% Indian equity
        "GOLD": 0.25,      # 25% safe haven
        "BITCOIN": 0.20,   # 20% crypto exposure
        "REIT": 0.15       # 15% real estate
    }
    
    # Asset-Specific Configurations
    NIFTY_CONFIG = {
        "symbols": ["^NSEI", "NIFTY_50.NS"],
        "kpis": 5,
        "update_frequency": "15min"
    }
    
    GOLD_CONFIG = {
        "symbols": ["GLD", "GOLD", "GC=F"],
        "kpis": 5,
        "update_frequency": "30min"
    }
    
    BITCOIN_CONFIG = {
        "symbols": ["BTC-USD", "BTCUSD"],
        "exchange": "kraken",
        "kpis": 4,
        "update_frequency": "1min"
    }
    
    REIT_CONFIG = {
        "symbols": ["MINDSPACE.NS", "EMBASSY.NS", "BIRET.NS"],
        "kpis": 4,
        "update_frequency": "daily"
    }
    
    @classmethod
    def get_asset_config(cls, asset_type: str) -> Dict[str, Any]:
        """Get configuration for specific asset"""
        config_map = {
            "NIFTY50": cls.NIFTY_CONFIG,
            "GOLD": cls.GOLD_CONFIG,
            "BITCOIN": cls.BITCOIN_CONFIG,
            "REIT": cls.REIT_CONFIG
        }
        return config_map.get(asset_type, {})
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        import os
        os.makedirs(cls.RESULTS_DIRECTORY, exist_ok=True)
        os.makedirs(cls.LOG_DIRECTORY, exist_ok=True)
        
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": f"{cls.LOG_DIRECTORY}/backend.log",
                },
            },
            "loggers": {
                "": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }

# Environment-specific configurations
if BackendConfig.ENVIRONMENT == Environment.PRODUCTION:
    # Production settings
    BackendConfig.API_RELOAD = False
    BackendConfig.DEFAULT_ANALYSIS_TIMEOUT = 10  # Shorter timeout in prod
    
elif BackendConfig.ENVIRONMENT == Environment.STAGING:
    # Staging settings
    BackendConfig.API_RELOAD = False
    BackendConfig.DEFAULT_ANALYSIS_TIMEOUT = 12
    
# Create directories on import
BackendConfig.setup_directories()

# Export commonly used configs
ASSET_TIMEOUT_MAP = BackendConfig.ASSET_TIMEOUTS
PORTFOLIO_WEIGHTS = BackendConfig.DEFAULT_PORTFOLIO_WEIGHTS
API_CONFIG = {
    "host": BackendConfig.API_HOST,
    "port": BackendConfig.API_PORT,
    "reload": BackendConfig.API_RELOAD
}
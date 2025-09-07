
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "sentiment_analysis"
    username: str = "sentiment_user"
    password: str = os.getenv("DB_PASSWORD", "your_password")

@dataclass
class RedisConfig:
    """Redis configuration for caching and task queue"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = os.getenv("REDIS_PASSWORD", "")

@dataclass
class IntegrationConfig:
    """Configuration for downstream system integration"""
    
    # Model A (CSV Processor) - Update with your actual endpoints
    csv_processor_url: str = "http://localhost:8001/process-csv"
    csv_processor_timeout: int = 30
    csv_processor_retry_attempts: int = 3
    
    # Model B (JSON Context Processor) - Update with your actual endpoints  
    json_processor_url: str = "http://localhost:8002/process-context"
    json_processor_timeout: int = 45
    json_processor_retry_attempts: int = 3
    
    # Backup storage (optional)
    backup_storage_url: str = os.getenv("BACKUP_STORAGE_URL", "")
    
    # Webhook notifications (optional)
    completion_webhook_url: str = os.getenv("COMPLETION_WEBHOOK_URL", "")

class SystemConfig:
    """Main system configuration"""
    
    # Application settings
    APP_NAME = "Multi-Asset Sentiment Analysis Orchestrator"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "1"))
    
    # File storage paths
    BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
    CSV_DIR = BASE_DATA_DIR / "csv"
    JSON_DIR = BASE_DATA_DIR / "json"
    LOGS_DIR = BASE_DATA_DIR / "logs"
    TEMP_DIR = BASE_DATA_DIR / "temp"
    ARCHIVE_DIR = BASE_DATA_DIR / "archive"
    
    # Performance settings
    MAX_PARALLEL_ASSETS = int(os.getenv("MAX_PARALLEL_ASSETS", "10"))
    DEFAULT_TIMEOUT_MINUTES = int(os.getenv("DEFAULT_TIMEOUT_MINUTES", "15"))
    MAX_TIMEOUT_MINUTES = int(os.getenv("MAX_TIMEOUT_MINUTES", "60"))
    
    # Retry and reliability
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
    RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "5"))
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    
    # File management
    CLEANUP_AFTER_HOURS = int(os.getenv("CLEANUP_AFTER_HOURS", "24"))
    ARCHIVE_AFTER_DAYS = int(os.getenv("ARCHIVE_AFTER_DAYS", "7"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    
    # Monitoring and logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
    
    # Asset-specific settings
    ASSET_CONFIGS = {
        "NIFTY50": {
            "timeout_minutes": 15,
            "cache_ttl_minutes": 60,
            "priority": "high",
            "max_retries": 3
        },
        "GOLD": {
            "timeout_minutes": 20,
            "cache_ttl_minutes": 30,
            "priority": "high", 
            "max_retries": 3
        },
        "CRYPTO": {  # Future asset
            "timeout_minutes": 10,
            "cache_ttl_minutes": 15,
            "priority": "normal",
            "max_retries": 2
        },
        "FOREX": {  # Future asset
            "timeout_minutes": 12,
            "cache_ttl_minutes": 20,
            "priority": "normal",
            "max_retries": 2
        }
    }
    
    # Component configurations
    database = DatabaseConfig()
    redis = RedisConfig()
    integration = IntegrationConfig()
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.BASE_DATA_DIR,
            cls.CSV_DIR,
            cls.JSON_DIR,
            cls.LOGS_DIR,
            cls.TEMP_DIR,
            cls.ARCHIVE_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Created directory structure at {cls.BASE_DATA_DIR}")
    
    @classmethod
    def get_asset_config(cls, asset: str) -> Dict:
        """Get configuration for specific asset"""
        return cls.ASSET_CONFIGS.get(asset.upper(), {
            "timeout_minutes": cls.DEFAULT_TIMEOUT_MINUTES,
            "cache_ttl_minutes": 30,
            "priority": "normal",
            "max_retries": 2
        })
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check required environment variables
        if not os.getenv("DB_PASSWORD"):
            issues.append("DB_PASSWORD not set")
        
        # Check directory permissions
        try:
            cls.ensure_directories()
        except PermissionError:
            issues.append(f"Cannot create directories at {cls.BASE_DATA_DIR}")
        
        # Validate timeout settings
        if cls.DEFAULT_TIMEOUT_MINUTES > cls.MAX_TIMEOUT_MINUTES:
            issues.append("DEFAULT_TIMEOUT_MINUTES exceeds MAX_TIMEOUT_MINUTES")
        
        # Check integration URLs
        if not cls.integration.csv_processor_url.startswith("http"):
            issues.append("Invalid CSV processor URL")
        
        if not cls.integration.json_processor_url.startswith("http"):
            issues.append("Invalid JSON processor URL")
        
        return issues

# Environment-specific configurations
class DevelopmentConfig(SystemConfig):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    API_WORKERS = 1
    
class ProductionConfig(SystemConfig):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    API_WORKERS = int(os.getenv("API_WORKERS", "4"))
    
    # Enhanced production settings
    CLEANUP_AFTER_HOURS = 6  # More aggressive cleanup
    MAX_PARALLEL_ASSETS = 20  # Higher concurrency
    
class TestConfig(SystemConfig):
    """Testing environment configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    
    # Use test directories
    BASE_DATA_DIR = Path("./test_data")
    
    # Faster timeouts for testing
    DEFAULT_TIMEOUT_MINUTES = 5
    CLEANUP_AFTER_HOURS = 1

# Configuration factory
def get_config():
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()

# Load configuration
config = get_config()

# Validation on import
validation_issues = config.validate_config()
if validation_issues:
    print("⚠️ Configuration Issues Found:")
    for issue in validation_issues:
        print(f"   • {issue}")
else:
    print("✅ Configuration validation passed")

# Export commonly used settings
CSV_PROCESSOR_URL = config.integration.csv_processor_url
JSON_PROCESSOR_URL = config.integration.json_processor_url
DATA_DIR = config.BASE_DATA_DIR
TIMEOUT_MINUTES = config.DEFAULT_TIMEOUT_MINUTES
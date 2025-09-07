# STARTUP SCRIPT FOR 4-ASSET BACKEND
# ===================================
# Start the complete 4-asset sentiment analysis backend

import uvicorn
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backend_config import BackendConfig, API_CONFIG
from main_4asset_backend import app

def setup_logging():
    """Setup logging configuration"""
    logging_config = BackendConfig.get_logging_config()
    logging.config.dictConfig(logging_config)

def main():
    """Main startup function"""
    print("🚀 Starting 4-Asset Sentiment Analysis Backend")
    print("=" * 50)
    print(f"Environment: {BackendConfig.ENVIRONMENT.value}")
    print(f"Host: {API_CONFIG['host']}")
    print(f"Port: {API_CONFIG['port']}")
    print(f"Reload: {API_CONFIG['reload']}")
    print("")
    print("📊 Assets supported:")
    print("  • NIFTY50 (5 KPIs)")
    print("  • GOLD (5 KPIs)")
    print("  • BITCOIN (4 KPIs)")
    print("  • REIT (4 KPIs)")
    print("")
    print("🌐 API Endpoints:")
    print("  • GET  /health - Health check")
    print("  • GET  /assets - List all assets")
    print("  • POST /analyze/all-assets - Analyze all 4 assets")
    print("  • POST /analyze/category/traditional - Nifty + Gold")
    print("  • POST /analyze/category/alternative - Bitcoin + REIT")
    print("")
    
    # Setup logging
    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("🎯 Backend starting up...")
    except Exception as e:
        print(f"⚠️ Logging setup failed: {e}")
    
    # Start the server
    try:
        uvicorn.run(
            "main_4asset_backend:app",
            host=API_CONFIG['host'],
            port=API_CONFIG['port'],
            reload=API_CONFIG['reload'],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Backend shutdown requested")
    except Exception as e:
        print(f"❌ Backend startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# SETUP & EXECUTION GUIDE - 4-Asset System
# =========================================
# Complete setup guide for Bitcoin & REIT integration
# Step-by-step instructions to get everything working

# 📋 QUICK SETUP CHECKLIST
"""
✅ Directory Structure Setup
✅ Dependencies Installation  
✅ Component File Placement
✅ Integration Updates
✅ Testing & Validation
✅ Production Deployment
"""

import os
import subprocess
import sys
from pathlib import Path

class SetupManager:
    """Setup manager for 4-asset system"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.required_dirs = [
            'bitcoin_kpis',
            'reit_kpis',
            'nifty_50_kpis',  # Existing
            'gold_kpis',      # Existing
            'orchestrator',   # Existing
            'data/csv',
            'data/json',
            'data/logs'
        ]
        
    def create_directory_structure(self):
        """Create required directory structure"""
        print("📁 Creating Directory Structure...")
        
        for dir_path in self.required_dirs:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {dir_path}")
        
        # Create __init__.py files
        for kpi_dir in ['bitcoin_kpis', 'reit_kpis', 'nifty_50_kpis', 'gold_kpis']:
            init_file = self.base_dir / kpi_dir / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"  ✅ Created: {kpi_dir}/__init__.py")
    
    def install_dependencies(self):
        """Install required Python packages"""
        print("📦 Installing Dependencies...")
        
        dependencies = [
            'fastapi>=0.104.0',
            'uvicorn>=0.24.0',
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'requests>=2.31.0',
            'aiofiles>=23.0.0',
            'httpx>=0.25.0',
            'yfinance>=0.2.0',
            'python-dateutil>=2.8.2',
            'certifi>=2023.0.0',
            'urllib3>=2.0.0',
            'pydantic>=2.0.0'
        ]
        
        for dep in dependencies:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                print(f"  ✅ Installed: {dep}")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️ Failed to install {dep}: {e}")
    
    def validate_file_placement(self):
        """Validate all required files are in place"""
        print("🔍 Validating File Placement...")
        
        required_files = {
            'bitcoin_kpis': [
                'btc_micro_momentum.py',
                'btc_funding_basis.py',
                'btc_liquidity.py',
                'btc_orderflow.py',
                'bitcoin_sentiment_analyzer.py'
            ],
            'reit_kpis': [
                'reit_technical_momentum.py',
                'reit_yield_spread.py',
                'reit_accumulation_flow.py',
                'reit_liquidity_risk.py',
                'reit_sentiment_analyzer.py'
            ],
            '.': [
                'enhanced_asset_integration.py',
                'comprehensive_test_suite.py',
                'enhanced_4asset_orchestrator.py'
            ]
        }
        
        missing_files = []
        
        for directory, files in required_files.items():
            dir_path = self.base_dir if directory == '.' else self.base_dir / directory
            
            for file in files:
                file_path = dir_path / file
                if file_path.exists():
                    print(f"  ✅ Found: {directory}/{file}")
                else:
                    print(f"  ❌ Missing: {directory}/{file}")
                    missing_files.append(f"{directory}/{file}")
        
        if missing_files:
            print(f"\n⚠️ Missing Files: {len(missing_files)}")
            return False
        else:
            print(f"\n✅ All files validated successfully!")
            return True

def print_setup_instructions():
    """Print detailed setup instructions"""
    print("🚀 4-ASSET SYSTEM SETUP GUIDE")
    print("=" * 50)
    
    print("\n📋 STEP 1: FILE PLACEMENT")
    print("-" * 30)
    print("Place the following files in your project directory:")
    print("")
    print("bitcoin_kpis/")
    print("├── btc_micro_momentum.py")
    print("├── btc_funding_basis.py")  
    print("├── btc_liquidity.py")
    print("├── btc_orderflow.py")
    print("└── bitcoin_sentiment_analyzer.py")
    print("")
    print("reit_kpis/")
    print("├── reit_technical_momentum.py")
    print("├── reit_yield_spread.py")
    print("├── reit_accumulation_flow.py")
    print("├── reit_liquidity_risk.py")
    print("└── reit_sentiment_analyzer.py")
    print("")
    print("Root directory:")
    print("├── enhanced_asset_integration.py")
    print("├── comprehensive_test_suite.py")
    print("└── enhanced_4asset_orchestrator.py")
    
    print("\n📦 STEP 2: DEPENDENCIES")
    print("-" * 30)
    print("Run: pip install -r requirements.txt")
    print("Or manually install:")
    print("• fastapi uvicorn pandas numpy requests")
    print("• yfinance aiofiles httpx certifi")
    
    print("\n🧪 STEP 3: TESTING")
    print("-" * 30)
    print("Test Bitcoin components:")
    print("  python comprehensive_test_suite.py bitcoin")
    print("")
    print("Test REIT components:")
    print("  python comprehensive_test_suite.py reit")
    print("")
    print("Quick integration test:")
    print("  python comprehensive_test_suite.py quick")
    print("")
    print("Full system test:")
    print("  python comprehensive_test_suite.py")
    
    print("\n🔧 STEP 4: INTEGRATION")
    print("-" * 30)
    print("Update your existing asset_integration.py:")
    print("1. Replace with enhanced_asset_integration.py")
    print("2. Add Bitcoin/REIT imports")
    print("3. Update AssetType enum")
    
    print("\n🚀 STEP 5: PRODUCTION")
    print("-" * 30)
    print("Start enhanced orchestrator:")
    print("  python enhanced_4asset_orchestrator.py")
    print("")
    print("Or integrate with existing orchestrator:")
    print("  python fixed_orchestrator.py")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """# 4-Asset Sentiment Analysis System Requirements
# ===============================================

# Core Framework
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.2

# HTTP & Networking  
requests>=2.31.0
httpx>=0.25.0
aiofiles>=23.0.0
certifi>=2023.0.0
urllib3>=2.0.0

# Financial Data
yfinance>=0.2.0

# Optional: Enhanced Features
# trading-economics>=0.1.0  # Requires API key
# asyncio-throttle>=1.0.0   # For rate limiting

# Development & Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0

# Production
gunicorn>=21.0.0  # For production deployment
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def generate_test_commands():
    """Generate test command reference"""
    commands = """# TESTING COMMANDS REFERENCE
# ===========================

# Individual Component Tests
python -c "import asyncio; from bitcoin_kpis.btc_micro_momentum import test_micro_momentum; asyncio.run(test_micro_momentum())"
python -c "import asyncio; from bitcoin_kpis.btc_funding_basis import test_funding_basis; asyncio.run(test_funding_basis())"
python -c "import asyncio; from reit_kpis.reit_technical_momentum import test_technical_momentum; asyncio.run(test_technical_momentum())"
python -c "import asyncio; from reit_kpis.reit_yield_spread import test_yield_spread; asyncio.run(test_yield_spread())"

# Integration Tests
python -c "import asyncio; from bitcoin_kpis.bitcoin_sentiment_analyzer import bitcoin_performance_test; asyncio.run(bitcoin_performance_test())"
python -c "import asyncio; from reit_kpis.reit_sentiment_analyzer import reit_performance_test; asyncio.run(reit_performance_test())"

# Full System Tests
python comprehensive_test_suite.py bitcoin    # Bitcoin only
python comprehensive_test_suite.py reit      # REIT only  
python comprehensive_test_suite.py quick     # Quick integration
python comprehensive_test_suite.py          # Full test suite

# Integration Tests
python -c "import asyncio; from enhanced_asset_integration import test_all_integrations; asyncio.run(test_all_integrations())"

# Production Orchestrator Test
python enhanced_4asset_orchestrator.py
"""
    
    with open('test_commands.sh', 'w') as f:
        f.write(commands)
    
    print("✅ Created test_commands.sh")

def print_postman_setup():
    """Print Postman testing setup"""
    print("\n📡 POSTMAN TESTING SETUP")
    print("=" * 30)
    
    endpoints = [
        {
            "method": "GET",
            "url": "http://localhost:8000/health",
            "description": "Health check - should show 4 assets"
        },
        {
            "method": "GET", 
            "url": "http://localhost:8000/assets",
            "description": "List all assets and categories"
        },
        {
            "method": "POST",
            "url": "http://localhost:8000/analyze/all-assets",
            "body": {
                "assets": ["NIFTY50", "GOLD", "BITCOIN", "REIT"],
                "timeout_minutes": 15,
                "include_portfolio_view": True
            },
            "description": "Analyze all 4 assets"
        },
        {
            "method": "POST",
            "url": "http://localhost:8000/analyze/category/traditional",
            "body": {
                "timeout_minutes": 15
            },
            "description": "Analyze traditional assets (Nifty + Gold)"
        },
        {
            "method": "POST",
            "url": "http://localhost:8000/analyze/category/alternative", 
            "body": {
                "timeout_minutes": 15
            },
            "description": "Analyze alternative assets (Bitcoin + REIT)"
        }
    ]
    
    for i, endpoint in enumerate(endpoints, 1):
        print(f"\n{i}. {endpoint['method']} {endpoint['url']}")
        print(f"   Description: {endpoint['description']}")
        if 'body' in endpoint:
            print(f"   Body: {endpoint['body']}")

def main():
    """Main setup execution"""
    print_setup_instructions()
    
    print(f"\n🛠️ AUTOMATED SETUP")
    print("=" * 20)
    
    setup = SetupManager()
    
    # Create directory structure
    setup.create_directory_structure()
    
    # Create requirements file
    create_requirements_file()
    
    # Generate test commands
    generate_test_commands()
    
    # Validate file placement
    if setup.validate_file_placement():
        print("\n✅ SETUP VALIDATION PASSED")
        print("🚀 Ready to test components!")
        
        print("\n📋 NEXT STEPS:")
        print("1. pip install -r requirements.txt")
        print("2. python comprehensive_test_suite.py quick")
        print("3. python enhanced_4asset_orchestrator.py")
        
    else:
        print("\n⚠️ SETUP VALIDATION FAILED")
        print("Please ensure all files are placed correctly")
    
    # Print Postman setup
    print_postman_setup()
    
    print(f"\n🎯 SYSTEM READY FOR:")
    print("✅ Bitcoin sentiment analysis (4 KPIs)")
    print("✅ REIT sentiment analysis (4 KPIs)") 
    print("✅ Unified 4-asset portfolio analysis")
    print("✅ REST API orchestrator")
    print("✅ CSV/JSON output generation")

if __name__ == "__main__":
    main()
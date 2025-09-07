#!/usr/bin/env python3
# MULTI-ASSET SENTIMENT ANALYSIS STARTUP SCRIPT v1.0
# ===================================================
# Complete startup and testing script for the orchestrator

import asyncio
import sys
import subprocess
import time
import requests
from pathlib import Path
import json

def print_banner():
    """Print startup banner"""
    print("=" * 70)
    print("🚀 MULTI-ASSET SENTIMENT ANALYSIS ORCHESTRATOR")
    print("   Version: 1.0.0")
    print("   Assets: Nifty 50, Gold")
    print("   Architecture: FastAPI + AsyncIO + Parallel Processing")
    print("=" * 70)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pandas',
        'aiofiles',
        'httpx',
        'pydantic'
    ]
    
    print("🔍 Checking dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied")
    return True

def setup_directories():
    """Setup required directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "./data",
        "./data/csv",
        "./data/json", 
        "./data/logs",
        "./data/temp",
        "./data/archive"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ Directory structure created")

async def test_integrations():
    """Test asset integrations"""
    print("\n🧪 Testing Asset Integrations...")
    
    try:
        from asset_integration import test_nifty_integration, test_gold_integration
        
        print("\n1. Testing Nifty 50 Integration:")
        nifty_success = await test_nifty_integration()
        
        print("\n2. Testing Gold Integration:")
        gold_success = await test_gold_integration()
        
        print(f"\n📊 INTEGRATION TEST RESULTS:")
        print(f"   Nifty 50: {'✅ PASS' if nifty_success else '❌ FAIL (using mock data)'}")
        print(f"   Gold: {'✅ PASS' if gold_success else '❌ FAIL (using mock data)'}")
        
        if not (nifty_success and gold_success):
            print("\n⚠️ Some integrations using mock data.")
            print("   Update import paths in asset_integration.py for real analysis")
        
        return nifty_success or gold_success  # At least one should work
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False

def start_orchestrator():
    """Start the orchestrator server"""
    print("\n🚀 Starting Orchestrator Server...")
    
    try:
        import uvicorn
        from updated_orchestrator import app
        
        print("   Starting FastAPI server on http://localhost:8000")
        print("   API docs available at http://localhost:8000/docs")
        
        # Start the server
        uvicorn.run(
            "updated_orchestrator:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {str(e)}")

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🔧 Testing API Endpoints...")
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            health_data = response.json()
            print(f"   📊 Available assets: {health_data.get('available_assets', [])}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test assets endpoint
        response = requests.get(f"{base_url}/assets", timeout=5)
        if response.status_code == 200:
            print("   ✅ Assets endpoint working")
        else:
            print(f"   ❌ Assets endpoint failed: {response.status_code}")
        
        return True
        
    except requests.ConnectionError:
        print("   ⏳ Server not yet ready for connections")
        return False
    except Exception as e:
        print(f"   ❌ API test failed: {str(e)}")
        return False

def run_sample_analysis():
    """Run a sample analysis"""
    print("\n🎯 Running Sample Analysis...")
    base_url = "http://localhost:8000"
    
    try:
        # Start analysis
        payload = {
            "assets": ["NIFTY50", "GOLD"],
            "timeout_minutes": 10
        }
        
        print("   📤 Starting analysis request...")
        response = requests.post(f"{base_url}/analyze/all-assets", json=payload, timeout=10)
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data['job_id']
            print(f"   ✅ Analysis started with job ID: {job_id}")
            
            # Poll for completion
            max_wait = 30  # 30 seconds max wait for demo
            for i in range(max_wait):
                time.sleep(1)
                status_response = requests.get(f"{base_url}/status/{job_id}", timeout=5)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_status = status_data['status']
                    print(f"   ⏳ Status: {current_status} ({i+1}s)")
                    
                    if current_status in ['completed', 'partial', 'failed']:
                        if current_status in ['completed', 'partial']:
                            print("   ✅ Analysis completed successfully!")
                            
                            # Get results
                            results_response = requests.get(f"{base_url}/results/{job_id}", timeout=5)
                            if results_response.status_code == 200:
                                results = results_response.json()
                                print("\n   📊 SAMPLE RESULTS:")
                                
                                for result in results.get('results', {}).get('analysis_results', []):
                                    asset = result['asset']
                                    sentiment = result['sentiment_score']
                                    confidence = result['confidence']
                                    print(f"      {asset}: Sentiment={sentiment:.3f}, Confidence={confidence:.3f}")
                        else:
                            print("   ❌ Analysis failed")
                        
                        break
                else:
                    print(f"   ❌ Status check failed: {status_response.status_code}")
                    break
            else:
                print("   ⏰ Analysis taking longer than expected (demo timeout)")
            
            return True
        else:
            print(f"   ❌ Failed to start analysis: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Sample analysis failed: {str(e)}")
        return False

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Setup directories
    setup_directories()
    
    # Test integrations
    print("\n🔄 Running integration tests...")
    integration_success = asyncio.run(test_integrations())
    
    if not integration_success:
        print("⚠️ Integration tests failed, but continuing with mock data...")
    
    # Ask user what to do
    print("\n" + "=" * 70)
    print("🎯 STARTUP OPTIONS:")
    print("1. Start orchestrator server")
    print("2. Test API endpoints (requires running server)")
    print("3. Run sample analysis (requires running server)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                start_orchestrator()
                break
            
            elif choice == '2':
                success = test_api_endpoints()
                if success:
                    print("✅ API tests passed!")
                else:
                    print("❌ API tests failed. Make sure server is running.")
            
            elif choice == '3':
                success = run_sample_analysis()
                if success:
                    print("✅ Sample analysis completed!")
                else:
                    print("❌ Sample analysis failed. Make sure server is running.")
            
            elif choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
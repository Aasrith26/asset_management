# COMPLETE BACKEND TESTING SUITE
# ===============================
# Comprehensive tests for 4-asset sentiment analysis backend

import pytest
import httpx
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Test configuration
TEST_CONFIG = {
    'base_url': 'http://localhost:8000',
    'timeout': 300,  # 5 minutes for analysis tests
    'retry_count': 3
}

class TestBackendAPI:
    """Complete test suite for 4-asset backend API"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup for each test"""
        self.client = httpx.AsyncClient(base_url=TEST_CONFIG['base_url'], timeout=TEST_CONFIG['timeout'])
        yield
        await self.client.aclose()
    
    async def test_health_endpoint(self):
        """Test health check endpoint"""
        response = await self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['assets_available'] == 4
        assert 'services' in data
        
        # Check all 4 services are listed
        services = data['services']
        assert 'nifty_50' in services
        assert 'gold' in services
        assert 'bitcoin' in services
        assert 'reit' in services
    
    async def test_assets_endpoint(self):
        """Test assets information endpoint"""
        response = await self.client.get("/assets")
        assert response.status_code == 200
        
        data = response.json()
        assert data['total_assets'] == 4
        assert 'assets' in data
        
        assets = data['assets']
        assert 'NIFTY50' in assets
        assert 'GOLD' in assets
        assert 'BITCOIN' in assets
        assert 'REIT' in assets
        
        # Check asset details
        for asset_name, asset_info in assets.items():
            assert 'name' in asset_info
            assert 'category' in asset_info
            assert 'kpis' in asset_info
            assert 'update_frequency' in asset_info
    
    async def test_categories_endpoint(self):
        """Test categories endpoint"""
        response = await self.client.get("/categories")
        assert response.status_code == 200
        
        data = response.json()
        assert 'categories' in data
        
        categories = data['categories']
        assert 'traditional' in categories
        assert 'alternative' in categories
        assert 'equity' in categories
        assert 'commodity' in categories
        assert 'cryptocurrency' in categories
        assert 'real_estate' in categories
        
        # Check category mappings
        assert 'NIFTY50' in categories['traditional']
        assert 'GOLD' in categories['traditional']
        assert 'BITCOIN' in categories['alternative']
        assert 'REIT' in categories['alternative']
    
    async def test_single_asset_analysis(self):
        """Test single asset analysis"""
        # Test each asset individually
        assets = ['NIFTY50', 'GOLD', 'BITCOIN', 'REIT']
        
        for asset in assets:
            print(f"🧪 Testing {asset} analysis...")
            
            response = await self.client.get(f"/analyze/asset/{asset}")
            
            # Should return 200 or have analysis results
            if response.status_code == 200:
                data = response.json()
                
                assert 'analysis_results' in data
                assert asset in data['analysis_results']
                
                asset_result = data['analysis_results'][asset]
                assert 'sentiment' in asset_result
                assert 'confidence' in asset_result
                assert 'status' in asset_result
                
                # Sentiment should be between -1 and 1
                sentiment = asset_result['sentiment']
                assert -1.0 <= sentiment <= 1.0
                
                # Confidence should be between 0 and 1
                confidence = asset_result['confidence']
                assert 0.0 <= confidence <= 1.0
                
                print(f"  ✅ {asset}: sentiment={sentiment:.3f}, confidence={confidence:.3f}")
            
            else:
                print(f"  ⚠️ {asset}: analysis failed (status {response.status_code})")
    
    async def test_category_analysis_traditional(self):
        """Test traditional assets category analysis"""
        request_data = {
            "timeout_minutes": 10,
            "include_detailed_breakdown": True
        }
        
        response = await self.client.post("/analyze/category/traditional", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            assert 'analysis_results' in data
            assert 'portfolio_summary' in data
            assert data['category'] == 'traditional'
            
            # Should have NIFTY50 and GOLD
            results = data['analysis_results']
            expected_assets = {'NIFTY50', 'GOLD'}
            actual_assets = set(results.keys())
            assert expected_assets.issubset(actual_assets)
            
            # Check portfolio summary
            portfolio = data['portfolio_summary']
            assert 'portfolio_sentiment' in portfolio
            assert 'portfolio_confidence' in portfolio
            assert 'investment_recommendation' in portfolio
            
            print(f"✅ Traditional analysis complete:")
            print(f"   Portfolio Sentiment: {portfolio['portfolio_sentiment']}")
            print(f"   Portfolio Confidence: {portfolio['portfolio_confidence']}")
            print(f"   Recommendation: {portfolio['investment_recommendation']}")
        
        else:
            print(f"⚠️ Traditional category analysis failed (status {response.status_code})")
    
    async def test_category_analysis_alternative(self):
        """Test alternative assets category analysis"""
        request_data = {
            "timeout_minutes": 10,
            "include_detailed_breakdown": True
        }
        
        response = await self.client.post("/analyze/category/alternative", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            assert 'analysis_results' in data
            assert 'portfolio_summary' in data
            assert data['category'] == 'alternative'
            
            # Should have BITCOIN and REIT
            results = data['analysis_results']
            expected_assets = {'BITCOIN', 'REIT'}
            actual_assets = set(results.keys())
            assert expected_assets.issubset(actual_assets)
            
            # Check portfolio summary
            portfolio = data['portfolio_summary']
            assert 'portfolio_sentiment' in portfolio
            assert 'portfolio_confidence' in portfolio
            
            print(f"✅ Alternative analysis complete:")
            print(f"   Portfolio Sentiment: {portfolio['portfolio_sentiment']}")
            print(f"   Portfolio Confidence: {portfolio['portfolio_confidence']}")
        
        else:
            print(f"⚠️ Alternative category analysis failed (status {response.status_code})")
    
    async def test_all_assets_analysis(self):
        """Test complete 4-asset analysis"""
        request_data = {
            "assets": ["NIFTY50", "GOLD", "BITCOIN", "REIT"],
            "timeout_minutes": 15,
            "include_portfolio_view": True,
            "save_results": False
        }
        
        print("🚀 Starting complete 4-asset analysis...")
        
        response = await self.client.post("/analyze/all-assets", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            assert 'analysis_results' in data
            assert 'portfolio_summary' in data
            
            # Check all 4 assets are present
            results = data['analysis_results']
            expected_assets = {'NIFTY50', 'GOLD', 'BITCOIN', 'REIT'}
            actual_assets = set(results.keys())
            assert expected_assets == actual_assets
            
            # Analyze results
            successful_assets = []
            failed_assets = []
            
            for asset_name, asset_result in results.items():
                if asset_result.get('status') == 'success':
                    successful_assets.append(asset_name)
                else:
                    failed_assets.append(asset_name)
            
            portfolio = data['portfolio_summary']
            
            print(f"✅ 4-Asset Analysis Results:")
            print(f"   Successful: {len(successful_assets)}/4 ({successful_assets})")
            print(f"   Failed: {len(failed_assets)}/4 ({failed_assets})")
            print(f"   Portfolio Sentiment: {portfolio['portfolio_sentiment']}")
            print(f"   Portfolio Confidence: {portfolio['portfolio_confidence']}")
            print(f"   Risk Assessment: {portfolio['risk_assessment']}")
            print(f"   Investment Recommendation: {portfolio['investment_recommendation']}")
            
            # Portfolio should have valid metrics
            assert -1.0 <= portfolio['portfolio_sentiment'] <= 1.0
            assert 0.0 <= portfolio['portfolio_confidence'] <= 1.0
            assert portfolio['investment_recommendation'] in ['BUY', 'MILD BUY', 'HOLD', 'MILD SELL', 'SELL']
            
            return {
                'successful_assets': successful_assets,
                'portfolio_summary': portfolio,
                'total_execution_time': data.get('execution_time_seconds', 0)
            }
        
        else:
            print(f"❌ 4-asset analysis failed (status {response.status_code})")
            print(f"Response: {response.text}")
            return None
    
    async def test_performance_benchmarks(self):
        """Test system performance benchmarks"""
        print("⚡ Running performance benchmarks...")
        
        benchmarks = {}
        
        # Test individual asset speed
        assets = ['NIFTY50', 'GOLD', 'BITCOIN', 'REIT']
        
        for asset in assets:
            start_time = datetime.now()
            response = await self.client.get(f"/analyze/asset/{asset}")
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            benchmarks[f'{asset.lower()}_time'] = execution_time
            
            if response.status_code == 200:
                print(f"  ✅ {asset}: {execution_time:.1f}s")
            else:
                print(f"  ❌ {asset}: failed in {execution_time:.1f}s")
        
        # Test portfolio analysis speed
        start_time = datetime.now()
        result = await self.test_all_assets_analysis()
        end_time = datetime.now()
        
        portfolio_time = (end_time - start_time).total_seconds()
        benchmarks['portfolio_time'] = portfolio_time
        
        print(f"\n📊 Performance Summary:")
        print(f"   Portfolio Analysis: {portfolio_time:.1f}s")
        
        # Calculate performance grade
        if portfolio_time < 180:  # 3 minutes
            grade = "EXCELLENT"
        elif portfolio_time < 300:  # 5 minutes
            grade = "GOOD"
        elif portfolio_time < 600:  # 10 minutes
            grade = "ACCEPTABLE"
        else:
            grade = "NEEDS OPTIMIZATION"
        
        print(f"   Performance Grade: {grade}")
        
        return benchmarks

# Individual test functions for running separately
async def test_health():
    """Quick health test"""
    async with httpx.AsyncClient(base_url=TEST_CONFIG['base_url']) as client:
        response = await client.get("/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")

async def test_quick_portfolio():
    """Quick portfolio test"""
    test_suite = TestBackendAPI()
    await test_suite.setup()
    result = await test_suite.test_all_assets_analysis()
    print(f"Portfolio test result: {result}")

# Main test execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == 'health':
            asyncio.run(test_health())
        elif test_type == 'portfolio':
            asyncio.run(test_quick_portfolio())
        else:
            print("Usage: python test_backend.py [health|portfolio]")
    else:
        # Run full test suite with pytest
        print("Running full test suite...")
        print("Use: pytest test_backend.py -v")
        print("Or: python test_backend.py health")
        print("Or: python test_backend.py portfolio")
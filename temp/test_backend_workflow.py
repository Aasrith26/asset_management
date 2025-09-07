#!/usr/bin/env python3
# BACKEND WORKFLOW TEST SCRIPT
# =============================
# Quick script to test your complete backend setup

import requests
import time
import json
from datetime import datetime

class BackendTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_health_check(self):
        """Test 1: Health Check"""
        print("🔍 Testing Health Check...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed!")
                print(f"   Status: {data.get('status')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Available Assets: {data.get('available_assets')}")
                return True
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                return False
        except requests.ConnectionError:
            print("❌ Cannot connect to server - is it running?")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_assets_list(self):
        """Test 2: Assets List"""
        print("\n🔍 Testing Assets List...")
        try:
            response = requests.get(f"{self.base_url}/assets", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Assets list retrieved!")
                print(f"   Available: {data.get('available_assets')}")
                print(f"   Total: {data.get('total_assets')}")
                return True
            else:
                print(f"❌ Assets list failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Assets list error: {e}")
            return False
    
    def start_analysis(self):
        """Test 3: Start Analysis"""
        print("\n🔍 Starting Analysis...")
        try:
            payload = {
                "assets": ["NIFTY50", "GOLD"],
                "timeout_minutes": 5
            }
            response = requests.post(
                f"{self.base_url}/analyze/all-assets",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data.get('job_id')
                print("✅ Analysis started!")
                print(f"   Job ID: {job_id}")
                print(f"   Status: {data.get('status')}")
                print(f"   Assets: {data.get('assets_requested')}")
                return job_id
            else:
                print(f"❌ Analysis start failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Analysis start error: {e}")
            return None
    
    def check_job_status(self, job_id):
        """Test 4: Check Job Status"""
        print(f"\n🔍 Checking Job Status: {job_id}")
        try:
            response = requests.get(f"{self.base_url}/status/{job_id}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                progress = data.get('progress', {})
                
                print(f"📊 Status: {status}")
                for asset, asset_status in progress.items():
                    print(f"   {asset}: {asset_status}")
                
                return status
            else:
                print(f"❌ Status check failed with status {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return None
    
    def get_final_results(self, job_id):
        """Test 5: Get Final Results"""
        print(f"\n🔍 Getting Final Results: {job_id}")
        try:
            response = requests.get(f"{self.base_url}/results/{job_id}", timeout=15)
            if response.status_code == 200:
                data = response.json()
                print("✅ Results retrieved!")
                
                results = data.get('results', {}).get('analysis_results', [])
                for result in results:
                    asset = result.get('asset')
                    sentiment = result.get('sentiment_score')
                    confidence = result.get('confidence')
                    status = result.get('status')
                    execution_time = result.get('execution_time')
                    
                    print(f"\n📊 {asset} Results:")
                    print(f"   Sentiment: {sentiment:.3f}")
                    print(f"   Confidence: {confidence:.1%}")
                    print(f"   Status: {status}")
                    print(f"   Execution Time: {execution_time:.1f}s")
                
                files = data.get('files', {})
                if files:
                    print(f"\n📁 Generated Files:")
                    print(f"   CSV: {files.get('csv_file', 'N/A')}")
                    json_files = files.get('json_files', {})
                    for asset, file_path in json_files.items():
                        print(f"   {asset} JSON: {file_path}")
                
                return data
            elif response.status_code == 400:
                print("⏳ Job not completed yet")
                return None
            else:
                print(f"❌ Results failed with status {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Results error: {e}")
            return None
    
    def run_complete_test(self):
        """Run complete backend test"""
        print("🚀 COMPLETE BACKEND TEST")
        print("=" * 50)
        
        # Test 1: Health Check
        if not self.test_health_check():
            print("\n❌ CRITICAL: Health check failed - server may not be running")
            print("💡 Start the server with: python fixed_orchestrator.py")
            return False
        
        # Test 2: Assets List
        if not self.test_assets_list():
            print("\n❌ Assets list failed")
            return False
        
        # Test 3: Start Analysis
        job_id = self.start_analysis()
        if not job_id:
            print("\n❌ CRITICAL: Could not start analysis")
            return False
        
        # Test 4: Monitor Job Progress
        print(f"\n⏳ Monitoring job progress...")
        max_wait_time = 120  # 2 minutes max
        check_interval = 10   # Check every 10 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            status = self.check_job_status(job_id)
            if status in ['completed', 'partial']:
                print(f"\n🎉 Analysis completed with status: {status}")
                break
            elif status == 'failed':
                print(f"\n❌ Analysis failed")
                return False
            elif status == 'running':
                print(f"⏳ Still running... ({elapsed_time}s elapsed)")
            
            time.sleep(check_interval)
            elapsed_time += check_interval
        else:
            print(f"\n⏰ Timeout after {max_wait_time}s - analysis may still be running")
        
        # Test 5: Get Results
        results = self.get_final_results(job_id)
        if results:
            print("\n🎉 COMPLETE TEST SUCCESSFUL!")
            print("✅ Your backend is working properly")
            
            # Summary
            analysis_results = results.get('results', {}).get('analysis_results', [])
            successful_analyses = [r for r in analysis_results if r.get('status') == 'success']
            
            print(f"\n📊 SUMMARY:")
            print(f"   Total Assets: {len(analysis_results)}")
            print(f"   Successful: {len(successful_analyses)}")
            print(f"   Success Rate: {len(successful_analyses)/len(analysis_results)*100:.0f}%")
            
            return True
        else:
            print("\n❌ Could not retrieve final results")
            return False

def main():
    print(f"🧪 BACKEND WORKFLOW TEST")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check if server is likely running
    print("💡 Make sure your server is running:")
    print("   python fixed_orchestrator.py")
    print("   Server should be at: http://localhost:8000")
    
    input("\n⏳ Press Enter when server is ready...")
    
    tester = BackendTester()
    success = tester.run_complete_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎊 ALL TESTS PASSED!")
        print("✅ Your backend is ready for production")
        print("\n📝 You can now use these endpoints:")
        print("   Health: GET  http://localhost:8000/health")
        print("   Assets: GET  http://localhost:8000/assets") 
        print("   Analyze: POST http://localhost:8000/analyze/all-assets")
        print("   Status: GET  http://localhost:8000/status/{job_id}")
        print("   Results: GET http://localhost:8000/results/{job_id}")
        print("\n🚀 Ready for Postman testing!")
    else:
        print("❌ SOME TESTS FAILED")
        print("💡 Check the error messages above")
        print("🔧 Common issues:")
        print("   - Server not running (python fixed_orchestrator.py)")
        print("   - Missing asset_integration.py file")
        print("   - Import errors in KPI files")
        print("   - Port 8000 already in use")
    
    print(f"\n⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
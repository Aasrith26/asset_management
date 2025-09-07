# QUICK TEST SCRIPT FOR JOB-BASED PIPELINE
# ========================================
# Test the complete job-based system

import requests
import json
import time
import uuid

def test_job_based_pipeline():
    """Complete test of the job-based pipeline system"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 TESTING JOB-BASED PIPELINE SYSTEM")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        health = response.json()
        print(f"✅ Health: {health['status']}")
        print(f"📊 Assets available: {health['assets_available']}")
        print(f"🔧 Job tracking: {health['job_tracking']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Run Complete Analysis
    print("\n2️⃣ Running Complete 4-Asset Analysis...")
    analysis_request = {
        "assets": ["NIFTY50", "GOLD", "BITCOIN", "REIT"],
        "timeout_minutes": 20,
        "generate_pipeline_outputs": True,
        "save_results": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/analyze/all-assets", 
            json=analysis_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            execution_time = time.time() - start_time
            
            print(f"✅ Analysis completed in {execution_time:.1f}s")
            print(f"📋 Job ID: {job_id}")
            print(f"📊 Portfolio Sentiment: {result['portfolio_summary']['portfolio_sentiment']}")
            print(f"🎯 Successful Assets: {result['portfolio_summary']['successful_assets']}/4")
            print(f"💡 Recommendation: {result['portfolio_summary']['investment_recommendation']}")
            
            if 'pipeline_outputs' in result:
                pipeline = result['pipeline_outputs']
                print(f"📁 Files Generated: {pipeline['files_generated']}")
                print(f"🔗 Ready for Downstream: {pipeline['ready_for_downstream']}")
            
            return job_id
        else:
            print(f"❌ Analysis failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Analysis request failed: {e}")
        return None

def test_file_access(job_id):
    """Test file access for downstream models"""
    
    if not job_id:
        print("⚠️ No job ID provided, skipping file access tests")
        return
    
    base_url = "http://localhost:8000"
    
    print(f"\n3️⃣ Testing File Access for Job: {job_id}")
    print("-" * 40)
    
    # Test 3a: List Job Files
    print("📁 Listing job files...")
    try:
        response = requests.get(f"{base_url}/jobs/{job_id}/files")
        files_info = response.json()
        
        print(f"✅ Total files: {files_info['total_files']}")
        for file_info in files_info['files']:
            print(f"   📄 {file_info['filename']} ({file_info['type']}) - {file_info['size_kb']}KB")
    except Exception as e:
        print(f"❌ File listing failed: {e}")
    
    # Test 3b: Access Consolidated CSV (Backend Model A)
    print("\n📊 Testing CSV access (Backend Model A)...")
    try:
        response = requests.get(f"{base_url}/jobs/{job_id}/csv")
        csv_data = response.json()
        
        metadata = csv_data['metadata']
        print(f"✅ CSV data retrieved:")
        print(f"   Rows: {metadata['rows']}")
        print(f"   Columns: {metadata['columns']}")
        print(f"   Assets: {metadata['assets_included']}")
        print(f"   Size: {metadata['file_size_kb']}KB")
        
        # Show sample data
        sample_data = csv_data['data'][:3]
        print(f"   Sample rows: {len(sample_data)}")
        for row in sample_data:
            print(f"     {row['asset']}: {row['metric_name']} = {row['value']}")
            
    except Exception as e:
        print(f"❌ CSV access failed: {e}")
    
    # Test 3c: Access Asset Context Files (Backend Model B)
    print("\n📁 Testing Context Files (Backend Model B)...")
    assets = ['BITCOIN', 'REIT', 'NIFTY50', 'GOLD']
    
    for asset in assets:
        try:
            response = requests.get(f"{base_url}/jobs/{job_id}/context/{asset}")
            if response.status_code == 200:
                context_data = response.json()
                context = context_data['context_data']
                
                sentiment = context['sentiment_analysis']
                print(f"✅ {asset}:")
                print(f"   Sentiment: {sentiment['overall_sentiment']}")
                print(f"   Confidence: {sentiment['confidence_level']}")
                print(f"   Grade: {sentiment['confidence_grade']}")
                print(f"   Status: {context['asset_metadata']['status']}")
            else:
                print(f"⚠️ {asset}: Not available ({response.status_code})")
                
        except Exception as e:
            print(f"❌ {asset} context failed: {e}")

def test_job_management():
    """Test job management endpoints"""
    
    base_url = "http://localhost:8000"
    
    print(f"\n4️⃣ Testing Job Management...")
    print("-" * 40)
    
    # Test 4a: List All Jobs
    print("📋 Listing all jobs...")
    try:
        response = requests.get(f"{base_url}/jobs")
        jobs_data = response.json()
        
        print(f"✅ Total jobs: {jobs_data['total_jobs']}")
        
        if jobs_data['jobs']:
            print("   Recent jobs:")
            for job in jobs_data['jobs'][:3]:  # Show first 3
                print(f"     {job['job_id'][:8]}... - {job.get('files_available', 0)} files - {job.get('created_at', 'Unknown')}")
        else:
            print("   No jobs found")
            
    except Exception as e:
        print(f"❌ Job listing failed: {e}")

def test_downstream_integration():
    """Demonstrate downstream model integration"""
    
    print(f"\n5️⃣ Downstream Model Integration Examples")
    print("-" * 40)
    
    print("🤖 Backend Model A (CSV Processing) Example:")
    print("""
    import requests
    import pandas as pd
    
    def get_consolidated_data(job_id):
        response = requests.get(f"http://localhost:8000/jobs/{job_id}/csv")
        data = response.json()['data']
        df = pd.DataFrame(data)
        
        # Filter for Bitcoin sentiment metrics
        bitcoin_sentiment = df[
            (df['asset'] == 'BITCOIN') & 
            (df['metric_type'] == 'sentiment')
        ]
        return bitcoin_sentiment
    """)
    
    print("\n🧠 Backend Model B (Context Processing) Example:")
    print("""
    import requests
    
    def get_asset_context(job_id, asset):
        response = requests.get(f"http://localhost:8000/jobs/{job_id}/context/{asset}")
        context = response.json()['context_data']
        
        return {
            'sentiment': context['sentiment_analysis']['overall_sentiment'],
            'confidence': context['sentiment_analysis']['confidence_level'],
            'components': context['component_breakdown'],
            'risk_grade': context['sentiment_analysis']['confidence_grade']
        }
    """)

def main():
    """Run complete test suite"""
    
    print("🚀 COMPLETE JOB-BASED PIPELINE TEST SUITE")
    print("=" * 70)
    print("Testing the enhanced backend with:")
    print("  • Job ID tracking")
    print("  • Consolidated CSV output")
    print("  • Asset-specific context JSONs")
    print("  • Direct API file access")
    print("  • Downstream model integration")
    print()
    
    # Run tests
    job_id = test_job_based_pipeline()
    
    if job_id:
        test_file_access(job_id)
        test_job_management()
        test_downstream_integration()
        
        print(f"\n🎉 TESTING COMPLETE!")
        print(f"📋 Test Job ID: {job_id}")
        print(f"🔗 Access your files at:")
        print(f"   CSV: http://localhost:8000/jobs/{job_id}/csv")
        print(f"   Context: http://localhost:8000/jobs/{job_id}/context/BITCOIN")
        print(f"   Files: http://localhost:8000/jobs/{job_id}/files")
    else:
        print(f"\n❌ TESTING FAILED!")
        print("Please check if the backend is running and try again.")

if __name__ == "__main__":
    main()
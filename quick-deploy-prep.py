# SIMPLE DEPLOYMENT PREPARATION SCRIPT
# ====================================
# Run this to prepare your repo for deployment

import os
import json

def create_essential_files():
    """Create only the essential deployment files"""
    
    print("🚀 Preparing your repo for deployment...")
    
    # 1. requirements.txt - ESSENTIAL
    requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.1
numpy==1.24.3
requests==2.31.0
pydantic==2.4.2
python-multipart==0.0.6
aiofiles==23.2.1
httpx==0.25.0
yfinance==0.2.22
python-dotenv==1.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✅ Created requirements.txt")
    
    # 2. Dockerfile - ESSENTIAL for Railway/Render
    dockerfile = """FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p pipeline_outputs analysis_results logs

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main_4asset_backend_job_based:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    print("✅ Created Dockerfile")
    
    # 3. .dockerignore - ESSENTIAL for clean builds
    dockerignore = """__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv/
env/
venv/
.git/
README.md
*.md
.DS_Store
logs/
temp/
analysis_results/
pipeline_outputs/
"""
    
    with open('.dockerignore', 'w') as f:
        f.write(dockerignore)
    print("✅ Created .dockerignore")


def print_quick_deployment_steps():
    """Print concise deployment steps"""
    
    print("\n🚀 QUICK DEPLOYMENT GUIDE")
    print("=" * 25)
    

    


def print_essential_api_usage():
    """Print the most important API usage examples"""
    
    print("\n📚 ESSENTIAL API USAGE")
    print("=" * 22)
    
    usage = """
Replace YOUR_URL with your deployed URL:

1. 🏥 HEALTH CHECK:
   curl https://YOUR_URL/health

2. 🚀 START ANALYSIS:
   curl -X POST https://YOUR_URL/analyze/all-assets \\
     -H "Content-Type: application/json" \\
     -d '{"generate_pipeline_outputs": true}'
   
   Returns job_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

3. 📊 GET CSV DATA (Your Main Request!):
   curl https://YOUR_URL/jobs/YOUR_JOB_ID/csv

4. 📄 GET CONTEXT FILES:
   curl https://YOUR_URL/jobs/YOUR_JOB_ID/context/BITCOIN
   curl https://YOUR_URL/jobs/YOUR_JOB_ID/context/GOLD
   curl https://YOUR_URL/jobs/YOUR_JOB_ID/context/NIFTY50
   curl https://YOUR_URL/jobs/YOUR_JOB_ID/context/REIT

5. 🐍 PYTHON FILTER EXAMPLE:
```python
import requests
import pandas as pd

# Get CSV data
response = requests.get("https://YOUR_URL/jobs/YOUR_JOB_ID/csv")
df = pd.DataFrame(response.json()['data'])

# Filter for asset + sentiment (YOUR REQUEST!)
sentiment_data = df[
    (df['metric_type'] == 'sentiment') & 
    (df['metric_name'] == 'overall_sentiment') &
    (df['asset'] != 'PORTFOLIO')
][['asset', 'value']]

result = dict(zip(sentiment_data['asset'], sentiment_data['value']))
print(result)
# Output: {'NIFTY50': 0.0, 'GOLD': 0.32, 'BITCOIN': 0.20, 'REIT': -0.60}
```
    """
    
    print(usage)

def create_simple_test_script():
    """Create a simple test script for the deployed backend"""
    
    test_script = '''#!/usr/bin/env python3
"""
Simple test script for your deployed Asset Management Backend
Usage: python test_backend.py YOUR_DEPLOYED_URL
"""

import sys
import requests
import pandas as pd
import time

def test_backend(base_url):
    """Test the deployed backend"""
    
    print(f"🧪 Testing backend at: {base_url}")
    
    # 1. Health check
    print("\\n1. 🏥 Health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Backend is healthy!")
            print(f"   Service: {response.json().get('service', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    # 2. Start analysis
    print("\\n2. 🚀 Starting analysis...")
    try:
        data = {
            "assets": ["NIFTY50", "GOLD", "BITCOIN", "REIT"],
            "timeout_minutes": 15,
            "generate_pipeline_outputs": True
        }
        
        response = requests.post(
            f"{base_url}/analyze/all-assets",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get("job_id")
            print(f"✅ Analysis started!")
            print(f"   Job ID: {job_id}")
            
            # 3. Wait and get results
            print("\\n3. ⏳ Waiting for completion...")
            
            # Wait a bit
            time.sleep(60)  # Wait 1 minute
            
            # 4. Get CSV data
            print("\\n4. 📊 Getting CSV data...")
            csv_response = requests.get(f"{base_url}/jobs/{job_id}/csv")
            
            if csv_response.status_code == 200:
                csv_data = csv_response.json()
                df = pd.DataFrame(csv_data['data'])
                
                # Filter for sentiment
                sentiment_data = df[
                    (df['metric_type'] == 'sentiment') & 
                    (df['metric_name'] == 'overall_sentiment') &
                    (df['asset'] != 'PORTFOLIO')
                ][['asset', 'value']]
                
                sentiment_dict = dict(zip(sentiment_data['asset'], sentiment_data['value']))
                
                print("✅ Got sentiment data:")
                for asset, sentiment in sentiment_dict.items():
                    print(f"   {asset}: {sentiment:.4f}")
                
                # 5. Test context files
                print("\\n5. 📄 Testing context files...")
                assets = ["BITCOIN", "GOLD"]  # Test a couple
                
                for asset in assets:
                    context_response = requests.get(f"{base_url}/jobs/{job_id}/context/{asset}")
                    if context_response.status_code == 200:
                        print(f"   ✅ {asset} context file available")
                    else:
                        print(f"   ❌ {asset} context file failed")
                
                print("\\n🎉 All tests passed!")
                print(f"\\n📋 Your backend is working correctly at: {base_url}")
                
            else:
                print(f"❌ CSV data not ready yet. Status: {csv_response.status_code}")
                print("   Try again in a few minutes.")
        
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_backend.py YOUR_DEPLOYED_URL")
        print("Example: python test_backend.py https://your-app.railway.app")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    test_backend(base_url)
'''
    
    with open('test_backend.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_backend.py")
    print("   Usage: python test_backend.py https://your-deployed-url")

if __name__ == "__main__":
    print("🎯 SIMPLE DEPLOYMENT PREPARATION")
    print("=" * 35)
    
    create_essential_files()
    create_simple_test_script()
    print_quick_deployment_steps()  
    print_essential_api_usage()
    
    print("\n" + "="*50)
    print("🚀 YOU'RE READY TO DEPLOY!")
    print("="*50)
    print("1. Run this script to create deployment files")
    print("2. Commit and push to GitHub")
    print("3. Deploy on Railway (recommended) or Render") 
    print("4. Test with: python test_backend.py YOUR_URL")
    print("5. Use the API examples above to access your data")
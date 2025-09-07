#!/usr/bin/env python3
# ORCHESTRATOR USAGE EXAMPLE v1.0
# ================================
# Example script showing how to use the orchestrator from a frontend application

import requests
import time
import json
from typing import Dict, Optional

class SentimentOrchestrationClient:
    """Client for the Multi-Asset Sentiment Analysis Orchestrator"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def start_analysis(self, assets: list = None, timeout_minutes: int = 15) -> Dict:
        """Start sentiment analysis for specified assets"""
        if assets is None:
            assets = ["NIFTY50", "GOLD"]
        
        payload = {
            "assets": assets,
            "timeout_minutes": timeout_minutes,
            "priority": "normal"
        }
        
        response = requests.post(f"{self.base_url}/analyze/all-assets", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to start analysis: {response.status_code} - {response.text}")
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get current status of analysis job"""
        response = requests.get(f"{self.base_url}/status/{job_id}")
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise Exception(f"Job {job_id} not found")
        else:
            raise Exception(f"Failed to get status: {response.status_code} - {response.text}")
    
    def get_results(self, job_id: str) -> Dict:
        """Get detailed results of completed analysis"""
        response = requests.get(f"{self.base_url}/results/{job_id}")
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise Exception(f"Job {job_id} not found")
        elif response.status_code == 400:
            raise Exception("Job not completed yet")
        else:
            raise Exception(f"Failed to get results: {response.status_code} - {response.text}")
    
    def wait_for_completion(self, job_id: str, max_wait_seconds: int = 300) -> Dict:
        """Wait for analysis to complete and return results"""
        print(f"⏳ Waiting for job {job_id} to complete...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            try:
                status = self.get_job_status(job_id)
                current_status = status['status']
                progress = status['progress']
                
                # Print progress
                progress_str = ", ".join([f"{asset}: {status}" for asset, status in progress.items()])
                print(f"   Status: {current_status} | Progress: {progress_str}")
                
                if current_status in ['completed', 'partial', 'failed']:
                    if current_status in ['completed', 'partial']:
                        print(f"✅ Job {current_status}!")
                        return self.get_results(job_id)
                    else:
                        print(f"❌ Job failed: {status.get('error_message', 'Unknown error')}")
                        return status
                
                time.sleep(2)  # Wait 2 seconds before checking again
                
            except Exception as e:
                print(f"Error checking status: {str(e)}")
                time.sleep(5)
        
        raise Exception(f"Job did not complete within {max_wait_seconds} seconds")
    
    def health_check(self) -> Dict:
        """Check if orchestrator is healthy"""
        response = requests.get(f"{self.base_url}/health")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Health check failed: {response.status_code}")
    
    def get_available_assets(self) -> list:
        """Get list of available assets"""
        response = requests.get(f"{self.base_url}/assets")
        
        if response.status_code == 200:
            data = response.json()
            return data['available_assets']
        else:
            raise Exception(f"Failed to get assets: {response.status_code}")

def example_usage():
    """Example usage of the orchestrator client"""
    print("🚀 MULTI-ASSET SENTIMENT ORCHESTRATOR - USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize client
    client = SentimentOrchestrationClient()
    
    try:
        # 1. Health Check
        print("\n1. 🏥 Health Check:")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Available Assets: {health['available_assets']}")
        print(f"   Active Jobs: {health['active_jobs']}")
        
        # 2. Start Analysis
        print("\n2. 🚀 Starting Analysis:")
        job_response = client.start_analysis(assets=["NIFTY50", "GOLD"], timeout_minutes=20)
        job_id = job_response['job_id']
        print(f"   Job ID: {job_id}")
        print(f"   Assets: {job_response['assets_requested']}")
        print(f"   Estimated completion: {job_response.get('estimated_completion', 'Unknown')}")
        
        # 3. Wait for Completion
        print(f"\n3. ⏳ Waiting for Completion:")
        results = client.wait_for_completion(job_id, max_wait_seconds=600)  # 10 minutes max
        
        # 4. Display Results
        print(f"\n4. 📊 Results:")
        if 'results' in results:
            analysis_results = results['results']['analysis_results']
            print(f"   Total Assets Analyzed: {len(analysis_results)}")
            
            for result in analysis_results:
                asset = result['asset']
                sentiment = result['sentiment_score']
                confidence = result['confidence']
                execution_time = result['execution_time']
                status = result['status']
                
                print(f"\n   📈 {asset}:")
                print(f"      Sentiment Score: {sentiment:.4f}")
                print(f"      Confidence: {confidence:.4f}")
                print(f"      Execution Time: {execution_time:.1f}s")
                print(f"      Status: {status}")
            
            # Integration Status
            print(f"\n5. 🔗 Integration Status:")
            integration = results['results'].get('integration_status', {})
            csv_sent = integration.get('csv_sent', False)
            json_routing = integration.get('json_routing', {})
            
            print(f"   CSV sent to Model A: {'✅' if csv_sent else '❌'}")
            for asset, sent in json_routing.items():
                print(f"   {asset} JSON sent to Model B: {'✅' if sent else '❌'}")
            
            # File Information
            print(f"\n6. 📁 Generated Files:")
            files = results.get('files', {})
            if 'csv_file' in files:
                print(f"   CSV File: {files['csv_file']}")
            if 'json_files' in files:
                for asset, json_file in files['json_files'].items():
                    print(f"   {asset} JSON: {json_file}")
        
        print(f"\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")

def quick_sentiment_check():
    """Quick sentiment check function for integration into other systems"""
    client = SentimentOrchestrationClient()
    
    try:
        # Start analysis
        job_response = client.start_analysis()
        job_id = job_response['job_id']
        
        # Wait for completion (with shorter timeout for quick check)
        results = client.wait_for_completion(job_id, max_wait_seconds=180)  # 3 minutes
        
        # Extract just the sentiment scores
        sentiment_scores = {}
        if 'results' in results:
            for result in results['results']['analysis_results']:
                sentiment_scores[result['asset']] = {
                    'sentiment': result['sentiment_score'],
                    'confidence': result['confidence']
                }
        
        return sentiment_scores
        
    except Exception as e:
        print(f"Quick sentiment check failed: {str(e)}")
        return {}

def batch_analysis_example():
    """Example of running multiple analyses in sequence"""
    print("\n🔄 BATCH ANALYSIS EXAMPLE")
    print("-" * 40)
    
    client = SentimentOrchestrationClient()
    batch_results = []
    
    # Run 3 analyses with different configurations
    configurations = [
        {"assets": ["NIFTY50"], "timeout_minutes": 10},
        {"assets": ["GOLD"], "timeout_minutes": 15},
        {"assets": ["NIFTY50", "GOLD"], "timeout_minutes": 20}
    ]
    
    for i, config in enumerate(configurations, 1):
        print(f"\nBatch {i}: {config['assets']}")
        
        try:
            # Start analysis
            job_response = client.start_analysis(**config)
            job_id = job_response['job_id']
            
            # Wait for completion
            results = client.wait_for_completion(job_id, max_wait_seconds=300)
            batch_results.append(results)
            
            print(f"   ✅ Batch {i} completed")
            
        except Exception as e:
            print(f"   ❌ Batch {i} failed: {str(e)}")
    
    print(f"\n📊 Batch Summary: {len(batch_results)} successful analyses")
    return batch_results

if __name__ == "__main__":
    print("🎯 SELECT EXAMPLE TO RUN:")
    print("1. Full usage example")
    print("2. Quick sentiment check")
    print("3. Batch analysis example")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            example_usage()
        elif choice == '2':
            print("\n⚡ QUICK SENTIMENT CHECK")
            print("-" * 30)
            scores = quick_sentiment_check()
            if scores:
                for asset, data in scores.items():
                    sentiment = data['sentiment']
                    confidence = data['confidence']
                    print(f"{asset}: {sentiment:.4f} (conf: {confidence:.4f})")
        elif choice == '3':
            batch_analysis_example()
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Make sure the orchestrator server is running:")
        print("   python start_orchestrator.py")
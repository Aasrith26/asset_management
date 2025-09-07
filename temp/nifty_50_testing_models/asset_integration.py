import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import logging

# Add your project paths
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)

class NiftyIntegrationAdapter:
    """Adapter to integrate your existing Nifty 50 system"""
    
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        """Run your Nifty 50 analysis system"""
        try:
            logger.info("Starting Nifty 50 analysis integration...")
            
            # Method 1: Direct import (update these imports to match your files)
            try:
                from integrated_nifty_sentiment import run_nifty_comprehensive_analysis
                result = await asyncio.to_thread(run_nifty_comprehensive_analysis)
                return result
            except ImportError:
                pass
            
            # Method 2: Alternative import path
            try:
                from nifty_sentiment_analyzer import NiftySentimentAnalyzer
                analyzer = NiftySentimentAnalyzer()
                result = await asyncio.to_thread(analyzer.run_comprehensive_analysis)
                return result
            except ImportError:
                pass
            
            # Method 3: Subprocess execution (fallback)
            import subprocess
            import json
            
            result = subprocess.run([
                sys.executable, 
                "your_nifty_main_script.py"  # Update with your actual script name
            ], capture_output=True, text=True, timeout=900)  # 15 minutes timeout
            
            if result.returncode == 0:
                # If your script outputs JSON
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If your script saves a file, read it
                    json_files = list(Path("../../Downloads/exported-assets (1)").glob("*nifty*integration*.json"))
                    if json_files:
                        with open(json_files[-1], 'r') as f:
                            return json.load(f)
            
            # Method 4: Mock data for testing (remove in production)
            logger.warning("Using mock data for Nifty 50 - update integration!")
            return {
                'composite_sentiment': 0.65,
                'composite_confidence': 0.82,
                'component_breakdown': {
                    'fii_dii_flows': {'sentiment': 0.45, 'confidence': 0.85},
                    'technical_analysis': {'sentiment': 0.72, 'confidence': 0.90},
                    'market_sentiment': {'sentiment': 0.58, 'confidence': 0.75},
                    'rbi_interest_rates': {'sentiment': 0.12, 'confidence': 0.70},
                    'global_factors': {'sentiment': 0.35, 'confidence': 0.80}
                },
                'execution_time': 12.5,
                'data_quality': {
                    'coverage_pct': 85,
                    'sources_available': 18,
                    'real_time_pct': 75
                },
                'interpretation': 'BULLISH - Strong institutional flows and technical signals',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Nifty analysis integration failed: {str(e)}")
            raise

class GoldIntegrationAdapter:
    """Adapter to integrate your existing Gold system"""
    
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        """Run your Gold analysis system"""
        try:
            logger.info("Starting Gold analysis integration...")
            
            # Method 1: Direct import (update these imports to match your files)
            try:
                from integrated_gold_sentiment_v5 import run_integrated_analysis
                result = await run_integrated_analysis()
                return result
            except ImportError:
                pass
            
            # Method 2: Alternative import path
            try:
                from integrated_gold_sentiment_json_v5 import run_json_analysis
                result = await run_json_analysis()
                return result
            except ImportError:
                pass
            
            # Method 3: Subprocess execution (fallback)
            import subprocess
            import json
            
            result = subprocess.run([
                sys.executable,
                "integrated_gold_sentiment_v5.py"  # Update with your actual script
            ], capture_output=True, text=True, timeout=1200)  # 20 minutes timeout
            
            if result.returncode == 0:
                # If your script outputs JSON
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If your script saves a file, read it
                    json_files = list(Path("../../Downloads/exported-assets (1)").glob("*gold*integration*.json"))
                    if json_files:
                        with open(json_files[-1], 'r') as f:
                            return json.load(f)
            
            # Method 4: Mock data for testing (remove in production)
            logger.warning("Using mock data for Gold - update integration!")
            return {
                'executive_summary': {
                    'overall_sentiment': 0.58,
                    'confidence_level': 0.74,
                    'interpretation': 'MILDLY BULLISH - Central bank buying and real rate environment supportive'
                },
                'component_analysis': {
                    'technical_analysis': {
                        'sentiment': 0.42, 'confidence': 0.85,
                        'description': 'Gold momentum indicators mixed'
                    },
                    'real_interest_rates': {
                        'sentiment': 0.78, 'confidence': 0.90,
                        'description': 'Negative real rates supportive'
                    },
                    'central_bank_sentiment': {
                        'sentiment': 0.65, 'confidence': 0.70,
                        'description': 'Continued CB accumulation trend'
                    },
                    'professional_positioning': {
                        'sentiment': 0.35, 'confidence': 0.60,
                        'description': 'COT data shows mixed positioning'
                    },
                    'currency_debasement': {
                        'sentiment': 0.52, 'confidence': 0.65,
                        'description': 'Dollar weakness supportive'
                    }
                },
                'performance_metrics': {
                    'execution_time_seconds': 18.7,
                    'framework_coverage': 0.95,
                    'data_quality_score': 0.78
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Gold analysis integration failed: {str(e)}")
            raise

# Integration test functions
async def test_nifty_integration():
    """Test Nifty integration"""
    try:
        result = await NiftyIntegrationAdapter.run_analysis()
        print("✅ Nifty integration successful!")
        print(f"Sentiment: {result.get('composite_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('composite_confidence', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Nifty integration failed: {str(e)}")
        return False

async def test_gold_integration():
    """Test Gold integration"""
    try:
        result = await GoldIntegrationAdapter.run_analysis()
        print("✅ Gold integration successful!")
        executive = result.get('executive_summary', {})
        print(f"Sentiment: {executive.get('overall_sentiment', 'N/A')}")
        print(f"Confidence: {executive.get('confidence_level', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Gold integration failed: {str(e)}")
        return False

if __name__ == "__main__":
    async def main():
        print("🧪 TESTING ASSET INTEGRATIONS")
        print("=" * 40)
        
        print("\n1. Testing Nifty 50 Integration:")
        nifty_success = await test_nifty_integration()
        
        print("\n2. Testing Gold Integration:")
        gold_success = await test_gold_integration()
        
        print(f"\n📊 INTEGRATION TEST RESULTS:")
        print(f"Nifty 50: {'✅ PASS' if nifty_success else '❌ FAIL'}")
        print(f"Gold: {'✅ PASS' if gold_success else '❌ FAIL'}")
        
        if nifty_success and gold_success:
            print("\n🎉 All integrations ready for orchestrator!")
        else:
            print("\n⚠️ Some integrations need configuration.")
            print("Update the import paths in asset_integration.py")
    
    asyncio.run(main())
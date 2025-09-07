# ENHANCED ASSET INTEGRATION - 4 Asset Support
# =============================================
# Updated to include Bitcoin and REIT alongside existing Nifty 50 and Gold
# All assets now support unified output format

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Add paths for all asset KPIs
sys.path.append(str(Path(__file__).parent / "nifty_50_kpis"))
sys.path.append(str(Path(__file__).parent / "gold_kpis"))
sys.path.append(str(Path(__file__).parent / "bitcoin_kpis"))  # 🆕 New
sys.path.append(str(Path(__file__).parent / "reit_kpis"))     # 🆕 New

# Enhanced Asset Type Enum
class AssetType(str, Enum):
    NIFTY50 = "NIFTY50"
    GOLD = "GOLD"
    BITCOIN = "BITCOIN"    # 🆕 New
    REIT = "REIT"          # 🆕 New

class AssetCategory(str, Enum):
    TRADITIONAL = "traditional"     # Nifty + Gold
    ALTERNATIVE = "alternative"     # Bitcoin + REIT
    EQUITY = "equity"              # Nifty
    COMMODITY = "commodity"        # Gold
    CRYPTOCURRENCY = "crypto"      # Bitcoin
    REAL_ESTATE = "real_estate"    # REIT

@dataclass
class AssetAnalysisResult:
    """Standardized result format for all 4 assets"""
    asset: str
    sentiment_score: float
    confidence: float
    execution_time: float
    timestamp: datetime
    context_data: Dict[str, Any]
    status: str
    error_message: Optional[str] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# EXISTING INTEGRATION ADAPTERS (Nifty 50 & Gold)
# ================================================================

class NiftyIntegrationAdapter:
    """Nifty 50 integration adapter - existing implementation"""
    
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            print("📈 Starting Nifty 50 analysis...")
            
            # Import the complete Nifty analyzer
            from complete_nifty_analyzer import run_complete_nifty_analysis
            
            # Run analysis
            result = await run_complete_nifty_analysis()
            
            # Ensure standardized format
            if isinstance(result, dict):
                standardized = {
                    'composite_sentiment': result.get('composite_sentiment', 0.0),
                    'composite_confidence': result.get('composite_confidence', 0.5),
                    'framework_coverage': result.get('framework_coverage', 0.0),
                    'execution_time_seconds': result.get('execution_time_seconds', 0.0),
                    'component_details': result.get('component_analysis', {}),
                    'market_context': result.get('market_context', {}),
                    'timestamp': datetime.now().isoformat(),
                    'asset_type': 'equity_index',
                    'data_source': 'Multiple_Indian_APIs',
                    'status': 'success'
                }
                
                print(f"✅ Nifty 50 analysis complete: {standardized['composite_sentiment']:.3f}")
                return standardized
            
            raise ValueError("Invalid Nifty analysis result format")
            
        except Exception as e:
            logger.error(f"Nifty 50 analysis failed: {e}")
            return {
                'composite_sentiment': 0.0,
                'composite_confidence': 0.5,
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'asset_type': 'equity_index'
            }

class GoldIntegrationAdapter:
    """Gold integration adapter - existing implementation"""
    
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            print("🥇 Starting Gold analysis...")
            
            # Import the integrated gold analyzer
            from integrated_gold_sentiment_v5 import run_integrated_gold_analysis
            
            # Run analysis
            result = await run_integrated_gold_analysis()
            
            # Ensure standardized format
            if isinstance(result, dict) and 'executive_summary' in result:
                executive = result['executive_summary']
                standardized = {
                    'executive_summary': executive,
                    'overall_sentiment': executive.get('overall_sentiment', 0.0),
                    'confidence_level': executive.get('confidence_level', 0.5),
                    'framework_coverage': result.get('framework_coverage', 0.0),
                    'execution_time_seconds': result.get('performance_metrics', {}).get('execution_time_seconds', 0.0),
                    'component_analysis': result.get('component_analysis', {}),
                    'market_context': result.get('market_context', {}),
                    'timestamp': datetime.now().isoformat(),
                    'asset_type': 'commodity',
                    'data_source': 'Multiple_Global_APIs',
                    'status': 'success'
                }
                
                print(f"✅ Gold analysis complete: {executive.get('overall_sentiment', 0):.3f}")
                return standardized
            
            raise ValueError("Invalid Gold analysis result format")
            
        except Exception as e:
            logger.error(f"Gold analysis failed: {e}")
            return {
                'executive_summary': {
                    'overall_sentiment': 0.0,
                    'confidence_level': 0.5,
                    'interpretation': 'Error in analysis'
                },
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'asset_type': 'commodity'
            }

# ================================================================
# NEW INTEGRATION ADAPTERS (Bitcoin & REIT)
# ================================================================

class BitcoinIntegrationAdapter:
    """🆕 Bitcoin integration adapter - new implementation"""
    
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            print("₿ Starting Bitcoin analysis...")
            
            # Import the Bitcoin sentiment analyzer
            from bitcoin_sentiment_analyzer import run_bitcoin_analysis
            
            # Run analysis
            result = await run_bitcoin_analysis()
            
            # Ensure standardized format
            if isinstance(result, dict) and 'component_sentiment' in result:
                standardized = {
                    'component_sentiment': result.get('component_sentiment', 0.0),
                    'component_confidence': result.get('component_confidence', 0.5),
                    'component_weight': result.get('component_weight', 0.25),
                    'framework_coverage': result.get('framework_coverage', 0.0),
                    'execution_time_seconds': result.get('performance_metrics', {}).get('execution_time_seconds', 0.0),
                    'component_analysis': result.get('component_analysis', {}),
                    'executive_summary': result.get('executive_summary', {}),
                    'market_context': result.get('market_context', {}),
                    'timestamp': datetime.now().isoformat(),
                    'asset_type': 'cryptocurrency',
                    'data_source': 'Kraken_Multi_API',
                    'status': 'success'
                }
                
                print(f"✅ Bitcoin analysis complete: {standardized['component_sentiment']:.3f}")
                return standardized
            
            raise ValueError("Invalid Bitcoin analysis result format")
            
        except Exception as e:
            logger.error(f"Bitcoin analysis failed: {e}")
            return {
                'component_sentiment': 0.0,
                'component_confidence': 0.5,
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'asset_type': 'cryptocurrency'
            }

class REITIntegrationAdapter:
    """🆕 REIT integration adapter - new implementation"""
    
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            print("🏢 Starting REIT analysis...")
            
            # Import the REIT sentiment analyzer
            from reit_sentiment_analyzer import run_reit_analysis
            
            # Run analysis
            result = await run_reit_analysis()
            
            # Ensure standardized format
            if isinstance(result, dict) and 'component_sentiment' in result:
                standardized = {
                    'component_sentiment': result.get('component_sentiment', 0.0),
                    'component_confidence': result.get('component_confidence', 0.5),
                    'component_weight': result.get('component_weight', 0.25),
                    'framework_coverage': result.get('framework_coverage', 0.0),
                    'execution_time_seconds': result.get('performance_metrics', {}).get('execution_time_seconds', 0.0),
                    'component_analysis': result.get('component_analysis', {}),
                    'executive_summary': result.get('executive_summary', {}),
                    'market_context': result.get('market_context', {}),
                    'symbol': result.get('symbol', 'MINDSPACE'),
                    'timestamp': datetime.now().isoformat(),
                    'asset_type': 'real_estate_investment_trust',
                    'data_source': 'Yahoo_Finance_TradingEconomics',
                    'status': 'success'
                }
                
                print(f"✅ REIT analysis complete: {standardized['component_sentiment']:.3f}")
                return standardized
            
            raise ValueError("Invalid REIT analysis result format")
            
        except Exception as e:
            logger.error(f"REIT analysis failed: {e}")
            return {
                'component_sentiment': 0.0,
                'component_confidence': 0.5,
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'asset_type': 'real_estate_investment_trust'
            }

# ================================================================
# ENHANCED INTEGRATION ORCHESTRATOR
# ================================================================

class MultiAssetIntegrationOrchestrator:
    """Enhanced orchestrator for all 4 assets"""
    
    # Asset mapping
    ASSET_ADAPTERS = {
        AssetType.NIFTY50: NiftyIntegrationAdapter,
        AssetType.GOLD: GoldIntegrationAdapter,
        AssetType.BITCOIN: BitcoinIntegrationAdapter,  # 🆕 New
        AssetType.REIT: REITIntegrationAdapter         # 🆕 New
    }
    
    CATEGORY_MAPPINGS = {
        AssetCategory.TRADITIONAL: [AssetType.NIFTY50, AssetType.GOLD],
        AssetCategory.ALTERNATIVE: [AssetType.BITCOIN, AssetType.REIT],
        AssetCategory.EQUITY: [AssetType.NIFTY50],
        AssetCategory.COMMODITY: [AssetType.GOLD],
        AssetCategory.CRYPTOCURRENCY: [AssetType.BITCOIN],
        AssetCategory.REAL_ESTATE: [AssetType.REIT]
    }
    
    @classmethod
    async def run_all_assets(cls) -> Dict[str, Any]:
        """Run analysis for all 4 assets in parallel"""
        print("🚀 MULTI-ASSET INTEGRATION ORCHESTRATOR")
        print("=" * 60)
        print("🎯 Target: 4 assets (Nifty 50, Gold, Bitcoin, REIT)")
        
        start_time = datetime.now()
        results = {}
        
        # Run all assets in parallel
        tasks = []
        for asset_type in AssetType:
            adapter = cls.ASSET_ADAPTERS[asset_type]
            task = cls._run_single_asset_with_timeout(asset_type, adapter)
            tasks.append(task)
        
        # Await all results
        asset_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, asset_type in enumerate(AssetType):
            result = asset_results[i]
            if isinstance(result, Exception):
                print(f"❌ {asset_type.value} failed with exception: {result}")
                results[asset_type.value] = {
                    'status': 'exception',
                    'error': str(result),
                    'sentiment': 0.0,
                    'confidence': 0.3
                }
            else:
                results[asset_type.value] = result
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Generate summary
        summary = cls._generate_portfolio_summary(results, execution_time)
        
        print(f"\n🎯 MULTI-ASSET INTEGRATION COMPLETE")
        print(f"⏱️ Total Time: {execution_time:.1f}s")
        print(f"✅ Successful: {summary['successful_assets']}/4")
        print(f"📊 Portfolio Sentiment: {summary['portfolio_sentiment']:.3f}")
        
        return {
            'individual_results': results,
            'portfolio_summary': summary,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
    
    @classmethod
    async def _run_single_asset_with_timeout(cls, asset_type: AssetType, adapter_class, timeout: int = 120) -> Dict:
        """Run single asset analysis with timeout"""
        try:
            result = await asyncio.wait_for(adapter_class.run_analysis(), timeout=timeout)
            result['asset_type'] = asset_type.value
            return result
        except asyncio.TimeoutError:
            print(f"⏰ {asset_type.value} timed out after {timeout}s")
            return {
                'status': 'timeout',
                'asset_type': asset_type.value,
                'sentiment': 0.0,
                'confidence': 0.2,
                'error': f'Analysis timed out after {timeout}s'
            }
        except Exception as e:
            print(f"❌ {asset_type.value} error: {e}")
            return {
                'status': 'error',
                'asset_type': asset_type.value,
                'sentiment': 0.0,
                'confidence': 0.3,
                'error': str(e)
            }
    
    @classmethod
    async def run_asset_category(cls, category: AssetCategory) -> Dict[str, Any]:
        """Run analysis for specific asset category"""
        assets = cls.CATEGORY_MAPPINGS.get(category, [])
        print(f"🎯 Running {category.value} category analysis: {[a.value for a in assets]}")
        
        results = {}
        tasks = []
        
        for asset_type in assets:
            adapter = cls.ASSET_ADAPTERS[asset_type]
            task = cls._run_single_asset_with_timeout(asset_type, adapter)
            tasks.append((asset_type, task))
        
        # Execute tasks
        for asset_type, task in tasks:
            try:
                result = await task
                results[asset_type.value] = result
            except Exception as e:
                results[asset_type.value] = {
                    'status': 'exception',
                    'error': str(e),
                    'sentiment': 0.0,
                    'confidence': 0.3
                }
        
        return results
    
    @classmethod
    def _generate_portfolio_summary(cls, results: Dict[str, Any], execution_time: float) -> Dict:
        """Generate portfolio-level summary"""
        successful_results = [r for r in results.values() if r.get('status') == 'success']
        
        if not successful_results:
            return {
                'portfolio_sentiment': 0.0,
                'portfolio_confidence': 0.0,
                'successful_assets': 0,
                'total_assets': len(results),
                'execution_time': execution_time,
                'status': 'failed'
            }
        
        # Calculate portfolio metrics (equal weight for simplicity)
        sentiments = []
        confidences = []
        
        for result in successful_results:
            # Extract sentiment from different result formats
            sentiment = result.get('component_sentiment', 
                        result.get('composite_sentiment',
                        result.get('overall_sentiment',
                        result.get('executive_summary', {}).get('overall_sentiment', 0.0))))
            
            confidence = result.get('component_confidence',
                        result.get('composite_confidence', 
                        result.get('confidence_level',
                        result.get('executive_summary', {}).get('confidence_level', 0.5))))
            
            sentiments.append(float(sentiment))
            confidences.append(float(confidence))
        
        # Portfolio calculations
        portfolio_sentiment = sum(sentiments) / len(sentiments)
        portfolio_confidence = sum(confidences) / len(confidences)
        
        return {
            'portfolio_sentiment': round(portfolio_sentiment, 3),
            'portfolio_confidence': round(portfolio_confidence, 3),
            'successful_assets': len(successful_results),
            'total_assets': len(results),
            'asset_breakdown': {
                asset: {
                    'sentiment': round(sentiments[i], 3),
                    'confidence': round(confidences[i], 3),
                    'status': 'success'
                } for i, asset in enumerate([r.get('asset_type', 'unknown') for r in successful_results])
            },
            'execution_time': execution_time,
            'status': 'success' if len(successful_results) > 0 else 'failed'
        }

# ================================================================
# COMPATIBILITY FUNCTIONS
# ================================================================

async def test_all_integrations():
    """Test all 4 asset integrations"""
    print("🧪 TESTING ALL ASSET INTEGRATIONS")
    print("=" * 50)
    
    orchestrator = MultiAssetIntegrationOrchestrator()
    result = await orchestrator.run_all_assets()
    
    print(f"\n📋 INTEGRATION TEST SUMMARY:")
    print(f"Total Assets: {result['portfolio_summary']['total_assets']}")
    print(f"Successful: {result['portfolio_summary']['successful_assets']}")
    print(f"Portfolio Sentiment: {result['portfolio_summary']['portfolio_sentiment']}")
    print(f"Portfolio Confidence: {result['portfolio_summary']['portfolio_confidence']}")
    
    return result

async def test_individual_asset(asset: AssetType):
    """Test individual asset integration"""
    print(f"🧪 Testing {asset.value} integration...")
    
    adapter = MultiAssetIntegrationOrchestrator.ASSET_ADAPTERS[asset]
    result = await adapter.run_analysis()
    
    print(f"✅ {asset.value} test complete")
    return result

# Backwards compatibility exports
async def run_nifty_analysis():
    """Backwards compatibility for Nifty analysis"""
    return await NiftyIntegrationAdapter.run_analysis()

async def run_gold_analysis():
    """Backwards compatibility for Gold analysis"""
    return await GoldIntegrationAdapter.run_analysis()

# New asset exports
async def run_bitcoin_analysis():
    """New Bitcoin analysis export"""
    return await BitcoinIntegrationAdapter.run_analysis()

async def run_reit_analysis():
    """New REIT analysis export"""
    return await REITIntegrationAdapter.run_analysis()

# Main execution
if __name__ == "__main__":
    asyncio.run(test_all_integrations())
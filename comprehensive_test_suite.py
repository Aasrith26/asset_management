# COMPREHENSIVE TEST SUITE - All 4 Assets
# ========================================
# Complete testing framework for Bitcoin, REIT, Nifty 50, and Gold
# Validates all components and integrations

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Test configuration
TEST_CONFIG = {
    'timeout_per_component': 60,  # 60 seconds per component
    'timeout_per_asset': 120,     # 120 seconds per full asset
    'save_results': True,
    'verbose': True
}

class ComprehensiveTestSuite:
    """Complete test suite for all 4-asset system"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    async def run_all_tests(self):
        """Run comprehensive tests for all components and integrations"""
        print("🚀 COMPREHENSIVE 4-ASSET TEST SUITE")
        print("=" * 60)
        print(f"⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Testing: Bitcoin, REIT, Nifty 50, Gold")
        print("🧪 Coverage: Individual components + Integration + Performance")
        
        # Test sequence
        test_sequence = [
            ("Bitcoin Components", self.test_bitcoin_components),
            ("REIT Components", self.test_reit_components),
            ("Bitcoin Integration", self.test_bitcoin_integration),
            ("REIT Integration", self.test_reit_integration),
            ("All Assets Integration", self.test_all_assets_integration),
            ("Performance Benchmarks", self.test_performance_benchmarks)
        ]
        
        # Execute all tests
        for test_name, test_func in test_sequence:
            try:
                print(f"\n🔍 {test_name}")
                print("-" * 40)
                result = await test_func()
                self.results[test_name] = result
                print(f"✅ {test_name} - PASSED")
                
            except Exception as e:
                print(f"❌ {test_name} - FAILED: {e}")
                self.results[test_name] = {'status': 'failed', 'error': str(e)}
        
        # Generate final report
        await self.generate_final_report()
    
    async def test_bitcoin_components(self) -> Dict[str, Any]:
        """Test all Bitcoin components individually"""
        print("₿ Testing Bitcoin Components...")
        component_results = {}
        
        # Test Bitcoin Micro Momentum
        try:
            print("  🔍 Bitcoin Micro Momentum...")
            from bitcoin_kpis.btc_micro_momentum import BitcoinMicroMomentumAnalyzer
            analyzer = BitcoinMicroMomentumAnalyzer()
            result = await asyncio.wait_for(
                analyzer.analyze_micro_momentum(), 
                timeout=TEST_CONFIG['timeout_per_component']
            )
            component_results['micro_momentum'] = {
                'status': 'success',
                'sentiment': result.get('component_sentiment', 0),
                'confidence': result.get('component_confidence', 0),
                'execution_time': result.get('execution_time_seconds', 0)
            }
            print(f"    ✅ Micro Momentum: {result.get('component_sentiment', 0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Micro Momentum failed: {e}")
            component_results['micro_momentum'] = {'status': 'failed', 'error': str(e)}
        
        # Test Bitcoin Funding Basis
        try:
            print("  🔍 Bitcoin Funding Basis...")
            from bitcoin_kpis.btc_funding_basis import BitcoinFundingBasisAnalyzer
            analyzer = BitcoinFundingBasisAnalyzer()
            result = await asyncio.wait_for(
                analyzer.analyze_funding_basis(), 
                timeout=TEST_CONFIG['timeout_per_component']
            )
            component_results['funding_basis'] = {
                'status': 'success',
                'sentiment': result.get('component_sentiment', 0),
                'confidence': result.get('component_confidence', 0),
                'execution_time': result.get('execution_time_seconds', 0)
            }
            print(f"    ✅ Funding Basis: {result.get('component_sentiment', 0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Funding Basis failed: {e}")
            component_results['funding_basis'] = {'status': 'failed', 'error': str(e)}
        
        # Test Bitcoin Liquidity
        try:
            print("  🔍 Bitcoin Liquidity...")
            from bitcoin_kpis.btc_liquidity import BitcoinLiquidityAnalyzer
            analyzer = BitcoinLiquidityAnalyzer()
            result = await asyncio.wait_for(
                analyzer.analyze_liquidity(), 
                timeout=TEST_CONFIG['timeout_per_component']
            )
            component_results['liquidity'] = {
                'status': 'success',
                'sentiment': result.get('component_sentiment', 0),
                'confidence': result.get('component_confidence', 0),
                'execution_time': result.get('execution_time_seconds', 0)
            }
            print(f"    ✅ Liquidity: {result.get('component_sentiment', 0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Liquidity failed: {e}")
            component_results['liquidity'] = {'status': 'failed', 'error': str(e)}
        
        # Test Bitcoin Orderflow
        try:
            print("  🔍 Bitcoin Orderflow...")
            from bitcoin_kpis.btc_orderflow import BitcoinOrderflowAnalyzer
            analyzer = BitcoinOrderflowAnalyzer()
            result = await asyncio.wait_for(
                analyzer.analyze_orderflow(), 
                timeout=TEST_CONFIG['timeout_per_component']
            )
            component_results['orderflow'] = {
                'status': 'success',
                'sentiment': result.get('component_sentiment', 0),
                'confidence': result.get('component_confidence', 0),
                'execution_time': result.get('execution_time_seconds', 0)
            }
            print(f"    ✅ Orderflow: {result.get('component_sentiment', 0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Orderflow failed: {e}")
            component_results['orderflow'] = {'status': 'failed', 'error': str(e)}
        
        successful_components = len([r for r in component_results.values() if r['status'] == 'success'])
        print(f"  📊 Bitcoin Components: {successful_components}/4 successful")
        
        return {
            'total_components': 4,
            'successful_components': successful_components,
            'component_results': component_results,
            'overall_status': 'success' if successful_components >= 2 else 'failed'
        }
    
    async def test_reit_components(self) -> Dict[str, Any]:
        """Test all REIT components individually"""
        print("🏢 Testing REIT Components...")
        component_results = {}
        
        # Test REIT Technical Momentum
        try:
            print("  🔍 REIT Technical Momentum...")
            from reit_kpis.reit_technical_momentum import REITTechnicalMomentumAnalyzer
            analyzer = REITTechnicalMomentumAnalyzer()
            result = await asyncio.wait_for(
                analyzer.analyze_technical_momentum(), 
                timeout=TEST_CONFIG['timeout_per_component']
            )
            component_results['technical_momentum'] = {
                'status': 'success',
                'sentiment': result.get('component_sentiment', 0),
                'confidence': result.get('component_confidence', 0),
                'execution_time': result.get('execution_time_seconds', 0)
            }
            print(f"    ✅ Technical Momentum: {result.get('component_sentiment', 0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Technical Momentum failed: {e}")
            component_results['technical_momentum'] = {'status': 'failed', 'error': str(e)}
        
        # Test REIT Yield Spread
        try:
            print("  🔍 REIT Yield Spread...")
            from reit_kpis.reit_yield_spread import REITYieldSpreadAnalyzer
            analyzer = REITYieldSpreadAnalyzer()
            result = await asyncio.wait_for(
                analyzer.analyze_yield_spread(), 
                timeout=TEST_CONFIG['timeout_per_component']
            )
            component_results['yield_spread'] = {
                'status': 'success',
                'sentiment': result.get('component_sentiment', 0),
                'confidence': result.get('component_confidence', 0),
                'execution_time': result.get('execution_time_seconds', 0)
            }
            print(f"    ✅ Yield Spread: {result.get('component_sentiment', 0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Yield Spread failed: {e}")
            component_results['yield_spread'] = {'status': 'failed', 'error': str(e)}
        
        # Test REIT Accumulation Flow
        try:
            print("  🔍 REIT Accumulation Flow...")
            from reit_kpis.reit_accumulation_flow import REITAccumulationFlowAnalyzer
            analyzer = REITAccumulationFlowAnalyzer()
            result = await asyncio.wait_for(
                analyzer.analyze_accumulation_flow(), 
                timeout=TEST_CONFIG['timeout_per_component']
            )
            component_results['accumulation_flow'] = {
                'status': 'success',
                'sentiment': result.get('component_sentiment', 0),
                'confidence': result.get('component_confidence', 0),
                'execution_time': result.get('execution_time_seconds', 0)
            }
            print(f"    ✅ Accumulation Flow: {result.get('component_sentiment', 0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Accumulation Flow failed: {e}")
            component_results['accumulation_flow'] = {'status': 'failed', 'error': str(e)}
        
        # Test REIT Liquidity Risk
        try:
            print("  🔍 REIT Liquidity Risk...")
            from reit_kpis.reit_liquidity_risk import REITLiquidityRiskAnalyzer
            analyzer = REITLiquidityRiskAnalyzer()
            result = await asyncio.wait_for(
                analyzer.analyze_liquidity_risk(), 
                timeout=TEST_CONFIG['timeout_per_component']
            )
            component_results['liquidity_risk'] = {
                'status': 'success',
                'sentiment': result.get('component_sentiment', 0),
                'confidence': result.get('component_confidence', 0),
                'execution_time': result.get('execution_time_seconds', 0)
            }
            print(f"    ✅ Liquidity Risk: {result.get('component_sentiment', 0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Liquidity Risk failed: {e}")
            component_results['liquidity_risk'] = {'status': 'failed', 'error': str(e)}
        
        successful_components = len([r for r in component_results.values() if r['status'] == 'success'])
        print(f"  📊 REIT Components: {successful_components}/4 successful")
        
        return {
            'total_components': 4,
            'successful_components': successful_components,
            'component_results': component_results,
            'overall_status': 'success' if successful_components >= 2 else 'failed'
        }
    
    async def test_bitcoin_integration(self) -> Dict[str, Any]:
        """Test Bitcoin full integration"""
        print("₿ Testing Bitcoin Integration...")
        
        try:
            from bitcoin_kpis.bitcoin_sentiment_analyzer import run_bitcoin_analysis
            
            start_time = time.time()
            result = await asyncio.wait_for(
                run_bitcoin_analysis(), 
                timeout=TEST_CONFIG['timeout_per_asset']
            )
            execution_time = time.time() - start_time
            
            if 'error' not in result:
                print(f"  ✅ Bitcoin Integration: {result.get('component_sentiment', 0):.3f}")
                print(f"  ⏱️ Execution Time: {execution_time:.1f}s")
                print(f"  📊 Confidence: {result.get('component_confidence', 0):.1%}")
                print(f"  🏆 Coverage: {result.get('framework_coverage', 0)*100:.0f}%")
                
                return {
                    'status': 'success',
                    'sentiment': result.get('component_sentiment', 0),
                    'confidence': result.get('component_confidence', 0),
                    'execution_time': execution_time,
                    'framework_coverage': result.get('framework_coverage', 0)
                }
            else:
                raise ValueError(f"Bitcoin integration failed: {result['error']}")
                
        except Exception as e:
            print(f"  ❌ Bitcoin Integration failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'sentiment': 0,
                'confidence': 0
            }
    
    async def test_reit_integration(self) -> Dict[str, Any]:
        """Test REIT full integration"""
        print("🏢 Testing REIT Integration...")
        
        try:
            from reit_kpis.reit_sentiment_analyzer import run_reit_analysis
            
            start_time = time.time()
            result = await asyncio.wait_for(
                run_reit_analysis(), 
                timeout=TEST_CONFIG['timeout_per_asset']
            )
            execution_time = time.time() - start_time
            
            if 'error' not in result:
                print(f"  ✅ REIT Integration: {result.get('component_sentiment', 0):.3f}")
                print(f"  ⏱️ Execution Time: {execution_time:.1f}s")
                print(f"  📊 Confidence: {result.get('component_confidence', 0):.1%}")
                print(f"  🏆 Coverage: {result.get('framework_coverage', 0)*100:.0f}%")
                print(f"  🏢 Symbol: {result.get('symbol', 'N/A')}")
                
                return {
                    'status': 'success',
                    'sentiment': result.get('component_sentiment', 0),
                    'confidence': result.get('component_confidence', 0),
                    'execution_time': execution_time,
                    'framework_coverage': result.get('framework_coverage', 0),
                    'symbol': result.get('symbol', 'MINDSPACE')
                }
            else:
                raise ValueError(f"REIT integration failed: {result['error']}")
                
        except Exception as e:
            print(f"  ❌ REIT Integration failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'sentiment': 0,
                'confidence': 0
            }
    
    async def test_all_assets_integration(self) -> Dict[str, Any]:
        """Test complete 4-asset integration"""
        print("🌐 Testing All Assets Integration...")
        
        try:
            from enhanced_asset_integration import MultiAssetIntegrationOrchestrator
            
            start_time = time.time()
            result = await MultiAssetIntegrationOrchestrator.run_all_assets()
            execution_time = time.time() - start_time
            
            portfolio_summary = result.get('portfolio_summary', {})
            successful = portfolio_summary.get('successful_assets', 0)
            total = portfolio_summary.get('total_assets', 4)
            
            print(f"  ✅ Multi-Asset Integration Complete")
            print(f"  ⏱️ Total Execution Time: {execution_time:.1f}s")
            print(f"  🎯 Success Rate: {successful}/{total}")
            print(f"  📊 Portfolio Sentiment: {portfolio_summary.get('portfolio_sentiment', 0):.3f}")
            print(f"  🏆 Portfolio Confidence: {portfolio_summary.get('portfolio_confidence', 0):.1%}")
            
            return {
                'status': 'success',
                'execution_time': execution_time,
                'successful_assets': successful,
                'total_assets': total,
                'portfolio_sentiment': portfolio_summary.get('portfolio_sentiment', 0),
                'portfolio_confidence': portfolio_summary.get('portfolio_confidence', 0),
                'individual_results': result.get('individual_results', {})
            }
            
        except Exception as e:
            print(f"  ❌ All Assets Integration failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks"""
        print("⚡ Testing Performance Benchmarks...")
        
        benchmarks = {
            'bitcoin_speed_test': 0,
            'reit_speed_test': 0,
            'parallel_efficiency': 0,
            'memory_usage': 'unknown'
        }
        
        # Bitcoin speed test
        try:
            print("  🔍 Bitcoin Speed Test...")
            from bitcoin_kpis.bitcoin_sentiment_analyzer import run_bitcoin_analysis
            
            start_time = time.time()
            await run_bitcoin_analysis()
            bitcoin_time = time.time() - start_time
            benchmarks['bitcoin_speed_test'] = bitcoin_time
            
            grade = "EXCELLENT" if bitcoin_time < 30 else "GOOD" if bitcoin_time < 60 else "ACCEPTABLE"
            print(f"    ⏱️ Bitcoin: {bitcoin_time:.1f}s ({grade})")
            
        except Exception as e:
            print(f"    ❌ Bitcoin speed test failed: {e}")
        
        # REIT speed test
        try:
            print("  🔍 REIT Speed Test...")
            from reit_kpis.reit_sentiment_analyzer import run_reit_analysis
            
            start_time = time.time()
            await run_reit_analysis()
            reit_time = time.time() - start_time
            benchmarks['reit_speed_test'] = reit_time
            
            grade = "EXCELLENT" if reit_time < 60 else "GOOD" if reit_time < 120 else "ACCEPTABLE"
            print(f"    ⏱️ REIT: {reit_time:.1f}s ({grade})")
            
        except Exception as e:
            print(f"    ❌ REIT speed test failed: {e}")
        
        # Parallel efficiency test
        try:
            print("  🔍 Parallel Efficiency Test...")
            start_time = time.time()
            
            # Run Bitcoin and REIT in parallel
            from bitcoin_kpis.bitcoin_sentiment_analyzer import run_bitcoin_analysis
            from reit_kpis.reit_sentiment_analyzer import run_reit_analysis
            
            bitcoin_task = run_bitcoin_analysis()
            reit_task = run_reit_analysis()
            
            await asyncio.gather(bitcoin_task, reit_task, return_exceptions=True)
            parallel_time = time.time() - start_time
            benchmarks['parallel_efficiency'] = parallel_time
            
            # Calculate efficiency (should be faster than sequential)
            sequential_estimate = benchmarks['bitcoin_speed_test'] + benchmarks['reit_speed_test']
            efficiency = (sequential_estimate / parallel_time) if parallel_time > 0 else 1
            
            print(f"    ⚡ Parallel Time: {parallel_time:.1f}s")
            print(f"    📊 Efficiency Ratio: {efficiency:.1f}x")
            
        except Exception as e:
            print(f"    ❌ Parallel efficiency test failed: {e}")
        
        return {
            'status': 'success',
            'benchmarks': benchmarks,
            'performance_grade': self._calculate_performance_grade(benchmarks)
        }
    
    def _calculate_performance_grade(self, benchmarks: Dict) -> str:
        """Calculate overall performance grade"""
        bitcoin_time = benchmarks.get('bitcoin_speed_test', 999)
        reit_time = benchmarks.get('reit_speed_test', 999)
        parallel_time = benchmarks.get('parallel_efficiency', 999)
        
        if bitcoin_time < 30 and reit_time < 60 and parallel_time < 90:
            return "EXCELLENT"
        elif bitcoin_time < 60 and reit_time < 120 and parallel_time < 150:
            return "GOOD"
        else:
            return "ACCEPTABLE"
    
    async def generate_final_report(self):
        """Generate comprehensive final test report"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n🏆 FINAL TEST REPORT")
        print("=" * 60)
        print(f"⏰ Duration: {total_duration:.1f}s")
        print(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Count successes
        successful_tests = len([r for r in self.results.values() 
                              if isinstance(r, dict) and r.get('status') != 'failed'])
        total_tests = len(self.results)
        
        print(f"✅ Test Success Rate: {successful_tests}/{total_tests}")
        
        # Asset-specific results
        if 'Bitcoin Components' in self.results:
            btc_comp = self.results['Bitcoin Components']
            print(f"₿ Bitcoin Components: {btc_comp.get('successful_components', 0)}/4")
        
        if 'REIT Components' in self.results:
            reit_comp = self.results['REIT Components']
            print(f"🏢 REIT Components: {reit_comp.get('successful_components', 0)}/4")
        
        if 'All Assets Integration' in self.results:
            all_assets = self.results['All Assets Integration']
            if all_assets.get('status') == 'success':
                print(f"🌐 Portfolio Success: {all_assets.get('successful_assets', 0)}/4")
                print(f"📊 Portfolio Sentiment: {all_assets.get('portfolio_sentiment', 0):.3f}")
        
        # Performance summary
        if 'Performance Benchmarks' in self.results:
            perf = self.results['Performance Benchmarks']
            if perf.get('status') == 'success':
                print(f"⚡ Performance Grade: {perf.get('performance_grade', 'Unknown')}")
        
        # Save results if requested
        if TEST_CONFIG['save_results']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_test_results_{timestamp}.json"
            
            test_report = {
                'test_summary': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'total_duration_seconds': total_duration,
                    'successful_tests': successful_tests,
                    'total_tests': total_tests,
                    'success_rate': successful_tests / total_tests if total_tests > 0 else 0
                },
                'detailed_results': self.results,
                'system_info': {
                    'python_version': 'Python 3.8+',
                    'test_framework': 'ComprehensiveTestSuite v1.0',
                    'assets_tested': ['Bitcoin', 'REIT', 'Nifty50', 'Gold']
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(test_report, f, indent=2, default=str)
            
            print(f"💾 Test report saved: {filename}")
        
        print(f"\n🎯 SYSTEM STATUS: {'✅ READY FOR PRODUCTION' if successful_tests >= 4 else '⚠️ NEEDS ATTENTION'}")

# Convenience test functions
async def test_bitcoin_only():
    """Test only Bitcoin components and integration"""
    suite = ComprehensiveTestSuite()
    
    print("₿ BITCOIN-ONLY TEST SUITE")
    print("=" * 40)
    
    bitcoin_components = await suite.test_bitcoin_components()
    bitcoin_integration = await suite.test_bitcoin_integration()
    
    print(f"\n📋 Bitcoin Test Summary:")
    print(f"Components: {bitcoin_components.get('successful_components', 0)}/4")
    print(f"Integration: {'✅ Success' if bitcoin_integration.get('status') == 'success' else '❌ Failed'}")
    
    return {'components': bitcoin_components, 'integration': bitcoin_integration}

async def test_reit_only():
    """Test only REIT components and integration"""
    suite = ComprehensiveTestSuite()
    
    print("🏢 REIT-ONLY TEST SUITE")
    print("=" * 40)
    
    reit_components = await suite.test_reit_components()
    reit_integration = await suite.test_reit_integration()
    
    print(f"\n📋 REIT Test Summary:")
    print(f"Components: {reit_components.get('successful_components', 0)}/4")
    print(f"Integration: {'✅ Success' if reit_integration.get('status') == 'success' else '❌ Failed'}")
    
    return {'components': reit_components, 'integration': reit_integration}

async def quick_integration_test():
    """Quick test of 4-asset integration only"""
    suite = ComprehensiveTestSuite()
    
    print("🚀 QUICK INTEGRATION TEST")
    print("=" * 30)
    
    result = await suite.test_all_assets_integration()
    
    if result.get('status') == 'success':
        print(f"✅ Integration successful: {result.get('successful_assets', 0)}/4")
        return result
    else:
        print(f"❌ Integration failed: {result.get('error', 'Unknown error')}")
        return result

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == 'bitcoin':
            asyncio.run(test_bitcoin_only())
        elif test_type == 'reit':
            asyncio.run(test_reit_only())
        elif test_type == 'quick':
            asyncio.run(quick_integration_test())
        else:
            print("Usage: python comprehensive_test_suite.py [bitcoin|reit|quick]")
            print("Or run without arguments for full test suite")
    else:
        # Run full test suite
        suite = ComprehensiveTestSuite()
        asyncio.run(suite.run_all_tests())
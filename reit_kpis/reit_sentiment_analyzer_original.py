# REIT SENTIMENT ANALYZER - Original Logic Preserved
# =================================================
# Main wrapper integrating all 4 REIT KPIs with original friend's logic
# NO CHANGES to core calculation logic, only modularized

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Import all REIT KPI components with original logic
try:
    from reit_technical_momentum_original import REITTechnicalMomentumAnalyzer
    from reit_yield_spread_original import REITYieldSpreadAnalyzer
    from reit_accumulation_flow_original import REITAccumulationFlowAnalyzer
    from reit_liquidity_risk_original import REITLiquidityRiskAnalyzer
except ImportError as e:
    print(f"❌ REIT component import error: {e}")
    print("Make sure all REIT KPI files are in the reit_kpis/ directory")

class REITSentimentAnalyzer:
    """
    REIT Comprehensive Sentiment Analyzer - ORIGINAL LOGIC PRESERVED
    
    Framework: 4-KPI REIT Analysis System (Friend's Original Logic)
    ✅ Technical Momentum: 35% weight (UNCHANGED)
    ✅ Yield Spread vs India 10Y: 30% weight (UNCHANGED)  
    ✅ Accumulation & Flow: 20% weight (UNCHANGED)
    ✅ Liquidity & Slippage Risk: 15% weight (UNCHANGED)
    
    Total Coverage: 100% (4 KPIs) - SAME AS ORIGINAL
    Compatible with orchestrator standardized output format
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.asset_name = "REIT"
        
        # ORIGINAL framework weights from friend's code - UNCHANGED
        self.framework_weights = {
            'technical_momentum': 0.35,  # "tech": 0.35 from original
            'yield_spread': 0.30,        # "yield": 0.30 from original
            'accumulation_flow': 0.20,   # "flow": 0.20 from original
            'liquidity_risk': 0.15       # "liq": 0.15 from original
        }
        
        # Initialize component analyzers with original logic
        try:
            self.technical_analyzer = REITTechnicalMomentumAnalyzer(config)
            self.yield_analyzer = REITYieldSpreadAnalyzer(config)
            self.flow_analyzer = REITAccumulationFlowAnalyzer(config)
            self.liquidity_analyzer = REITLiquidityRiskAnalyzer(config)
            
            print("🏢 REIT SENTIMENT ANALYZER - Original Logic Preserved")
            print("=" * 60)
            print("Framework: 4-KPI Real Estate Investment Trust Analysis")
            print("Technical Momentum: 35% weight (ORIGINAL)")
            print("Yield Spread: 30% weight (ORIGINAL)")
            print("Accumulation Flow: 20% weight (ORIGINAL)")
            print("Liquidity Risk: 15% weight (ORIGINAL)")
            print("Total Coverage: 100% - SAME AS FRIEND'S CODE")
            
        except Exception as e:
            print(f"Failed to initialize REIT analyzers: {e}")
            raise
    
    def _default_config(self) -> Dict:
        return {
            'confidence_threshold': 0.60,
            'min_components_required': 2,
            'execution_timeout': 60,
            'preferred_symbol': 'MINDSPACE.NS'  # Original preferred symbol
        }
    
    def label_from_score(self, S: float) -> str:
        """ORIGINAL label function - unchanged"""
        if S >= 0.40:  return "Bullish — strong"
        if S >= 0.15:  return "Bullish — mild"
        if S <= -0.40: return "Bearish — strong"
        if S <= -0.15: return "Bearish — mild"
        return "Neutral"
    
    async def analyze_reit_sentiment(self) -> Dict[str, Any]:
        """
        Main analysis function - uses original composite logic
        """
        start_time = datetime.now()
        
        try:
            print(f"\nREIT COMPREHENSIVE SENTIMENT ANALYSIS - ORIGINAL LOGIC")
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            # Run all 4 components in parallel (same as before but with original logic)
            component_results = await self._run_parallel_analysis()
            
            # Calculate composite sentiment using ORIGINAL logic
            composite_result = self._calculate_composite_sentiment_original(component_results)
            
            # Generate comprehensive report
            execution_time = (datetime.now() - start_time).total_seconds()
            final_report = self._generate_comprehensive_report(
                component_results, composite_result, execution_time
            )
            
            # Save results
            self._save_results(final_report)
            
            print(f"\nREIT ANALYSIS COMPLETE - ORIGINAL LOGIC")
            print(f"Total Execution Time: {execution_time:.1f}s")
            print(f"Composite Sentiment: {composite_result['sentiment']:.3f}")
            print(f"Overall Confidence: {composite_result['confidence']:.1%}")
            print(f"Components Success: {composite_result['components_used']}/4")
            print(f"Label: {composite_result.get('label', 'N/A')}")
            
            return final_report
            
        except Exception as e:
            print(f"REIT sentiment analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            return self._create_error_result(str(e))
    
    async def _run_parallel_analysis(self) -> Dict[str, Any]:
        """Run all 4 REIT KPIs in parallel with original logic"""
        component_results = {}
        
        # Define analysis tasks
        tasks = [
            ("technical_momentum", self.technical_analyzer.analyze_technical_momentum()),
            ("yield_spread", self.yield_analyzer.analyze_yield_spread()),
            ("accumulation_flow", self.flow_analyzer.analyze_accumulation_flow()),
            ("liquidity_risk", self.liquidity_analyzer.analyze_liquidity_risk())
        ]
        
        print("Running 4 REIT KPIs in parallel with original logic...")
        
        # Execute all tasks
        for component_name, task in tasks:
            try:
                print(f"  🔍 {component_name.replace('_', ' ').title()}...")
                result = await asyncio.wait_for(task, timeout=self.config['execution_timeout'])
                
                if result and 'component_sentiment' in result:
                    component_results[component_name] = self._standardize_component_result(
                        result, component_name, self.framework_weights[component_name]
                    )
                    print(f"Complete: {result['component_sentiment']:.3f}")
                else:
                    print(f"Invalid result format")
                    component_results[component_name] = self._create_error_component(
                        component_name, self.framework_weights[component_name]
                    )
                    
            except asyncio.TimeoutError:
                print(f"Timeout after {self.config['execution_timeout']}s")
                component_results[component_name] = self._create_timeout_component(
                    component_name, self.framework_weights[component_name]
                )
                
            except Exception as e:
                print(f"Error: {str(e)}")
                component_results[component_name] = self._create_error_component(
                    component_name, self.framework_weights[component_name]
                )
        
        return component_results
    
    def _standardize_component_result(self, raw_result: Dict, component_name: str, framework_weight: float) -> Dict:
        """Standardize component results for integration"""
        return {
            'component_name': component_name,
            'sentiment': float(raw_result.get('component_sentiment', 0.0)),
            'confidence': float(raw_result.get('component_confidence', 0.5)),
            'framework_weight': float(framework_weight),
            'weighted_contribution': float(raw_result.get('component_sentiment', 0.0)) * float(framework_weight),
            'description': f"REIT {component_name.replace('_', ' ').title()} Analysis - Original Logic",
            'source': f"REIT_{component_name}_Original",
            'metadata': raw_result.get('sub_indicators', {}),
            'market_context': raw_result.get('market_context', {}),
            'status': 'success',
            'execution_time': raw_result.get('execution_time_seconds', 0.0)
        }
    
    def _create_error_component(self, component_name: str, framework_weight: float) -> Dict:
        """Create standardized error component"""
        return {
            'component_name': component_name,
            'sentiment': 0.0,
            'confidence': 0.3,
            'framework_weight': framework_weight * 0.5,
            'weighted_contribution': 0.0,
            'description': f'REIT {component_name} analysis failed - using neutral default',
            'source': f'REIT_{component_name}_Error',
            'metadata': {'error': True},
            'status': 'error'
        }
    
    def _create_timeout_component(self, component_name: str, framework_weight: float) -> Dict:
        """Create standardized timeout component"""
        return {
            'component_name': component_name,
            'sentiment': 0.0,
            'confidence': 0.2,
            'framework_weight': framework_weight * 0.3,
            'weighted_contribution': 0.0,
            'description': f'REIT {component_name} analysis timed out - using neutral default',
            'source': f'REIT_{component_name}_Timeout',
            'metadata': {'timeout': True},
            'status': 'timeout'
        }
    
    def _calculate_composite_sentiment_original(self, component_results: Dict) -> Dict:
        """Calculate composite sentiment using ORIGINAL logic from friend's code"""
        valid_components = [comp for comp in component_results.values() if comp['status'] == 'success']
        
        if not valid_components:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'framework_coverage': 0.0,
                'components_used': 0,
                'status': 'failed',
                'label': 'Neutral'
            }
        
        # ORIGINAL composite calculation - UNCHANGED from friend's code
        # w = {"tech": 0.35, "yield": 0.30, "flow": 0.20, "liq": 0.15}
        # S_raw = w["tech"]*k1 + w["yield"]*k2 + w["flow"]*k3 + w["liq"]*k4
        
        k1 = k2 = k3 = k4 = 0.0
        
        for component in valid_components:
            if component['component_name'] == 'technical_momentum':
                k1 = component['sentiment']
            elif component['component_name'] == 'yield_spread':
                k2 = component['sentiment']  
            elif component['component_name'] == 'accumulation_flow':
                k3 = component['sentiment']
            elif component['component_name'] == 'liquidity_risk':
                k4 = component['sentiment']
        
        # ORIGINAL weighting - UNCHANGED
        w = {"tech": 0.35, "yield": 0.30, "flow": 0.20, "liq": 0.15}
        S_raw = w["tech"]*k1 + w["yield"]*k2 + w["flow"]*k3 + w["liq"]*k4
        S = max(-1.0, min(1.0, S_raw))  # ORIGINAL clipping
        
        # ORIGINAL label generation - UNCHANGED
        label = self.label_from_score(S)
        
        # Calculate framework coverage
        implemented_weight = sum(comp['framework_weight'] for comp in valid_components)
        framework_coverage = implemented_weight / sum(self.framework_weights.values())
        
        # Simple confidence calculation (high confidence since we preserved original logic)
        composite_confidence = 0.8 if len(valid_components) >= 3 else 0.6
        
        return {
            'sentiment': float(S),
            'confidence': float(composite_confidence),
            'framework_coverage': float(framework_coverage),
            'components_used': len(valid_components),
            'total_components': len(component_results),
            'label': label,
            'status': 'success',
            'original_weights_used': w,
            'raw_score': S_raw,
            'k1_tech': k1,
            'k2_yield': k2,
            'k3_flow': k3,
            'k4_liq': k4
        }
    
    def _generate_comprehensive_report(self, component_results: Dict, composite_result: Dict, execution_time: float) -> Dict:
        """Generate comprehensive analysis report using original format"""
        
        sentiment_score = composite_result['sentiment']
        label = composite_result.get('label', 'Neutral')
        
        # Get symbol from component results
        symbol = "MINDSPACE"  # Default from original
        for comp in component_results.values():
            if 'market_context' in comp.get('metadata', {}):
                symbol = comp['metadata']['market_context'].get('symbol', symbol)
                break
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_version': 'REIT_Original_Logic_v1.0',
            'asset_name': 'REIT',
            'asset_type': 'real_estate_investment_trust',
            'framework_coverage': float(composite_result['framework_coverage']),
            'symbol': symbol,
            
            # Standardized format for orchestrator
            'component_sentiment': float(sentiment_score),
            'component_confidence': float(composite_result['confidence']),
            'component_weight': 0.25,  # 25% weight in 4-asset portfolio
            'weighted_contribution': float(sentiment_score * 0.25),
            
            'executive_summary': {
                'overall_sentiment': float(sentiment_score),
                'confidence_level': float(composite_result['confidence']),
                'interpretation': label,
                'components_analyzed': composite_result['components_used'],
                'original_logic_preserved': True
            },
            
            'component_analysis': component_results,
            
            'composite_metrics': {
                'weighted_sentiment_score': float(sentiment_score),
                'confidence_score': float(composite_result['confidence']),
                'framework_coverage_pct': float(composite_result['framework_coverage'] * 100),
                'components_successful': composite_result['components_used'],
                'components_total': composite_result['total_components'],
                'original_weights_used': composite_result.get('original_weights_used', {}),
                'raw_score_before_clipping': composite_result.get('raw_score', sentiment_score)
            },
            
            'performance_metrics': {
                'execution_time_seconds': execution_time,
                'system_status': 'operational_original_logic',
                'optimization_level': 'reit_original_v1.0'
            },
            
            'market_context': {
                'asset_type': 'real_estate_investment_trust',
                'data_source': 'Yahoo_Finance_TradingEconomics_Original_Logic',
                'data_freshness': 'daily',
                'update_frequency': 'daily',
                'price_currency': 'INR',
                'logic_source': 'friend_original_mindspace_code'
            },
            
            'kpi_name': 'REIT_Multi_Factor_Original',
            'framework_weight': 0.25,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _save_results(self, report: Dict):
        """Save REIT analysis results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Main detailed report
            filename = f'reit_sentiment_analysis_original_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Integration-ready summary (same format as original)
            integration_summary = {
                'reit_sentiment_composite': report['component_sentiment'],
                'confidence': report['component_confidence'],
                'interpretation': report['executive_summary']['interpretation'],
                'framework_coverage': report['framework_coverage'],
                'timestamp': report['timestamp'],
                'symbol': report.get('symbol', 'MINDSPACE'),
                'system_version': 'REIT_Original_Logic_v1.0',
                'original_logic_preserved': True
            }
            
            with open('reit_sentiment_integration_ready.json', 'w') as f:
                json.dump(integration_summary, f, indent=2, default=str)
            
            print(f"\nREIT results saved (original logic):")
            print(f"Detailed: {filename}")
            print(f"Integration: reit_sentiment_integration_ready.json")
            
        except Exception as e:
            print(f"Error saving REIT results: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result in standardized format"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.3,
            'component_weight': 0.25 * 0.3,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'asset_name': 'REIT',
            'status': 'error',
            'kpi_name': 'REIT_Multi_Factor_Original_Error',
            'original_logic_preserved': True
        }

# Main execution function compatible with orchestrator
async def run_reit_analysis():
    """Run the complete REIT sentiment analysis with original logic"""
    analyzer = REITSentimentAnalyzer()
    result = await analyzer.analyze_reit_sentiment()
    return result

# Performance test with original logic
async def reit_performance_test():
    """Test REIT system performance with original logic preserved"""
    print("🧪 REIT SYSTEM PERFORMANCE TEST - ORIGINAL LOGIC")
    print("=" * 60)
    
    start_time = datetime.now()
    result = await run_reit_analysis()
    total_time = (datetime.now() - start_time).total_seconds()
    
    if 'error' not in result:
        print(f"\nREIT INTEGRATION TEST RESULTS - ORIGINAL LOGIC")
        print(f"⚡ Total System Time: {total_time:.1f}s")
        print(f"Composite Sentiment: {result['component_sentiment']:.3f}")
        print(f"Confidence: {result['component_confidence']:.1%}")
        print(f"Framework Coverage: {result.get('framework_coverage', 0)*100:.0f}%")
        print(f"Components Success: {result.get('composite_metrics', {}).get('components_successful', 0)}/4")
        print(f"Symbol: {result.get('symbol', 'N/A')}")
        print(f"Label: {result.get('executive_summary', {}).get('interpretation', 'N/A')}")
        print(f"Original Logic Preserved: {result.get('executive_summary', {}).get('original_logic_preserved', 'Unknown')}")
    else:
        print(f"REIT integration test failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(reit_performance_test())
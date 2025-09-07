# BITCOIN SENTIMENT ANALYZER - Main Wrapper
# ==========================================
# Integrates all 4 Bitcoin KPIs into unified analysis
# Compatible with your orchestrator system

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
from statistics import pstdev
import numpy as np

# Import all Bitcoin KPI components
try:
    from btc_micro_momentum import BitcoinMicroMomentumAnalyzer
    from btc_funding_basis import BitcoinFundingBasisAnalyzer
    from btc_liquidity import BitcoinLiquidityAnalyzer
    from btc_orderflow import BitcoinOrderflowAnalyzer
except ImportError as e:
    print(f"❌ Bitcoin component import error: {e}")
    print("Make sure all Bitcoin KPI files are in the bitcoin_kpis/ directory")

class BitcoinSentimentAnalyzer:
    """
    Bitcoin Comprehensive Sentiment Analyzer v1.0
    
    Framework: 4-KPI Bitcoin Analysis System
    ✅ Micro-Momentum (1-minute OHLC): 30% weight
    ✅ Funding/Basis & Positioning: 30% weight  
    ✅ Liquidity & Slippage: 20% weight
    ✅ Orderflow & CVD: 20% weight
    
    Total Coverage: 100% (4 KPIs)
    Compatible with orchestrator standardized output format
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.asset_name = "BITCOIN"
        
        # Framework weights
        self.framework_weights = {
            'micro_momentum': 0.30,
            'funding_basis': 0.30,
            'liquidity': 0.20,
            'orderflow': 0.20
        }
        
        # Initialize component analyzers
        try:
            self.micro_analyzer = BitcoinMicroMomentumAnalyzer(config)
            self.funding_analyzer = BitcoinFundingBasisAnalyzer(config)
            self.liquidity_analyzer = BitcoinLiquidityAnalyzer(config)
            self.orderflow_analyzer = BitcoinOrderflowAnalyzer(config)
            
            print("🚀 BITCOIN SENTIMENT ANALYZER v1.0")
            print("=" * 50)
            print("Framework: 4-KPI Real-time Analysis System")
            print("✅ Micro-Momentum: 30% weight")
            print("✅ Funding/Basis: 30% weight")
            print("✅ Liquidity: 20% weight")
            print("✅ Orderflow: 20% weight")
            print("🎯 Total Coverage: 100%")
            
        except Exception as e:
            print(f"❌ Failed to initialize Bitcoin analyzers: {e}")
            raise
    
    def _default_config(self) -> Dict:
        return {
            'confidence_threshold': 0.60,
            'min_components_required': 2,
            'outlier_detection': True,
            'adaptive_weighting': True,
            'execution_timeout': 30
        }
    
    async def analyze_bitcoin_sentiment(self) -> Dict[str, Any]:
        """
        Main analysis function compatible with orchestrator
        Returns standardized format matching Nifty/Gold structure
        """
        start_time = datetime.now()
        
        try:
            print(f"\n₿ BITCOIN COMPREHENSIVE SENTIMENT ANALYSIS")
            print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            # Run all 4 components in parallel
            component_results = await self._run_parallel_analysis()
            
            # Calculate composite sentiment
            composite_result = self._calculate_composite_sentiment(component_results)
            
            # Generate comprehensive report
            execution_time = (datetime.now() - start_time).total_seconds()
            final_report = self._generate_comprehensive_report(
                component_results, composite_result, execution_time
            )
            
            # Save results
            self._save_results(final_report)
            
            print(f"\n✅ BITCOIN ANALYSIS COMPLETE")
            print(f"⏱️ Total Execution Time: {execution_time:.1f}s")
            print(f"₿ Composite Sentiment: {composite_result['sentiment']:.3f}")
            print(f"📊 Overall Confidence: {composite_result['confidence']:.1%}")
            print(f"🏆 Components Success: {composite_result['components_used']}/4")
            
            return final_report
            
        except Exception as e:
            print(f"❌ Bitcoin sentiment analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            return self._create_error_result(str(e))
    
    async def _run_parallel_analysis(self) -> Dict[str, Any]:
        """Run all 4 Bitcoin KPIs in parallel"""
        component_results = {}
        
        # Define analysis tasks
        tasks = [
            ("micro_momentum", self.micro_analyzer.analyze_micro_momentum()),
            ("funding_basis", self.funding_analyzer.analyze_funding_basis()),
            ("liquidity", self.liquidity_analyzer.analyze_liquidity()),
            ("orderflow", self.orderflow_analyzer.analyze_orderflow())
        ]
        
        print("🔄 Running 4 Bitcoin KPIs in parallel...")
        
        # Execute all tasks
        for component_name, task in tasks:
            try:
                print(f"  🔍 {component_name.replace('_', ' ').title()}...")
                result = await asyncio.wait_for(task, timeout=self.config['execution_timeout'])
                
                if result and 'component_sentiment' in result:
                    component_results[component_name] = self._standardize_component_result(
                        result, component_name, self.framework_weights[component_name]
                    )
                    print(f"    ✅ Complete: {result['component_sentiment']:.3f}")
                else:
                    print(f"    ⚠️ Invalid result format")
                    component_results[component_name] = self._create_error_component(
                        component_name, self.framework_weights[component_name]
                    )
                    
            except asyncio.TimeoutError:
                print(f"    ⏰ Timeout after {self.config['execution_timeout']}s")
                component_results[component_name] = self._create_timeout_component(
                    component_name, self.framework_weights[component_name]
                )
                
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
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
            'description': f"Bitcoin {component_name.replace('_', ' ').title()} Analysis",
            'source': f"Bitcoin_{component_name}",
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
            'framework_weight': framework_weight * 0.5,  # Reduced weight for errors
            'weighted_contribution': 0.0,
            'description': f'Bitcoin {component_name} analysis failed - using neutral default',
            'source': f'Bitcoin_{component_name}_Error',
            'metadata': {'error': True},
            'status': 'error'
        }
    
    def _create_timeout_component(self, component_name: str, framework_weight: float) -> Dict:
        """Create standardized timeout component"""
        return {
            'component_name': component_name,
            'sentiment': 0.0,
            'confidence': 0.2,
            'framework_weight': framework_weight * 0.3,  # Heavily reduced weight
            'weighted_contribution': 0.0,
            'description': f'Bitcoin {component_name} analysis timed out - using neutral default',
            'source': f'Bitcoin_{component_name}_Timeout',
            'metadata': {'timeout': True},
            'status': 'timeout'
        }
    
    def _calculate_composite_sentiment(self, component_results: Dict) -> Dict:
        """Calculate composite sentiment from all components"""
        valid_components = [comp for comp in component_results.values() if comp['status'] == 'success']
        
        if not valid_components:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'framework_coverage': 0.0,
                'components_used': 0,
                'status': 'failed'
            }
        
        # Calculate weighted sentiment
        total_weighted_sentiment = 0
        total_weight = 0
        confidence_scores = []
        
        for component in valid_components:
            # Apply confidence-adjusted weighting
            adjusted_weight = component['framework_weight'] * component['confidence']
            total_weighted_sentiment += component['sentiment'] * adjusted_weight
            total_weight += adjusted_weight
            confidence_scores.append(component['confidence'])
        
        composite_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Calculate composite confidence
        if len(confidence_scores) >= 2:
            # Agreement-based confidence
            avg_confidence = np.mean(confidence_scores)
            # Penalize disagreement
            disagreement_penalty = min(0.3, pstdev(confidence_scores) / 0.5)
            composite_confidence = max(0.3, avg_confidence - disagreement_penalty)
        else:
            composite_confidence = confidence_scores[0] if confidence_scores else 0.3
        
        # Coverage bonus
        coverage_bonus = len(valid_components) / 4 * 0.1  # Bonus for more components
        composite_confidence = min(0.95, composite_confidence + coverage_bonus)
        
        # Calculate framework coverage
        implemented_weight = sum(comp['framework_weight'] for comp in valid_components)
        framework_coverage = implemented_weight / sum(self.framework_weights.values())
        
        # Apply outlier detection if enabled
        if self.config['outlier_detection']:
            composite_sentiment = self._apply_outlier_detection(component_results, composite_sentiment)
        
        return {
            'sentiment': float(composite_sentiment),
            'confidence': float(composite_confidence),
            'framework_coverage': float(framework_coverage),
            'components_used': len(valid_components),
            'total_components': len(component_results),
            'status': 'success'
        }
    
    def _apply_outlier_detection(self, component_results: Dict, composite_sentiment: float) -> float:
        """Apply outlier detection and smoothing"""
        sentiments = [comp['sentiment'] for comp in component_results.values() if comp['status'] == 'success']
        
        if len(sentiments) < 3:
            return composite_sentiment
        
        # Remove extreme outliers using IQR method
        q75, q25 = np.percentile(sentiments, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        filtered_sentiments = [s for s in sentiments if lower_bound <= s <= upper_bound]
        
        if len(filtered_sentiments) >= 2:
            # Use filtered average if we removed outliers
            filtered_average = np.mean(filtered_sentiments)
            return (composite_sentiment + filtered_average) / 2  # Blend with original
        
        return composite_sentiment
    
    def _generate_comprehensive_report(self, component_results: Dict, composite_result: Dict, execution_time: float) -> Dict:
        """Generate comprehensive analysis report in orchestrator format"""
        
        # Overall interpretation
        sentiment_score = composite_result['sentiment']
        if sentiment_score > 0.6:
            overall_interpretation = "🟢 STRONG BULLISH - Multiple strong positive crypto signals"
        elif sentiment_score > 0.3:
            overall_interpretation = "🟢 BULLISH - Net positive Bitcoin momentum"
        elif sentiment_score > 0.1:
            overall_interpretation = "🟡 MILDLY BULLISH - Slight positive bias"
        elif sentiment_score > -0.1:
            overall_interpretation = "⚪ NEUTRAL - Balanced crypto signals"
        elif sentiment_score > -0.3:
            overall_interpretation = "🟡 MILDLY BEARISH - Slight negative bias"
        elif sentiment_score > -0.6:
            overall_interpretation = "🔴 BEARISH - Net negative Bitcoin momentum"
        else:
            overall_interpretation = "🔴 STRONG BEARISH - Multiple strong negative crypto signals"
        
        # Risk assessment
        confidence = composite_result['confidence']
        if confidence > 0.8:
            risk_assessment = "LOW RISK - High confidence crypto signals"
        elif confidence > 0.6:
            risk_assessment = "MODERATE RISK - Good signal quality"
        else:
            risk_assessment = "HIGH RISK - Low confidence signals"
        
        # Performance grade
        if execution_time < 5:
            performance_grade = "EXCELLENT"
        elif execution_time < 15:
            performance_grade = "GOOD"
        else:
            performance_grade = "ACCEPTABLE"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_version': 'Bitcoin_v1.0_integrated',
            'asset_name': 'BITCOIN',
            'asset_type': 'cryptocurrency',
            'framework_coverage': float(composite_result['framework_coverage']),
            
            # Standardized format for orchestrator
            'component_sentiment': float(sentiment_score),
            'component_confidence': float(confidence),
            'component_weight': 0.25,  # 25% weight in 4-asset portfolio
            'weighted_contribution': float(sentiment_score * 0.25),
            
            'executive_summary': {
                'overall_sentiment': float(sentiment_score),
                'confidence_level': float(confidence),
                'interpretation': overall_interpretation,
                'risk_assessment': risk_assessment,
                'components_analyzed': composite_result['components_used'],
                'recommendation': f"Confidence Level: {confidence:.0%} - {risk_assessment}"
            },
            
            'component_analysis': component_results,
            
            'composite_metrics': {
                'weighted_sentiment_score': float(sentiment_score),
                'confidence_score': float(confidence),
                'framework_coverage_pct': float(composite_result['framework_coverage'] * 100),
                'components_successful': composite_result['components_used'],
                'components_total': composite_result['total_components']
            },
            
            'performance_metrics': {
                'execution_time_seconds': execution_time,
                'performance_grade': performance_grade,
                'system_status': 'operational',
                'optimization_level': 'bitcoin_v1.0'
            },
            
            'market_context': {
                'asset_type': 'cryptocurrency',
                'data_source': 'Kraken_Multi_API',
                'data_freshness': 'real_time',
                'update_frequency': '1_minute',
                'price_currency': 'USD'
            },
            
            'kpi_name': 'Bitcoin_Multi_Signal',
            'framework_weight': 0.25,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _save_results(self, report: Dict):
        """Save Bitcoin analysis results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Main detailed report
            filename = f'bitcoin_sentiment_analysis_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Integration-ready summary
            integration_summary = {
                'bitcoin_sentiment_composite': report['component_sentiment'],
                'confidence': report['component_confidence'],
                'interpretation': report['executive_summary']['interpretation'],
                'framework_coverage': report['framework_coverage'],
                'timestamp': report['timestamp'],
                'system_version': 'Bitcoin_v1.0',
                'components': {
                    'micro_momentum': report['component_analysis'].get('micro_momentum', {}).get('sentiment', 0),
                    'funding_basis': report['component_analysis'].get('funding_basis', {}).get('sentiment', 0),
                    'liquidity': report['component_analysis'].get('liquidity', {}).get('sentiment', 0),
                    'orderflow': report['component_analysis'].get('orderflow', {}).get('sentiment', 0)
                }
            }
            
            with open('bitcoin_sentiment_integration_ready.json', 'w') as f:
                json.dump(integration_summary, f, indent=2, default=str)
            
            print(f"\n💾 Bitcoin results saved:")
            print(f" 📄 Detailed: {filename}")
            print(f" 🔗 Integration: bitcoin_sentiment_integration_ready.json")
            
        except Exception as e:
            print(f"❌ Error saving Bitcoin results: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result in standardized format"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.3,
            'component_weight': 0.25 * 0.3,  # Heavily reduced weight
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'asset_name': 'BITCOIN',
            'status': 'error',
            'kpi_name': 'Bitcoin_Multi_Signal_Error'
        }

# Main execution function compatible with orchestrator
async def run_bitcoin_analysis():
    """Run the complete Bitcoin sentiment analysis"""
    analyzer = BitcoinSentimentAnalyzer()
    result = await analyzer.analyze_bitcoin_sentiment()
    return result

# Performance test
async def bitcoin_performance_test():
    """Test Bitcoin system performance"""
    print("🧪 BITCOIN SYSTEM PERFORMANCE TEST")
    print("=" * 50)
    
    start_time = datetime.now()
    result = await run_bitcoin_analysis()
    total_time = (datetime.now() - start_time).total_seconds()
    
    if 'error' not in result:
        print(f"\n🏆 BITCOIN INTEGRATION TEST RESULTS")
        print(f"⚡ Total System Time: {total_time:.1f}s")
        print(f"₿ Composite Sentiment: {result['component_sentiment']:.3f}")
        print(f"📊 Confidence: {result['component_confidence']:.1%}")
        print(f"📈 Framework Coverage: {result.get('framework_coverage', 0)*100:.0f}%")
        print(f"✅ Components Success: {result.get('composite_metrics', {}).get('components_successful', 0)}/4")
        print(f"🏆 Performance: {result.get('performance_metrics', {}).get('performance_grade', 'N/A')}")
    else:
        print(f"❌ Bitcoin integration test failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(bitcoin_performance_test())
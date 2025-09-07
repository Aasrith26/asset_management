# INTEGRATED GOLD SENTIMENT SYSTEM v5.0 
# =====================================
# Complete framework with all 5 major KPIs (95% coverage)
# Integrates: Technical Analysis + Real Rates + Central Banks + COT + Currency Debasement

import json
import numpy as np
from datetime import datetime
from typing import Dict
import os
# Import all analyzers (assumes they're in the same directory or gold_kpis/)
try:
    from gold_momentum_analyzer import GoldMomentumAnalyzer
    from real_interest_rate_sentiment import RealRateSentimentScorer
    from optimized_real_time_enterprise import OptimizedRealTimeGoldSentimentAnalyzer
    from gold_kpis.cot_positioning_analyzer import ProfessionalPositioningAnalyzer
    from gold_kpis.currency_debasement_analyzer import CurrencyDebasementAnalyzer
except ImportError as e:
    print(f"⚠️ Import Error: {e}")
    print("Make sure all analyzer modules are available")

class IntegratedGoldSentimentSystem:
    """
    Complete Integrated Gold Sentiment Analysis System v5.0
    
    Framework Coverage:
    ✅ Technical Analysis (Gold Price Momentum): 25%
    ✅ Real Interest Rates: 25% 
    ✅ Central Bank Buying Sentiment: 20%
    ✅ Professional Positioning (COT): 15%
    ✅ Currency Debasement Risk: 10%
    🔮 Geopolitical/Crisis Risk: 5% (Future)
    
    Total: 95% Framework Coverage
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.framework_weights = {
            'technical_analysis': 0.25,
            'real_interest_rates': 0.25,
            'central_bank_sentiment': 0.20,
            'professional_positioning': 0.15,
            'currency_debasement': 0.10,
            'geopolitical_risk': 0.05  # Reserved for future
        }
        
        print("🚀 INTEGRATED GOLD SENTIMENT SYSTEM v5.0")
        print("=" * 60)
        print("Framework Coverage: 95% (5 major KPIs implemented)")
        print("🎯 Enterprise-grade multi-component analysis")
        
    def _default_config(self) -> Dict:
        return {
            'confidence_threshold': 0.60,
            'min_components_required': 3,
            'outlier_detection': True,
            'adaptive_weighting': True,
            'performance_optimization': True,
            'fred_api_key': None,  # Set your FRED API key here or in environment
            'execution_timeout': 30,  # seconds
        }
    
    async def run_comprehensive_analysis(self) -> Dict:
        """Run complete gold sentiment analysis with all components"""
        print(f"\n🔍 COMPREHENSIVE GOLD SENTIMENT ANALYSIS")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        start_time = datetime.now()
        component_results = {}
        
        try:
            # Component 1: Technical Analysis (25% weight)
            print("1️⃣ Analyzing Technical Indicators (Gold Price Momentum)...")
            technical_analyzer = GoldMomentumAnalyzer()
            technical_result = technical_analyzer.analyze_momentum()
            component_results['technical_analysis'] = self._standardize_component_result(
                technical_result, 'Technical_Analysis', self.framework_weights['technical_analysis']
            )
            
            # Component 2: Real Interest Rates (25% weight) 
            print("\n2️⃣ Analyzing Real Interest Rates...")
            real_rates_analyzer = RealRateSentimentScorer(api_key=self.config.get('fred_api_key'))
            real_rates_data, regime = real_rates_analyzer.run_optimized_analysis()
            if real_rates_data is not None and len(real_rates_data) > 0:
                latest_real_rates = real_rates_data.iloc[-1]
                real_rates_result = {
                    'sentiment': float(latest_real_rates['sentiment_composite']),
                    'confidence': float(latest_real_rates['confidence_composite']),
                    'interpretation': latest_real_rates['sentiment_interpretation'],
                    'real_rate_value': float(latest_real_rates['real_rate'])
                }
                component_results['real_interest_rates'] = self._standardize_component_result(
                    real_rates_result, 'Real_Interest_Rates', self.framework_weights['real_interest_rates']
                )
            else:
                component_results['real_interest_rates'] = self._create_error_component(
                    'Real_Interest_Rates', self.framework_weights['real_interest_rates']
                )
            
            # Component 3: Central Bank Sentiment (20% weight)
            print("\n3️⃣ Analyzing Central Bank Buying Sentiment...")
            cb_analyzer = OptimizedRealTimeGoldSentimentAnalyzer()
            cb_result = await cb_analyzer.run_optimized_comprehensive_analysis()
            if 'error' not in cb_result:
                cb_sentiment_result = {
                    'sentiment': float(cb_result['sentiment_analysis']['sentiment_score']),
                    'confidence': float(cb_result['sentiment_analysis']['confidence']),
                    'interpretation': cb_result['sentiment_analysis']['interpretation'],
                    'sources_available': cb_result['data_quality']['sources_available']
                }
                component_results['central_bank_sentiment'] = self._standardize_component_result(
                    cb_sentiment_result, 'Central_Bank_Sentiment', self.framework_weights['central_bank_sentiment']
                )
            else:
                component_results['central_bank_sentiment'] = self._create_error_component(
                    'Central_Bank_Sentiment', self.framework_weights['central_bank_sentiment']
                )
            
            # Component 4: Professional Positioning COT (15% weight)
            print("\n4️⃣ Analyzing Professional Positioning (COT)...")
            cot_analyzer = ProfessionalPositioningAnalyzer()
            cot_result = await cot_analyzer.analyze_professional_positioning()
            component_results['professional_positioning'] = self._standardize_component_result(
                cot_result, 'Professional_Positioning_COT', self.framework_weights['professional_positioning']
            )
            
            # Component 5: Currency Debasement Risk (10% weight) 
            print("\n5️⃣ Analyzing Currency Debasement Risk...")
            currency_analyzer = CurrencyDebasementAnalyzer(fred_api_key=self.config.get('fred_api_key'))
            currency_result = await currency_analyzer.analyze_currency_debasement_risk()
            component_results['currency_debasement'] = self._standardize_component_result(
                currency_result, 'Currency_Debasement_Risk', self.framework_weights['currency_debasement']
            )
            
            # Calculate composite sentiment
            composite_result = self._calculate_composite_sentiment(component_results)
            
            # Generate comprehensive report
            execution_time = (datetime.now() - start_time).total_seconds()
            final_report = self._generate_comprehensive_report(
                component_results, composite_result, execution_time
            )
            
            # Save results
            self._save_integrated_results(final_report)
            
            print(f"\n✅ ANALYSIS COMPLETE")
            print(f"⏱️ Total Execution Time: {execution_time:.1f}s")
            print(f"🎯 Composite Sentiment: {composite_result['sentiment']:.3f}")
            print(f"📊 Overall Confidence: {composite_result['confidence']:.1%}")
            print(f"🏆 Framework Coverage: {composite_result['framework_coverage']:.1%}")
            
            return final_report
            
        except Exception as e:
            print(f"❌ Comprehensive analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'partial_results': component_results
            }
    
    def _standardize_component_result(self, raw_result: Dict, component_name: str, framework_weight: float) -> Dict:
        """Standardize component results for integration"""
        if not raw_result or 'error' in raw_result:
            return self._create_error_component(component_name, framework_weight)
        
        # Extract standardized fields
        sentiment = raw_result.get('sentiment', raw_result.get('sentiment_score', 0))
        confidence = raw_result.get('confidence', 0.5)
        
        # Handle different result formats
        if isinstance(sentiment, str):
            sentiment = 0.0
        if isinstance(confidence, str):
            confidence = 0.5
            
        return {
            'component_name': component_name,
            'sentiment': float(sentiment),
            'confidence': float(confidence),
            'framework_weight': float(framework_weight),
            'weighted_contribution': float(sentiment) * float(framework_weight),
            'description': raw_result.get('description', raw_result.get('interpretation', f'{component_name} analysis')),
            'source': raw_result.get('source', component_name),
            'metadata': raw_result.get('metadata', {}),
            'status': 'success'
        }
    
    def _create_error_component(self, component_name: str, framework_weight: float) -> Dict:
        """Create standardized error component"""
        return {
            'component_name': component_name,
            'sentiment': 0.0,
            'confidence': 0.3,
            'framework_weight': framework_weight * 0.5,  # Reduced weight for errors
            'weighted_contribution': 0.0,
            'description': f'{component_name} analysis failed - using neutral default',
            'source': f'{component_name}_Error',
            'metadata': {'error': True},
            'status': 'error'
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
        avg_confidence = np.mean(confidence_scores)
        coverage_bonus = len(valid_components) / 5 * 0.1  # Bonus for more components
        composite_confidence = min(0.95, avg_confidence + coverage_bonus)
        
        # Calculate framework coverage
        implemented_weight = sum(comp['framework_weight'] for comp in valid_components)
        total_possible_weight = sum(self.framework_weights.values())
        framework_coverage = implemented_weight / total_possible_weight
        
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
        
        # Remove extreme outliers
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
        """Generate comprehensive analysis report"""
        
        # Overall interpretation
        sentiment_score = composite_result['sentiment']
        if sentiment_score > 0.6:
            overall_interpretation = "🟢 STRONG BULLISH - Multiple strong positive signals"
        elif sentiment_score > 0.3:
            overall_interpretation = "🟢 BULLISH - Net positive signals across components"
        elif sentiment_score > 0.1:
            overall_interpretation = "🟡 MILDLY BULLISH - Slight positive bias"
        elif sentiment_score > -0.1:
            overall_interpretation = "⚪ NEUTRAL - Balanced signals"
        elif sentiment_score > -0.3:
            overall_interpretation = "🟡 MILDLY BEARISH - Slight negative bias"
        elif sentiment_score > -0.6:
            overall_interpretation = "🔴 BEARISH - Net negative signals"
        else:
            overall_interpretation = "🔴 STRONG BEARISH - Multiple strong negative signals"
        
        # Risk assessment
        confidence = composite_result['confidence']
        if confidence > 0.8:
            risk_assessment = "LOW RISK - High confidence signals"
        elif confidence > 0.6:
            risk_assessment = "MODERATE RISK - Good signal quality"
        else:
            risk_assessment = "HIGH RISK - Low confidence signals"
        
        # Performance grade
        if execution_time < 10:
            performance_grade = "EXCELLENT"
        elif execution_time < 20:
            performance_grade = "GOOD"
        else:
            performance_grade = "ACCEPTABLE"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_version': '5.0_integrated_enterprise',
            'framework_coverage': float(composite_result['framework_coverage']),
            
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
                'optimization_level': 'enterprise_v5.0'
            },
            
            'integration_ready': {
                'composite_sentiment': float(sentiment_score),
                'confidence': float(confidence),
                'framework_version': '5.0',
                'coverage': float(composite_result['framework_coverage']),
                'quality_grade': 'A' if confidence > 0.8 else 'B' if confidence > 0.6 else 'C'
            }
        }
    
    def _save_integrated_results(self, report: Dict):
        """Save integrated analysis results"""
        os.makedirs('outputs', exist_ok=True)
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Main detailed report
            filename = f'outputs/integrated_gold_sentiment_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Integration-ready summary
            integration_summary = {
                'gold_sentiment_composite': report['executive_summary']['overall_sentiment'],
                'confidence': report['executive_summary']['confidence_level'],
                'interpretation': report['executive_summary']['interpretation'],
                'framework_coverage': report['framework_coverage'],
                'quality_grade': report['integration_ready']['quality_grade'],
                'timestamp': report['timestamp'],
                'system_version': '5.0_integrated',
                'components': {
                    'technical_analysis': report['component_analysis'].get('technical_analysis', {}).get('sentiment', 0),
                    'real_interest_rates': report['component_analysis'].get('real_interest_rates', {}).get('sentiment', 0),
                    'central_bank_sentiment': report['component_analysis'].get('central_bank_sentiment', {}).get('sentiment', 0),
                    'professional_positioning': report['component_analysis'].get('professional_positioning', {}).get('sentiment', 0),
                    'currency_debasement': report['component_analysis'].get('currency_debasement', {}).get('sentiment', 0)
                }
            }
            
            with open('outputs/gold_sentiment_integration_ready.json', 'w') as f:
                json.dump(integration_summary, f, indent=2, default=str)
            
            print(f"\n💾 Results saved:")
            print(f"  📄 Detailed: {filename}")
            print(f"  🔗 Integration: gold_sentiment_integration_ready.json")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

# Main execution function
async def run_integrated_analysis():
    """Run the complete integrated gold sentiment analysis"""
    system = IntegratedGoldSentimentSystem()
    results = await system.run_comprehensive_analysis()
    return results

# Performance test
async def performance_test():
    """Test system performance and accuracy"""
    print("🧪 INTEGRATED SYSTEM PERFORMANCE TEST")
    print("=" * 50)
    
    start_time = datetime.now()
    results = await run_integrated_analysis()
    total_time = (datetime.now() - start_time).total_seconds()
    
    if 'error' not in results:
        print(f"\n🏆 INTEGRATION TEST RESULTS")
        print(f"⚡ Total System Time: {total_time:.1f}s")
        print(f"🎯 Composite Sentiment: {results['executive_summary']['overall_sentiment']:.3f}")
        print(f"📊 Confidence: {results['executive_summary']['confidence_level']:.1%}")
        print(f"📈 Framework Coverage: {results['framework_coverage']*100:.0f}%")
        print(f"✅ Components Success: {results['composite_metrics']['components_successful']}/{results['composite_metrics']['components_total']}")
        print(f"🏆 Quality Grade: {results['integration_ready']['quality_grade']}")
        print(f"💯 Performance: {results['performance_metrics']['performance_grade']}")
    else:
        print(f"❌ Integration test failed: {results['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(performance_test())
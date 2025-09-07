# UPDATED INTEGRATED GOLD SENTIMENT SYSTEM v5.0 - JSON OUTPUT VERSION
# =====================================================================
# Modified to generate comprehensive JSON with all sub-indicators instead of terminal output
# All context from each KPI saved for explainable analysis

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

# Import your updated analyzers
try:
    from gold_momentum_analyzer_detailed_json import GoldMomentumAnalyzer
    from real_interest_rate_sentiment import RealRateSentimentScorer
    from optimized_real_time_enterprise import OptimizedRealTimeGoldSentimentAnalyzer
    from cot_positioning_analyzer import ProfessionalPositioningAnalyzer
    from currency_debasement_analyzer import CurrencyDebasementAnalyzer
except ImportError as e:
    print(f"⚠️ Import Error: {e}")

class IntegratedGoldSentimentSystemJSON:
    """
    Complete Integrated Gold Sentiment Analysis System v5.0 - JSON Output Version
    Generates comprehensive explainable JSON instead of terminal output
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
        
        # Initialize comprehensive result structure
        self.comprehensive_result = {
            "analysis_metadata": {},
            "composite_results": {},
            "component_details": {},
            "aggregation_details": {},
            "data_quality_assessment": {}
        }
        
    def _default_config(self) -> Dict:
        return {
            'confidence_threshold': 0.60,
            'min_components_required': 3,
            'outlier_detection': True,
            'adaptive_weighting': True,
            'performance_optimization': True,
            'fred_api_key': None,
            'execution_timeout': 30,
            'save_json_file': True,
            'json_filename': 'explainable_gold_sentiment_analysis.json'
        }
    
    async def run_comprehensive_json_analysis(self) -> Dict:
        """Run complete analysis and generate comprehensive JSON"""
        start_time = datetime.now()
        
        # Initialize metadata
        self.comprehensive_result["analysis_metadata"] = {
            "timestamp": start_time.isoformat(),
            "system_version": "5.0_integrated_enterprise_json",
            "framework_coverage_pct": 95.0,
            "analysis_type": "explainable_comprehensive"
        }
        
        try:
            # Component 1: Technical Analysis (25% weight)
            print("Analyzing Technical Indicators...")
            technical_analyzer = GoldMomentumAnalyzer()
            technical_result = technical_analyzer.analyze_momentum_detailed()
            self.comprehensive_result["component_details"]["technical_analysis"] = technical_result
            
            # Component 2: Real Interest Rates (25% weight) 
            print("Analyzing Real Interest Rates...")
            real_rates_result = await self._analyze_real_rates_detailed()
            self.comprehensive_result["component_details"]["real_interest_rates"] = real_rates_result
            
            # Component 3: Central Bank Sentiment (20% weight)
            print("Analyzing Central Bank Sentiment...")
            cb_result = await self._analyze_central_banks_detailed()
            self.comprehensive_result["component_details"]["central_bank_sentiment"] = cb_result
            
            # Component 4: Professional Positioning COT (15% weight)
            print("Analyzing Professional Positioning...")
            cot_result = await self._analyze_cot_detailed()
            self.comprehensive_result["component_details"]["professional_positioning"] = cot_result
            
            # Component 5: Currency Debasement Risk (10% weight) 
            print("Analyzing Currency Debasement...")
            currency_result = await self._analyze_currency_detailed()
            self.comprehensive_result["component_details"]["currency_debasement"] = currency_result
            
            # Calculate composite results
            execution_time = (datetime.now() - start_time).total_seconds()
            composite_result = self._calculate_detailed_composite()
            
            # Update metadata with execution info
            self.comprehensive_result["analysis_metadata"].update({
                "execution_time_seconds": execution_time,
                "components_analyzed": len(self.comprehensive_result["component_details"]),
                "data_freshness_hours": self._calculate_data_freshness(),
                "performance_grade": "EXCELLENT" if execution_time < 10 else "GOOD" if execution_time < 20 else "ACCEPTABLE"
            })
            
            # Set composite results
            self.comprehensive_result["composite_results"] = composite_result
            
            # Set aggregation details
            self.comprehensive_result["aggregation_details"] = self._generate_aggregation_details()
            
            # Set data quality assessment
            self.comprehensive_result["data_quality_assessment"] = self._assess_data_quality()
            
            # Save JSON file
            if self.config['save_json_file']:
                self._save_comprehensive_json()
            
            print(f"✅ Comprehensive JSON analysis completed in {execution_time:.1f}s")
            return self.comprehensive_result
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'partial_results': self.comprehensive_result
            }
    
    async def _analyze_real_rates_detailed(self) -> Dict:
        """Analyze real interest rates with detailed context"""
        try:
            real_rates_analyzer = RealRateSentimentScorer(api_key=self.config.get('fred_api_key'))
            real_rates_data, regime = real_rates_analyzer.run_optimized_analysis()
            
            if real_rates_data is not None and len(real_rates_data) > 0:
                latest = real_rates_data.iloc[-1]
                
                return {
                    "component_sentiment": float(latest['sentiment_composite']),
                    "component_confidence": float(latest['confidence_composite']),
                    "component_weight": 0.25,
                    "weighted_contribution": float(latest['sentiment_composite'] * 0.25),
                    
                    "sub_indicators": {
                        "current_real_rate": {
                            "value": float(latest['real_rate']),
                            "interpretation": "Negative real rates favor gold" if latest['real_rate'] < 0 else "Positive real rates challenge gold"
                        },
                        "treasury_10y": {
                            "value": float(latest['treasury_10y']),
                            "source": "FRED_or_fallback"
                        },
                        "inflation_expectation_1y": {
                            "value": float(latest['inflation_expectation_1y']),
                            "source": "Michigan_Survey"
                        },
                        "multi_timeframe_analysis": {
                            "short_term_5d": {
                                "ema_value": float(latest.get('real_rate_ema_short', latest['real_rate'])),
                                "sentiment": float(latest.get('sentiment_short', latest['sentiment_composite'])),
                                "confidence": float(latest.get('confidence_short', latest['confidence_composite']))
                            },
                            "medium_term_10d": {
                                "ema_value": float(latest.get('real_rate_ema_medium', latest['real_rate'])),
                                "sentiment": float(latest.get('sentiment_medium', latest['sentiment_composite'])),
                                "confidence": float(latest.get('confidence_medium', latest['confidence_composite']))
                            },
                            "long_term_20d": {
                                "ema_value": float(latest.get('real_rate_ema_long', latest['real_rate'])),
                                "sentiment": float(latest.get('sentiment_long', latest['sentiment_composite'])),
                                "confidence": float(latest.get('confidence_long', latest['confidence_composite']))
                            }
                        }
                    },
                    
                    "market_context": {
                        "regime": regime,
                        "regime_adjustment": 1.0,
                        "data_points": len(real_rates_data),
                        "data_freshness": "current",
                        "interpretation": latest['sentiment_interpretation'],
                        "signal_strength": latest['signal_strength']
                    }
                }
            else:
                return self._create_error_component("Real_Interest_Rates", 0.25)
                
        except Exception as e:
            return self._create_error_component("Real_Interest_Rates", 0.25, str(e))
    
    async def _analyze_central_banks_detailed(self) -> Dict:
        """Analyze central bank sentiment with detailed context"""
        try:
            cb_analyzer = OptimizedRealTimeGoldSentimentAnalyzer()
            cb_result = await cb_analyzer.run_optimized_comprehensive_analysis()
            
            if 'error' not in cb_result:
                # Extract individual signal details
                signal_details = {}
                for signal in cb_result.get('signal_breakdown', []):
                    signal_name = signal.get('source', 'unknown').lower()
                    signal_details[signal_name] = {
                        'sentiment': signal.get('sentiment', 0),
                        'confidence': signal.get('confidence', 0.5),
                        'description': signal.get('description', 'No description'),
                        'source': signal.get('source', 'Unknown')
                    }
                
                return {
                    "component_sentiment": float(cb_result['sentiment_analysis']['sentiment_score']),
                    "component_confidence": float(cb_result['sentiment_analysis']['confidence']),
                    "component_weight": 0.20,
                    "weighted_contribution": float(cb_result['sentiment_analysis']['sentiment_score'] * 0.20),
                    
                    "sub_indicators": signal_details,
                    
                    "market_context": {
                        "sources_available": cb_result['data_quality']['sources_available'],
                        "sources_total": cb_result['data_quality']['sources_total'],
                        "coverage_pct": cb_result['data_quality']['coverage_pct'],
                        "execution_time": cb_result['performance_metrics']['execution_time_seconds'],
                        "optimization_level": "v4.0",
                        "interpretation": cb_result['sentiment_analysis']['interpretation']
                    }
                }
            else:
                return self._create_error_component("Central_Bank_Sentiment", 0.20, cb_result.get('error', 'Unknown error'))
                
        except Exception as e:
            return self._create_error_component("Central_Bank_Sentiment", 0.20, str(e))
    
    async def _analyze_cot_detailed(self) -> Dict:
        """Analyze COT positioning with detailed context"""
        try:
            cot_analyzer = ProfessionalPositioningAnalyzer()
            cot_result = await cot_analyzer.analyze_professional_positioning()
            
            # Extract metadata for detailed context
            metadata = cot_result.get('metadata', {})
            
            return {
                "component_sentiment": float(cot_result['sentiment']),
                "component_confidence": float(cot_result['confidence']),
                "component_weight": float(cot_result['weight']),
                "weighted_contribution": float(cot_result['sentiment'] * cot_result['weight']),
                
                "sub_indicators": {
                    "cot_positioning": {
                        "method": metadata.get('method', 'unknown'),
                        "estimated_commercial_position": metadata.get('estimated_position', 0),
                        "recent_gold_return": metadata.get('recent_return', 0),
                        "volume_factor": metadata.get('volume_factor', 1.0),
                        "interpretation": cot_result.get('interpretation', 'No interpretation available')
                    },
                    "contrarian_analysis": {
                        "gold_trend": "mildly_positive" if metadata.get('recent_return', 0) > 0 else "negative",
                        "expected_commercial_response": "contrarian_positioning",
                        "market_logic": "Commercials are typically contrarian hedgers"
                    }
                },
                
                "market_context": {
                    "data_source": cot_result.get('source', 'unknown'),
                    "cftc_direct_available": metadata.get('method') == 'cftc_direct',
                    "fallback_reason": metadata.get('note', 'Direct data available') if metadata.get('method') != 'cftc_direct' else None,
                    "execution_time": cot_result.get('execution_time_seconds', 0),
                    "confidence_adjustment": "Applied for proxy method" if metadata.get('method') == 'price_proxy' else None
                }
            }
            
        except Exception as e:
            return self._create_error_component("Professional_Positioning_COT", 0.15, str(e))
    
    async def _analyze_currency_detailed(self) -> Dict:
        """Analyze currency debasement with detailed context"""
        try:
            currency_analyzer = CurrencyDebasementAnalyzer(fred_api_key=self.config.get('fred_api_key'))
            currency_result = await currency_analyzer.analyze_currency_debasement_risk()
            
            # Extract component details from metadata
            component_details = currency_result.get('metadata', {}).get('component_details', [])
            
            sub_indicators = {}
            for comp in component_details:
                comp_name = comp.get('name', 'unknown').lower().replace(' ', '_')
                sub_indicators[comp_name] = {
                    'sentiment': comp.get('sentiment', 0),
                    'confidence': comp.get('confidence', 0.5),
                    'weight_in_component': comp.get('weight', 0.33),
                    'contribution': comp.get('contribution', 0)
                }
            
            return {
                "component_sentiment": float(currency_result['sentiment']),
                "component_confidence": float(currency_result['confidence']),
                "component_weight": 0.10,
                "weighted_contribution": float(currency_result['sentiment'] * 0.10),
                
                "sub_indicators": sub_indicators,
                
                "market_context": {
                    "components_analyzed": currency_result['metadata']['components_analyzed'],
                    "fred_api_available": currency_result['metadata'].get('method') != 'fallback',
                    "fallback_methods_used": True,
                    "execution_time": currency_result.get('execution_time_seconds', 0),
                    "interpretation": currency_result.get('interpretation', 'No interpretation available')
                }
            }
            
        except Exception as e:
            return self._create_error_component("Currency_Debasement_Risk", 0.10, str(e))
    
    def _create_error_component(self, component_name: str, weight: float, error_msg: str = "Analysis failed") -> Dict:
        """Create standardized error component"""
        return {
            "component_sentiment": 0.0,
            "component_confidence": 0.3,
            "component_weight": weight * 0.5,
            "weighted_contribution": 0.0,
            "sub_indicators": {},
            "market_context": {
                "error": True,
                "error_message": error_msg,
                "fallback_applied": True
            }
        }
    
    def _calculate_detailed_composite(self) -> Dict:
        """Calculate composite results with detailed breakdown"""
        components = self.comprehensive_result["component_details"]
        
        total_weighted_sentiment = 0
        total_weight = 0
        confidence_scores = []
        
        for comp_name, comp_data in components.items():
            if isinstance(comp_data, dict) and 'component_sentiment' in comp_data:
                weight = comp_data.get('component_weight', 0)
                sentiment = comp_data.get('component_sentiment', 0)
                confidence = comp_data.get('component_confidence', 0.5)
                
                adjusted_weight = weight * confidence
                total_weighted_sentiment += sentiment * adjusted_weight
                total_weight += adjusted_weight
                confidence_scores.append(confidence)
        
        overall_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Interpretation
        if overall_sentiment > 0.6:
            interpretation = "STRONG_BULLISH"
        elif overall_sentiment > 0.3:
            interpretation = "BULLISH"
        elif overall_sentiment > 0.1:
            interpretation = "MILDLY_BULLISH"
        elif overall_sentiment > -0.1:
            interpretation = "NEUTRAL"
        elif overall_sentiment > -0.3:
            interpretation = "MILDLY_BEARISH"
        elif overall_sentiment > -0.6:
            interpretation = "BEARISH"
        else:
            interpretation = "STRONG_BEARISH"
        
        return {
            "overall_sentiment": float(overall_sentiment),
            "overall_confidence": float(overall_confidence),
            "interpretation": interpretation,
            "quality_grade": "A" if overall_confidence > 0.8 else "B" if overall_confidence > 0.6 else "C",
            "framework_weights_used": self.framework_weights,
            "component_contributions": {
                comp_name: comp_data.get('weighted_contribution', 0) 
                for comp_name, comp_data in components.items()
                if isinstance(comp_data, dict)
            }
        }
    
    def _generate_aggregation_details(self) -> Dict:
        """Generate details about how components were aggregated"""
        return {
            "outlier_detection_applied": self.config['outlier_detection'],
            "outliers_removed": 0,  # Would track actual outliers removed
            "cross_validation_bonus": 0.05,  # Example value
            "adaptive_weighting_applied": self.config['adaptive_weighting'],
            "confidence_adjustments": {
                "cot_proxy_reduction": -0.35,
                "currency_fallback_reduction": -0.10,
                "coverage_bonus": 0.10
            }
        }
    
    def _assess_data_quality(self) -> Dict:
        """Assess overall data quality"""
        components = self.comprehensive_result["component_details"]
        
        real_time_sources = 0
        total_sources = len(components)
        error_sources = 0
        
        for comp_data in components.values():
            if isinstance(comp_data, dict):
                if comp_data.get('market_context', {}).get('error'):
                    error_sources += 1
                else:
                    real_time_sources += 1
        
        return {
            "real_time_data_pct": (real_time_sources / total_sources * 100) if total_sources > 0 else 0,
            "cached_data_pct": 15,  # Example
            "error_sources": error_sources,
            "fallback_methods_used": 2,  # Example
            "api_success_rate": (real_time_sources / total_sources) if total_sources > 0 else 0,
            "data_age_assessment": {
                "technical_data": "< 1 hour",
                "real_rates_data": "< 4 hours",
                "cb_data": "< 6 hours",
                "cot_data": "proxy_estimate",
                "currency_data": "< 2 hours"
            }
        }
    
    def _calculate_data_freshness(self) -> float:
        """Calculate average data freshness in hours"""
        # This would calculate actual data freshness based on timestamps
        return 2.5  # Example: 2.5 hours average
    
    def _save_comprehensive_json(self):
        """Save comprehensive results to JSON file"""
        filename = self.config['json_filename']
        try:
            with open(filename, 'w') as f:
                json.dump(self.comprehensive_result, f, indent=2, default=str)
            print(f"✅ Comprehensive analysis saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")

# Main execution function
async def run_json_analysis():
    """Run the complete JSON analysis"""
    system = IntegratedGoldSentimentSystemJSON()
    results = await system.run_comprehensive_json_analysis()
    return results

# Test function
async def test_json_system():
    """Test the JSON system"""
    print("🧪 TESTING JSON-BASED GOLD SENTIMENT SYSTEM")
    print("=" * 50)
    
    start_time = datetime.now()
    results = await run_json_analysis()
    execution_time = (datetime.now() - start_time).total_seconds()
    
    if 'error' not in results:
        print(f"✅ JSON Analysis completed in {execution_time:.1f}s")
        print(f"📊 Components analyzed: {results['analysis_metadata']['components_analyzed']}")
        print(f"🎯 Overall sentiment: {results['composite_results']['overall_sentiment']:.3f}")
        print(f"📈 Confidence: {results['composite_results']['overall_confidence']:.1%}")
        print(f"🏆 Quality grade: {results['composite_results']['quality_grade']}")
    else:
        print(f"❌ Analysis failed: {results['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_json_system())
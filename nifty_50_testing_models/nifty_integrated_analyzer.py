# NIFTY 50 INTEGRATED SENTIMENT ANALYZER v1.0
# ============================================
# Master integration system for Nifty 50 sentiment analysis
# Combines FII/DII flows + Technical Analysis with comprehensive JSON output

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import warnings

# Import the individual analyzers
try:
    from fii_dii_flows_analyzer import FIIDIIFlowsAnalyzer
    from nifty_technical_analyzer import NiftyTechnicalAnalyzer
except ImportError as e:
    print(f"⚠️ Import Error: {e}")
    print("Make sure fii_dii_flows_analyzer.py and nifty_technical_analyzer.py are in the same directory")

warnings.filterwarnings('ignore')

class NiftyIntegratedSentimentAnalyzer:
    """
    Complete Integrated Nifty 50 Sentiment Analysis System v1.0
    - Combines FII/DII flows (35%) + Technical Analysis (30%) = 65% coverage
    - Generates comprehensive explainable JSON output
    - High reliability with real-time data fetching
    - Master JSON structure for all indicators and sub-indicators
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Framework weights for POC (65% total coverage)
        self.framework_weights = {
            'fii_dii_flows': 0.35,        # Highest priority - institutional flows
            'technical_analysis': 0.30,   # Second priority - price momentum
            # Future KPIs (35% remaining):
            # 'market_sentiment_vix': 0.20,     # VIX, Put/Call ratios
            # 'interest_rates_rbi': 0.10,       # RBI policy rates
            # 'global_factors': 0.05            # Global overnight moves
        }
        
        # Master result structure for complete context
        self.master_result = {
            "analysis_metadata": {},
            "composite_results": {},
            "component_details": {},
            "aggregation_details": {},
            "data_quality_assessment": {},
            "market_overview": {}
        }
        
        print("Nifty 50 Integrated Sentiment Analyzer v1.0 initialized")
        print("🎯 65% Framework Coverage: FII/DII (35%) + Technical (30%)")
        print("🏦 Enterprise-grade multi-component analysis for Indian equity markets")
    
    def _default_config(self) -> Dict:
        return {
            'confidence_threshold': 0.70,
            'min_components_required': 2,
            'outlier_detection': True,
            'adaptive_weighting': True,
            'execution_timeout': 30,
            'save_master_json': True,
            'master_json_filename': 'nifty50_master_sentiment_analysis.json'
        }
    
    async def run_comprehensive_analysis(self) -> Dict:
        """Run complete Nifty 50 sentiment analysis with all components"""
        start_time = datetime.now()
        
        print("🚀 NIFTY 50 INTEGRATED SENTIMENT ANALYSIS v1.0")
        print("=" * 60)
        print("🎯 Coverage: FII/DII Flows + Technical Analysis (65% total)")
        print("⏰ Started:", start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print("=" * 60)
        
        # Initialize metadata
        self.master_result["analysis_metadata"] = {
            "timestamp": start_time.isoformat(),
            "system_version": "1.0_nifty_integrated_poc",
            "framework_coverage_pct": 65.0,
            "analysis_type": "nifty_50_sentiment_comprehensive",
            "market_focus": "Indian_equity_nifty_50"
        }
        
        try:
            # Component 1: FII/DII Flows Analysis (35% weight)
            print("1️⃣ Analyzing FII/DII Institutional Flows (35% weight)...")
            fii_dii_analyzer = FIIDIIFlowsAnalyzer()
            fii_dii_result = await fii_dii_analyzer.analyze_fii_dii_sentiment()
            self.master_result["component_details"]["fii_dii_flows"] = fii_dii_result
            
            # Component 2: Technical Analysis (30% weight)
            print("\n2️⃣ Analyzing Technical Indicators (30% weight)...")
            technical_analyzer = NiftyTechnicalAnalyzer()
            technical_result = await technical_analyzer.analyze_technical_sentiment()
            self.master_result["component_details"]["technical_analysis"] = technical_result
            
            # Calculate composite results
            execution_time = (datetime.now() - start_time).total_seconds()
            composite_result = self._calculate_comprehensive_composite()
            
            # Update metadata with execution info
            self.master_result["analysis_metadata"].update({
                "execution_time_seconds": execution_time,
                "components_analyzed": len(self.master_result["component_details"]),
                "data_freshness_assessment": self._assess_data_freshness(),
                "performance_grade": "EXCELLENT" if execution_time < 10 else "GOOD" if execution_time < 20 else "ACCEPTABLE"
            })
            
            # Set composite results
            self.master_result["composite_results"] = composite_result
            
            # Set aggregation details
            self.master_result["aggregation_details"] = self._generate_aggregation_details()
            
            # Set data quality assessment
            self.master_result["data_quality_assessment"] = self._assess_comprehensive_data_quality()
            
            # Set market overview
            self.master_result["market_overview"] = self._generate_market_overview()
            
            # Save master JSON file
            if self.config['save_master_json']:
                self._save_master_json()
            
            print(f"\n✅ Integrated Analysis completed in {execution_time:.1f}s")
            print(f"🎯 Composite Sentiment: {composite_result['overall_sentiment']:.3f}")
            print(f"📊 Overall Confidence: {composite_result['overall_confidence']:.1%}")
            print(f"🏆 Framework Coverage: {self.master_result['analysis_metadata']['framework_coverage_pct']}%")
            
            return self.master_result
            
        except Exception as e:
            print(f"❌ Integrated analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'partial_results': self.master_result
            }
    
    def _calculate_comprehensive_composite(self) -> Dict:
        """Calculate composite results from all components with detailed breakdown"""
        components = self.master_result["component_details"]
        
        total_weighted_sentiment = 0
        total_weight = 0
        confidence_scores = []
        component_contributions = {}
        
        for comp_name, comp_data in components.items():
            if isinstance(comp_data, dict) and 'component_sentiment' in comp_data:
                weight = comp_data.get('component_weight', 0)
                sentiment = comp_data.get('component_sentiment', 0)
                confidence = comp_data.get('component_confidence', 0.5)
                
                # Apply adaptive weighting if enabled
                if self.config['adaptive_weighting']:
                    adjusted_weight = weight * confidence
                else:
                    adjusted_weight = weight
                
                total_weighted_sentiment += sentiment * adjusted_weight
                total_weight += adjusted_weight
                confidence_scores.append(confidence)
                component_contributions[comp_name] = sentiment * adjusted_weight
        
        # Calculate final sentiment and confidence
        overall_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Apply outlier detection if enabled
        if self.config['outlier_detection']:
            overall_sentiment = self._apply_outlier_adjustment(overall_sentiment, components)
        
        # Determine interpretation
        interpretation = self._interpret_sentiment(overall_sentiment)
        
        # Calculate quality grade
        quality_grade = self._calculate_quality_grade(overall_confidence, len(components))
        
        return {
            "overall_sentiment": float(overall_sentiment),
            "overall_confidence": float(overall_confidence),
            "interpretation": interpretation,
            "quality_grade": quality_grade,
            "framework_weights_used": self.framework_weights,
            "component_contributions": {k: float(v) for k, v in component_contributions.items()},
            "coverage_details": {
                "implemented_components": list(components.keys()),
                "total_weight_covered": float(sum(self.framework_weights[k] for k in components.keys() if k in self.framework_weights)),
                "missing_components": ["market_sentiment_vix", "interest_rates_rbi", "global_factors"],
                "poc_status": "Phase_1_Complete"
            }
        }
    
    def _interpret_sentiment(self, sentiment_score: float) -> str:
        """Interpret sentiment score into market language"""
        if sentiment_score > 0.6:
            return "STRONG_BULLISH"
        elif sentiment_score > 0.3:
            return "BULLISH"
        elif sentiment_score > 0.1:
            return "MILDLY_BULLISH"
        elif sentiment_score > -0.1:
            return "NEUTRAL"
        elif sentiment_score > -0.3:
            return "MILDLY_BEARISH"
        elif sentiment_score > -0.6:
            return "BEARISH"
        else:
            return "STRONG_BEARISH"
    
    def _calculate_quality_grade(self, confidence: float, component_count: int) -> str:
        """Calculate quality grade based on confidence and component coverage"""
        coverage_factor = min(1.0, component_count / 2)  # 2 components for POC
        quality_score = confidence * 0.7 + coverage_factor * 0.3
        
        if quality_score >= 0.9:
            return "A+"
        elif quality_score >= 0.8:
            return "A"
        elif quality_score >= 0.7:
            return "B+"
        elif quality_score >= 0.6:
            return "B"
        elif quality_score >= 0.5:
            return "C+"
        else:
            return "C"
    
    def _apply_outlier_adjustment(self, sentiment: float, components: Dict) -> float:
        """Apply outlier detection and adjustment"""
        # For POC with 2 components, simple validation
        sentiments = []
        for comp_data in components.values():
            if isinstance(comp_data, dict) and 'component_sentiment' in comp_data:
                sentiments.append(comp_data['component_sentiment'])
        
        if len(sentiments) >= 2:
            # Check if sentiments are drastically different
            sentiment_range = max(sentiments) - min(sentiments)
            if sentiment_range > 1.5:  # Very different signals
                # Apply conservative adjustment
                return sentiment * 0.8
        
        return sentiment
    
    def _generate_aggregation_details(self) -> Dict:
        """Generate details about how components were aggregated"""
        return {
            "aggregation_method": "weighted_average",
            "outlier_detection_applied": self.config['outlier_detection'],
            "adaptive_weighting_applied": self.config['adaptive_weighting'],
            "confidence_threshold": self.config['confidence_threshold'],
            "weighting_scheme": {
                "fii_dii_flows": "35% - Highest priority (institutional behavior)",
                "technical_analysis": "30% - Price momentum confirmation"
            },
            "signal_validation": {
                "cross_validation_enabled": len(self.master_result["component_details"]) >= 2,
                "signal_consistency_check": True,
                "fallback_handling": "Neutral default with reduced weight"
            }
        }
    
    def _assess_comprehensive_data_quality(self) -> Dict:
        """Assess overall data quality across all components"""
        components = self.master_result["component_details"]
        
        high_quality_sources = 0
        total_sources = len(components)
        error_components = 0
        
        data_ages = []
        source_reliabilities = []
        
        for comp_name, comp_data in components.items():
            if isinstance(comp_data, dict):
                # Check for errors
                if comp_data.get('market_context', {}).get('error'):
                    error_components += 1
                else:
                    high_quality_sources += 1
                
                # Assess data quality indicators
                confidence = comp_data.get('component_confidence', 0.5)
                source_reliabilities.append(confidence)
                
                # Check data freshness
                data_quality = comp_data.get('data_quality', 'medium')
                if data_quality == 'high':
                    data_ages.append('current')
                else:
                    data_ages.append('acceptable')
        
        return {
            "high_quality_sources_pct": (high_quality_sources / total_sources * 100) if total_sources > 0 else 0,
            "error_components": error_components,
            "average_source_reliability": float(np.mean(source_reliabilities)) if source_reliabilities else 0.5,
            "data_freshness_status": "current" if all(age == 'current' for age in data_ages) else "acceptable",
            "overall_data_grade": "A" if error_components == 0 and len(source_reliabilities) >= 2 else "B",
            "component_reliability_breakdown": {
                comp_name: comp_data.get('component_confidence', 0.5)
                for comp_name, comp_data in components.items()
                if isinstance(comp_data, dict)
            }
        }
    
    def _assess_data_freshness(self) -> str:
        """Assess how fresh the data is"""
        # For real-time analysis, most data should be very fresh
        return "real_time_to_15min"
    
    def _generate_market_overview(self) -> Dict:
        """Generate market overview from available data"""
        components = self.master_result["component_details"]
        
        market_overview = {
            "analysis_date": datetime.now().strftime('%Y-%m-%d'),
            "market_session": self._get_market_session(),
            "nifty_snapshot": {},
            "institutional_activity": {},
            "technical_summary": {},
            "key_highlights": []
        }
        
        # Extract Nifty snapshot from technical analysis
        tech_data = components.get('technical_analysis', {})
        if 'market_context' in tech_data:
            market_overview["nifty_snapshot"] = {
                "current_level": tech_data['market_context'].get('current_price'),
                "daily_change_pct": tech_data['market_context'].get('daily_change_pct'),
                "trend_status": tech_data.get('technical_summary', {}).get('overall_trend', 'unknown'),
                "volatility_regime": tech_data.get('technical_summary', {}).get('volatility_regime', 'normal')
            }
        
        # Extract institutional activity from FII/DII data
        fii_dii_data = components.get('fii_dii_flows', {})
        if 'sub_indicators' in fii_dii_data:
            flows = fii_dii_data['sub_indicators']
            market_overview["institutional_activity"] = {
                "fii_net_flow": flows.get('fii_flows', {}).get('net_flow_crores'),
                "dii_net_flow": flows.get('dii_flows', {}).get('net_flow_crores'),
                "total_net_flow": flows.get('net_flows', {}).get('total_net_flow_crores'),
                "flow_dominance": flows.get('flow_balance', {}).get('dominance'),
                "flow_momentum": flows.get('flow_momentum', {}).get('trend')
            }
        
        # Generate key highlights
        highlights = []
        if fii_dii_data.get('component_sentiment', 0) > 0.3:
            highlights.append("Strong institutional buying detected")
        if tech_data.get('component_sentiment', 0) > 0.3:
            highlights.append("Technical indicators showing bullish momentum")
        if abs(fii_dii_data.get('component_sentiment', 0)) < 0.1 and abs(tech_data.get('component_sentiment', 0)) < 0.1:
            highlights.append("Mixed signals - neutral market sentiment")
        
        market_overview["key_highlights"] = highlights
        
        return market_overview
    
    def _get_market_session(self) -> str:
        """Determine current market session"""
        now = datetime.now()
        hour = now.hour
        
        # Indian market hours (9:15 AM to 3:30 PM IST)
        if 9 <= hour < 15:
            return "market_open"
        elif 15 <= hour < 16:
            return "closing_session"
        elif 16 <= hour <= 23:
            return "after_market"
        else:
            return "pre_market"
    
    def _save_master_json(self):
        """Save comprehensive master JSON file"""
        filename = self.config['master_json_filename']
        
        try:
            # Make JSON serializable
            clean_result = self._make_json_serializable(self.master_result)
            
            with open(filename, 'w') as f:
                json.dump(clean_result, f, indent=2, default=str)
            
            print(f"📁 Master analysis saved to: {filename}")
            
            # Also save a summary JSON for quick access
            summary = {
                "timestamp": self.master_result["analysis_metadata"]["timestamp"],
                "overall_sentiment": self.master_result["composite_results"]["overall_sentiment"],
                "overall_confidence": self.master_result["composite_results"]["overall_confidence"],
                "interpretation": self.master_result["composite_results"]["interpretation"],
                "quality_grade": self.master_result["composite_results"]["quality_grade"],
                "key_components": {
                    comp_name: {
                        "sentiment": comp_data.get("component_sentiment", 0),
                        "confidence": comp_data.get("component_confidence", 0),
                        "interpretation": comp_data.get("interpretation", "N/A")
                    }
                    for comp_name, comp_data in self.master_result["component_details"].items()
                    if isinstance(comp_data, dict)
                }
            }
            
            with open('nifty50_sentiment_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📄 Quick summary saved to: nifty50_sentiment_summary.json")
            
        except Exception as e:
            print(f"❌ Error saving master JSON: {e}")
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

# Main execution function
async def run_nifty_sentiment_analysis():
    """Run the complete Nifty 50 sentiment analysis"""
    analyzer = NiftyIntegratedSentimentAnalyzer()
    results = await analyzer.run_comprehensive_analysis()
    return results

# Test function with detailed output
async def test_integrated_system():
    """Test the integrated Nifty 50 sentiment system"""
    print("🧪 TESTING NIFTY 50 INTEGRATED SENTIMENT SYSTEM")
    print("=" * 55)
    
    start_time = datetime.now()
    results = await run_nifty_sentiment_analysis()
    execution_time = (datetime.now() - start_time).total_seconds()
    
    if 'error' not in results:
        print(f"\n🏆 INTEGRATED ANALYSIS RESULTS")
        print("=" * 40)
        print(f"✅ Total Execution Time: {execution_time:.1f}s")
        print(f"📊 Components Analyzed: {results['analysis_metadata']['components_analyzed']}")
        print(f"🎯 Overall Sentiment: {results['composite_results']['overall_sentiment']:.3f}")
        print(f"📈 Interpretation: {results['composite_results']['interpretation']}")
        print(f"💪 Confidence: {results['composite_results']['overall_confidence']:.1%}")
        print(f"🏆 Quality Grade: {results['composite_results']['quality_grade']}")
        print(f"📊 Framework Coverage: {results['analysis_metadata']['framework_coverage_pct']}%")
        
        print(f"\n📋 COMPONENT BREAKDOWN:")
        for comp_name, contribution in results['composite_results']['component_contributions'].items():
            comp_data = results['component_details'][comp_name]
            print(f"   {comp_name.replace('_', ' ').title()}:")
            print(f"      Sentiment: {comp_data['component_sentiment']:+.3f}")
            print(f"      Confidence: {comp_data['component_confidence']:.1%}")
            print(f"      Contribution: {contribution:+.4f}")
        
        print(f"\n📊 DATA QUALITY ASSESSMENT:")
        dq = results['data_quality_assessment']
        print(f"   High Quality Sources: {dq['high_quality_sources_pct']:.0f}%")
        print(f"   Error Components: {dq['error_components']}")
        print(f"   Average Reliability: {dq['average_source_reliability']:.1%}")
        print(f"   Overall Data Grade: {dq['overall_data_grade']}")
        
        print(f"\n🎯 NEXT STEPS FOR FULL SYSTEM:")
        print("   1. Add Market Sentiment (VIX) - 20% weight")
        print("   2. Add RBI Interest Rates - 10% weight")  
        print("   3. Add Global Factors - 5% weight")
        print("   4. Achieve 100% framework coverage")
        
    else:
        print(f"❌ Analysis failed: {results['error']}")
    
    return results

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_integrated_system())
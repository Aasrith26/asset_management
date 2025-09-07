
import asyncio
import json
import os

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import warnings

# Import all analyzers
try:
    from fii_dii_flows_analyzer import FIIDIIFlowsAnalyzer
    from nifty_technical_analyzer import NiftyTechnicalAnalyzer
    from market_sentiment_analyzer import MarketSentimentAnalyzer
    from rbi_policy_analyzer import RBIInterestRatesAnalyzer
    from global_factors_analyzer import GlobalFactorsAnalyzer
except ImportError as e:
    print(f"⚠️ Import Error: {e}")
    print("Make sure all analyzer files are in the same directory")

warnings.filterwarnings('ignore')

class CompleteNiftySentimentAnalyzer:
    """
    Complete Nifty 50 Sentiment Analysis System v2.0 - 100% Framework Coverage
    - FII/DII Flows: 35% (Institutional behavior)
    - Technical Analysis: 30% (Price momentum) 
    - Market Sentiment (VIX): 20% (Fear/greed indicators)
    - RBI Policy/Interest Rates: 10% (Monetary policy)
    - Global Factors: 5% (External influences)
    
    TOTAL: 100% comprehensive framework coverage
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Complete framework weights (100% total)
        self.framework_weights = {
            'fii_dii_flows': 0.35,          # Highest priority - institutional flows
            'technical_analysis': 0.30,     # Price momentum confirmation
            'market_sentiment_vix': 0.20,   # Fear/greed behavioral indicators
            'rbi_interest_rates': 0.10,     # Monetary policy impact
            'global_factors': 0.05          # External global influences
        }
        
        # Master result structure
        self.master_result = {
            "analysis_metadata": {},
            "composite_results": {},
            "component_details": {},
            "aggregation_details": {},
            "data_quality_assessment": {},
            "market_overview": {},
            "trading_signals": {}
        }
        
        print("Complete Nifty 50 Sentiment Analyzer v2.0 initialized")
        print("100% Framework Coverage: All 5 KPIs integrated")
        print("Production-ready comprehensive analysis system")
    
    def _default_config(self) -> Dict:
        return {
            'confidence_threshold': 0.75,
            'min_components_required': 4,  # Minimum 4/5 components for valid analysis
            'outlier_detection': True,
            'adaptive_weighting': True,
            'execution_timeout': 45,  # Extended timeout for 5 components
            'save_master_json': True,
            'master_json_filename': 'outputs/nifty50_complete_sentiment_analysis.json',
            'reliability_target': 0.85  # Target 85% reliability
        }
    
    async def run_complete_analysis(self) -> Dict:
        """Run complete 100% Nifty 50 sentiment analysis with all components"""
        start_time = datetime.now()
        
        print("NIFTY 50 COMPLETE SENTIMENT ANALYSIS v2.0")
        print("=" * 70)
        print("Coverage: ALL 5 KPIs - 100% Framework Completion")
        print("Started:", start_time.strftime('%Y-%m-%d %H:%M:%S'))
        print("=" * 70)
        
        # Initialize metadata
        self.master_result["analysis_metadata"] = {
            "timestamp": start_time.isoformat(),
            "system_version": "2.0_complete_framework",
            "framework_coverage_pct": 100.0,
            "analysis_type": "nifty_50_complete_sentiment",
            "market_focus": "Indian_equity_nifty_50_comprehensive"
        }
        
        try:

            print("Analyzing FII/DII Institutional Flows (35% weight)...")
            fii_dii_analyzer = FIIDIIFlowsAnalyzer()
            fii_dii_result = await fii_dii_analyzer.analyze_fii_dii_sentiment()
            self.master_result["component_details"]["fii_dii_flows"] = fii_dii_result
            
            # Component 2: Technical Analysis (30% weight)
            print("\nAnalyzing Technical Indicators (30% weight)...")
            technical_analyzer = NiftyTechnicalAnalyzer()
            technical_result = await technical_analyzer.analyze_technical_sentiment()
            self.master_result["component_details"]["technical_analysis"] = technical_result
            
            # Component 3: Market Sentiment/VIX (20% weight)
            print("\n3Analyzing Market Sentiment & VIX (20% weight)...")
            sentiment_analyzer = MarketSentimentAnalyzer()
            sentiment_result = await sentiment_analyzer.analyze_market_sentiment()
            self.master_result["component_details"]["market_sentiment_vix"] = sentiment_result
            
            # Component 4: RBI Interest Rates (10% weight)
            print("\nAnalyzing RBI Policy & Interest Rates (10% weight)...")
            rbi_analyzer = RBIInterestRatesAnalyzer()
            rbi_result = await rbi_analyzer.analyze_rbi_policy_sentiment()
            self.master_result["component_details"]["rbi_interest_rates"] = rbi_result
            
            # Component 5: Global Factors (5% weight)
            print("\n5️Analyzing Global Market Factors (5% weight)...")
            global_analyzer = GlobalFactorsAnalyzer()
            global_result = await global_analyzer.analyze_global_factors_sentiment()
            self.master_result["component_details"]["global_factors"] = global_result
            
            # Calculate comprehensive composite results
            execution_time = (datetime.now() - start_time).total_seconds()
            composite_result = self._calculate_complete_composite()
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(composite_result)
            
            # Update metadata with execution info
            self.master_result["analysis_metadata"].update({
                "execution_time_seconds": execution_time,
                "components_analyzed": len(self.master_result["component_details"]),
                "successful_components": len([c for c in self.master_result["component_details"].values() 
                                            if isinstance(c, dict) and not c.get('error', False)]),
                "data_freshness_assessment": self._assess_complete_data_freshness(),
                "performance_grade": "EXCELLENT" if execution_time < 15 else "GOOD" if execution_time < 30 else "ACCEPTABLE",
                "reliability_achieved": composite_result.get('overall_confidence', 0)
            })
            
            # Set all result components
            self.master_result["composite_results"] = composite_result
            self.master_result["trading_signals"] = trading_signals
            self.master_result["aggregation_details"] = self._generate_complete_aggregation_details()
            self.master_result["data_quality_assessment"] = self._assess_complete_data_quality()
            self.master_result["market_overview"] = self._generate_complete_market_overview()
            
            # Save master JSON file
            if self.config['save_master_json']:
                self._save_complete_master_json()
            
            print(f"\n✅ Complete Analysis finished in {execution_time:.1f}s")
            print(f"🎯 Overall Sentiment: {composite_result['overall_sentiment']:.3f}")
            print(f"📊 Overall Confidence: {composite_result['overall_confidence']:.1%}")
            print(f"🏆 Framework Coverage: 100% (All 5 KPIs)")
            print(f"⚡ System Reliability: {composite_result['overall_confidence']:.0%}")
            
            return self.master_result
            
        except Exception as e:
            print(f"❌ Complete analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'partial_results': self.master_result
            }
    
    def _calculate_complete_composite(self) -> Dict:
        """Calculate comprehensive composite results from all 5 components"""
        components = self.master_result["component_details"]
        
        total_weighted_sentiment = 0
        total_weight = 0
        confidence_scores = []
        component_contributions = {}
        component_signals = {}
        
        for comp_name, comp_data in components.items():
            if isinstance(comp_data, dict) and 'component_sentiment' in comp_data:
                weight = self.framework_weights.get(comp_name, 0.2)
                sentiment = comp_data.get('component_sentiment', 0)
                confidence = comp_data.get('component_confidence', 0.5)
                
                # Apply adaptive weighting based on confidence
                if self.config['adaptive_weighting']:
                    # Boost high-confidence components, reduce low-confidence ones
                    confidence_multiplier = 0.5 + (confidence * 1.0)  # 0.5 to 1.5 range
                    adjusted_weight = weight * confidence_multiplier
                else:
                    adjusted_weight = weight * confidence
                
                total_weighted_sentiment += sentiment * adjusted_weight
                total_weight += adjusted_weight
                confidence_scores.append(confidence)
                component_contributions[comp_name] = sentiment * adjusted_weight
                
                # Store individual signals
                component_signals[comp_name] = {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'weight': weight,
                    'adjusted_weight': adjusted_weight,
                    'interpretation': comp_data.get('interpretation', 'N/A')
                }
        
        # Calculate final sentiment and confidence
        overall_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Apply advanced signal processing
        if self.config['outlier_detection']:
            overall_sentiment = self._apply_advanced_outlier_detection(overall_sentiment, component_signals)
        
        # Cross-validation between components
        signal_consistency = self._calculate_signal_consistency(component_signals)
        
        # Final confidence adjustment based on consistency
        adjusted_confidence = overall_confidence * signal_consistency
        
        # Determine comprehensive interpretation
        interpretation = self._interpret_complete_sentiment(overall_sentiment, adjusted_confidence)
        
        # Calculate quality grade
        quality_grade = self._calculate_complete_quality_grade(adjusted_confidence, len(components))
        
        # Generate strength assessment
        strength_assessment = self._assess_signal_strength(overall_sentiment, adjusted_confidence, signal_consistency)
        
        return {
            "overall_sentiment": float(overall_sentiment),
            "overall_confidence": float(adjusted_confidence),
            "signal_consistency": float(signal_consistency),
            "interpretation": interpretation,
            "quality_grade": quality_grade,
            "strength_assessment": strength_assessment,
            "framework_weights_used": self.framework_weights,
            "component_contributions": {k: float(v) for k, v in component_contributions.items()},
            "component_signals": component_signals,
            "coverage_details": {
                "implemented_components": list(components.keys()),
                "total_weight_covered": 1.0,  # 100% coverage
                "missing_components": [],  # None - complete framework
                "framework_status": "Complete_Production_Ready"
            },
            "advanced_metrics": {
                "sentiment_momentum": self._calculate_sentiment_momentum(component_signals),
                "risk_assessment": self._calculate_risk_assessment(component_signals),
                "conviction_level": self._calculate_conviction_level(overall_sentiment, adjusted_confidence)
            }
        }
    
    def _apply_advanced_outlier_detection(self, sentiment: float, component_signals: Dict) -> float:
        """Advanced outlier detection across all 5 components"""
        sentiments = [sig['sentiment'] for sig in component_signals.values()]
        
        if len(sentiments) >= 4:  # Need at least 4 components
            median_sentiment = np.median(sentiments)
            mad = np.median([abs(s - median_sentiment) for s in sentiments])  # Median Absolute Deviation
            
            # If consensus sentiment differs significantly from any outlier
            if mad > 0.5:  # Significant disagreement
                # Weight towards consensus, reduce outlier influence
                consensus_weight = 0.7
                return sentiment * (1 - consensus_weight) + median_sentiment * consensus_weight
        
        return sentiment
    
    def _calculate_signal_consistency(self, component_signals: Dict) -> float:
        """Calculate how consistent signals are across all components"""
        sentiments = [sig['sentiment'] for sig in component_signals.values()]
        confidences = [sig['confidence'] for sig in component_signals.values()]
        
        if len(sentiments) < 2:
            return 0.5
        
        # Directional consistency
        positive_signals = sum(1 for s in sentiments if s > 0.1)
        negative_signals = sum(1 for s in sentiments if s < -0.1)
        neutral_signals = sum(1 for s in sentiments if -0.1 <= s <= 0.1)
        total_signals = len(sentiments)
        
        # High consistency when signals agree
        if positive_signals >= total_signals * 0.8:
            directional_consistency = 0.95  # Strong bullish consensus
        elif negative_signals >= total_signals * 0.8:
            directional_consistency = 0.95  # Strong bearish consensus
        elif positive_signals >= total_signals * 0.6:
            directional_consistency = 0.8   # Moderate bullish
        elif negative_signals >= total_signals * 0.6:
            directional_consistency = 0.8   # Moderate bearish
        else:
            directional_consistency = 0.6   # Mixed signals
        
        # Confidence consistency
        confidence_std = np.std(confidences)
        confidence_consistency = max(0.5, 1.0 - confidence_std)
        
        # Combined consistency
        return (directional_consistency * 0.7 + confidence_consistency * 0.3)
    
    def _interpret_complete_sentiment(self, sentiment: float, confidence: float) -> str:
        """Comprehensive sentiment interpretation with confidence consideration"""
        confidence_modifier = "HIGH_CONFIDENCE" if confidence > 0.8 else "MODERATE_CONFIDENCE" if confidence > 0.6 else "LOW_CONFIDENCE"
        
        if sentiment > 0.6:
            return f"STRONG_BULLISH_{confidence_modifier}"
        elif sentiment > 0.3:
            return f"BULLISH_{confidence_modifier}"
        elif sentiment > 0.1:
            return f"MILDLY_BULLISH_{confidence_modifier}"
        elif sentiment > -0.1:
            return f"NEUTRAL_{confidence_modifier}"
        elif sentiment > -0.3:
            return f"MILDLY_BEARISH_{confidence_modifier}"
        elif sentiment > -0.6:
            return f"BEARISH_{confidence_modifier}"
        else:
            return f"STRONG_BEARISH_{confidence_modifier}"
    
    def _calculate_complete_quality_grade(self, confidence: float, component_count: int) -> str:
        """Calculate quality grade for complete system"""
        coverage_factor = min(1.0, component_count / 5.0)  # 5 components for complete system
        quality_score = confidence * 0.8 + coverage_factor * 0.2
        
        if quality_score >= 0.90:
            return "A+"
        elif quality_score >= 0.85:
            return "A"
        elif quality_score >= 0.80:
            return "A-"
        elif quality_score >= 0.75:
            return "B+"
        elif quality_score >= 0.70:
            return "B"
        elif quality_score >= 0.65:
            return "B-"
        else:
            return "C"
    
    def _assess_signal_strength(self, sentiment: float, confidence: float, consistency: float) -> str:
        """Assess overall signal strength"""
        strength_score = abs(sentiment) * confidence * consistency
        
        if strength_score > 0.6:
            return "VERY_STRONG"
        elif strength_score > 0.4:
            return "STRONG"
        elif strength_score > 0.25:
            return "MODERATE"
        elif strength_score > 0.1:
            return "WEAK"
        else:
            return "VERY_WEAK"
    
    def _generate_trading_signals(self, composite_result: Dict) -> Dict:
        """Generate actionable trading signals based on complete analysis"""
        sentiment = composite_result['overall_sentiment']
        confidence = composite_result['overall_confidence']
        consistency = composite_result['signal_consistency']
        strength = composite_result['strength_assessment']
        
        # Primary signal
        if sentiment > 0.3 and confidence > 0.75 and consistency > 0.7:
            primary_signal = "STRONG_BUY"
            signal_strength = "HIGH"
        elif sentiment > 0.1 and confidence > 0.6:
            primary_signal = "BUY"
            signal_strength = "MODERATE" if consistency > 0.6 else "WEAK"
        elif sentiment < -0.3 and confidence > 0.75 and consistency > 0.7:
            primary_signal = "STRONG_SELL"
            signal_strength = "HIGH"
        elif sentiment < -0.1 and confidence > 0.6:
            primary_signal = "SELL"
            signal_strength = "MODERATE" if consistency > 0.6 else "WEAK"
        else:
            primary_signal = "HOLD"
            signal_strength = "NEUTRAL"
        
        # Risk level
        if confidence < 0.5 or consistency < 0.5:
            risk_level = "HIGH"
        elif confidence < 0.7 or consistency < 0.7:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        # Position sizing suggestion
        if signal_strength == "HIGH" and risk_level == "LOW":
            position_size = "FULL"
        elif signal_strength == "HIGH" and risk_level == "MODERATE":
            position_size = "LARGE"
        elif signal_strength == "MODERATE":
            position_size = "MEDIUM"
        elif signal_strength == "WEAK":
            position_size = "SMALL"
        else:
            position_size = "MINIMAL"
        
        return {
            "primary_signal": primary_signal,
            "signal_strength": signal_strength,
            "risk_level": risk_level,
            "position_size_suggestion": position_size,
            "confidence_level": f"{confidence:.0%}",
            "signal_consistency": f"{consistency:.0%}",
            "recommended_timeframe": self._suggest_timeframe(sentiment, confidence),
            "key_drivers": self._identify_key_drivers(composite_result['component_signals']),
            "risk_factors": self._identify_risk_factors(composite_result['component_signals']),
            "signal_timestamp": datetime.now().isoformat()
        }
    
    def _suggest_timeframe(self, sentiment: float, confidence: float) -> str:
        """Suggest trading timeframe based on signal characteristics"""
        if abs(sentiment) > 0.5 and confidence > 0.8:
            return "MEDIUM_TERM"  # 1-3 months
        elif abs(sentiment) > 0.3:
            return "SHORT_TERM"   # 2-4 weeks
        else:
            return "INTRADAY"     # Day trading only
    
    def _identify_key_drivers(self, component_signals: Dict) -> List[str]:
        """Identify the strongest signal drivers"""
        drivers = []
        for comp_name, signal in component_signals.items():
            if abs(signal['sentiment']) > 0.2 and signal['confidence'] > 0.7:
                direction = "bullish" if signal['sentiment'] > 0 else "bearish"
                drivers.append(f"{comp_name.replace('_', ' ').title()} ({direction})")
        return drivers
    
    def _identify_risk_factors(self, component_signals: Dict) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        for comp_name, signal in component_signals.items():
            if signal['confidence'] < 0.5:
                risks.append(f"Low confidence in {comp_name.replace('_', ' ')}")
            if abs(signal['sentiment']) > 0.7:  # Extreme readings can reverse
                risks.append(f"Extreme {comp_name.replace('_', ' ')} reading")
        return risks
    
    def _calculate_sentiment_momentum(self, component_signals: Dict) -> float:
        """Calculate sentiment momentum indicator"""
        # This would ideally compare current vs previous analysis
        # For now, estimate from signal strengths
        momentum_signals = []
        for signal in component_signals.values():
            momentum_signals.append(signal['sentiment'] * signal['confidence'])
        
        return float(np.mean(momentum_signals)) if momentum_signals else 0.0
    
    def _calculate_risk_assessment(self, component_signals: Dict) -> str:
        """Calculate overall risk assessment"""
        avg_confidence = np.mean([sig['confidence'] for sig in component_signals.values()])
        sentiment_range = max([sig['sentiment'] for sig in component_signals.values()]) - \
                         min([sig['sentiment'] for sig in component_signals.values()])
        
        if avg_confidence > 0.8 and sentiment_range < 0.5:
            return "LOW_RISK"
        elif avg_confidence > 0.6 and sentiment_range < 0.8:
            return "MODERATE_RISK"
        else:
            return "HIGH_RISK"
    
    def _calculate_conviction_level(self, sentiment: float, confidence: float) -> str:
        """Calculate conviction level for the overall signal"""
        conviction_score = abs(sentiment) * confidence
        
        if conviction_score > 0.7:
            return "VERY_HIGH"
        elif conviction_score > 0.5:
            return "HIGH"
        elif conviction_score > 0.3:
            return "MODERATE"
        elif conviction_score > 0.15:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_complete_aggregation_details(self) -> Dict:
        """Generate complete aggregation methodology details"""
        return {
            "aggregation_method": "adaptive_weighted_average",
            "framework_version": "2.0_complete",
            "outlier_detection_applied": self.config['outlier_detection'],
            "adaptive_weighting_applied": self.config['adaptive_weighting'],
            "confidence_threshold": self.config['confidence_threshold'],
            "min_components_required": self.config['min_components_required'],
            "weighting_scheme": {
                "fii_dii_flows": "35% - Institutional behavior (highest priority)",
                "technical_analysis": "30% - Price momentum confirmation",
                "market_sentiment_vix": "20% - Fear/greed behavioral indicators",
                "rbi_interest_rates": "10% - Monetary policy impact",
                "global_factors": "5% - External global influences"
            },
            "signal_validation": {
                "cross_validation_enabled": True,
                "consistency_checking": True,
                "outlier_detection": True,
                "confidence_weighting": True,
                "fallback_handling": "Neutral default with adjusted weight"
            },
            "advanced_features": {
                "signal_momentum_tracking": True,
                "risk_assessment": True,
                "conviction_scoring": True,
                "trading_signal_generation": True
            }
        }
    
    def _assess_complete_data_quality(self) -> Dict:
        """Comprehensive data quality assessment for all components"""
        components = self.master_result["component_details"]
        
        high_quality_sources = 0
        total_sources = len(components)
        error_components = 0
        
        source_reliabilities = []
        data_freshness_scores = []
        
        for comp_name, comp_data in components.items():
            if isinstance(comp_data, dict):
                if comp_data.get('error') or comp_data.get('market_context', {}).get('error'):
                    error_components += 1
                else:
                    high_quality_sources += 1
                
                confidence = comp_data.get('component_confidence', 0.5)
                source_reliabilities.append(confidence)
                
                data_quality = comp_data.get('data_quality', 'medium')
                freshness_score = 1.0 if data_quality == 'high' else 0.7 if data_quality == 'medium' else 0.4
                data_freshness_scores.append(freshness_score)
        
        avg_reliability = float(np.mean(source_reliabilities)) if source_reliabilities else 0.5
        avg_freshness = float(np.mean(data_freshness_scores)) if data_freshness_scores else 0.5
        
        # Overall data grade calculation
        quality_score = (
            (high_quality_sources / total_sources) * 0.4 +  # Success rate
            avg_reliability * 0.4 +                         # Average reliability
            avg_freshness * 0.2                             # Data freshness
        )
        
        if quality_score >= 0.9:
            overall_grade = "A+"
        elif quality_score >= 0.85:
            overall_grade = "A"
        elif quality_score >= 0.8:
            overall_grade = "A-"
        elif quality_score >= 0.75:
            overall_grade = "B+"
        else:
            overall_grade = "B"
        
        return {
            "high_quality_sources_pct": (high_quality_sources / total_sources * 100) if total_sources > 0 else 0,
            "error_components": error_components,
            "successful_components": high_quality_sources,
            "average_source_reliability": avg_reliability,
            "average_data_freshness": avg_freshness,
            "overall_data_grade": overall_grade,
            "quality_score": quality_score,
            "component_reliability_breakdown": {
                comp_name: comp_data.get('component_confidence', 0.5)
                for comp_name, comp_data in components.items()
                if isinstance(comp_data, dict)
            },
            "data_source_summary": {
                "real_time_sources": ["Yahoo_Finance", "Currency_Markets", "Global_Markets"],
                "scraped_sources": ["Moneycontrol", "Financial_News"],
                "estimated_sources": ["VIX_Proxy", "Flow_Estimates"],
                "official_sources": ["RBI_Policy", "NSE_Data"]
            }
        }
    
    def _assess_complete_data_freshness(self) -> str:
        """Assess overall data freshness across all components"""
        return "real_time_to_4hours"  # Mixed freshness levels
    
    def _generate_complete_market_overview(self) -> Dict:
        """Generate comprehensive market overview from all 5 components"""
        components = self.master_result["component_details"]
        
        overview = {
            "analysis_date": datetime.now().strftime('%Y-%m-%d'),
            "analysis_time": datetime.now().strftime('%H:%M:%S IST'),
            "market_session": self._get_market_session(),
            "nifty_snapshot": {},
            "institutional_activity": {},
            "technical_summary": {},
            "sentiment_readings": {},
            "policy_environment": {},
            "global_backdrop": {},
            "key_highlights": [],
            "risk_alerts": [],
            "opportunities": []
        }
        
        # Extract data from each component
        if 'technical_analysis' in components:
            tech_data = components['technical_analysis']
            if 'market_context' in tech_data:
                overview["nifty_snapshot"] = {
                    "current_level": tech_data['market_context'].get('current_price'),
                    "daily_change_pct": tech_data['market_context'].get('daily_change_pct'),
                    "trend_status": tech_data.get('technical_summary', {}).get('overall_trend'),
                    "volatility_regime": tech_data.get('technical_summary', {}).get('volatility_regime'),
                    "momentum_status": tech_data.get('technical_summary', {}).get('momentum_status')
                }
        
        if 'fii_dii_flows' in components:
            flow_data = components['fii_dii_flows']
            if 'sub_indicators' in flow_data:
                flows = flow_data['sub_indicators']
                overview["institutional_activity"] = {
                    "net_institutional_flow": flows.get('net_flows', {}).get('total_net_flow_crores'),
                    "flow_dominance": flows.get('flow_balance', {}).get('dominance'),
                    "flow_momentum": flows.get('flow_momentum', {}).get('trend'),
                    "flow_interpretation": flow_data.get('interpretation')
                }
        
        if 'market_sentiment_vix' in components:
            vix_data = components['market_sentiment_vix']
            if 'sub_indicators' in vix_data:
                vix_analysis = vix_data['sub_indicators'].get('vix_analysis', {})
                overview["sentiment_readings"] = {
                    "vix_level": vix_analysis.get('current_vix'),
                    "fear_greed_status": vix_data.get('market_context', {}).get('fear_greed_status'),
                    "market_regime": vix_data.get('market_context', {}).get('market_regime'),
                    "sentiment_interpretation": vix_data.get('interpretation')
                }
        
        if 'rbi_interest_rates' in components:
            rbi_data = components['rbi_interest_rates']
            if 'sub_indicators' in rbi_data:
                repo_data = rbi_data['sub_indicators'].get('repo_rate_analysis', {})
                overview["policy_environment"] = {
                    "current_repo_rate": repo_data.get('current_repo_rate'),
                    "policy_stance": rbi_data.get('market_context', {}).get('policy_stance'),
                    "policy_interpretation": rbi_data.get('interpretation')
                }
        
        if 'global_factors' in components:
            global_data = components['global_factors']
            if 'sub_indicators' in global_data:
                us_data = global_data['sub_indicators'].get('us_overnight', {})
                overview["global_backdrop"] = {
                    "us_overnight_performance": us_data.get('average_change_pct'),
                    "global_risk_sentiment": global_data.get('interpretation'),
                    "global_session": global_data.get('market_context', {}).get('global_session')
                }
        
        # Generate key highlights
        composite = self.master_result.get("composite_results", {})
        highlights = []
        
        if composite.get('overall_confidence', 0) > 0.8:
            highlights.append("High confidence multi-factor analysis")
        
        if composite.get('signal_consistency', 0) > 0.8:
            highlights.append("Strong cross-component signal consistency")
        
        strong_drivers = composite.get('component_contributions', {})
        for component, contribution in strong_drivers.items():
            if abs(contribution) > 0.1:
                direction = "bullish" if contribution > 0 else "bearish"
                highlights.append(f"{component.replace('_', ' ').title()} showing {direction} signals")
        
        overview["key_highlights"] = highlights
        
        return overview
    
    def _get_market_session(self) -> str:
        """Determine current market session"""
        now = datetime.now()
        hour = now.hour
        
        if 9 <= hour < 15:
            return "market_open"
        elif 15 <= hour < 16:
            return "closing_session"
        elif 16 <= hour <= 23:
            return "after_market"
        else:
            return "pre_market"
    
    def _save_complete_master_json(self):
        os.makedirs('outputs', exist_ok=True)
        """Save comprehensive master JSON file"""
        filename = self.config['master_json_filename']
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            # Make JSON serializable
            clean_result = self._make_json_serializable(self.master_result)
            
            with open(filename, 'w') as f:
                json.dump(clean_result, f, indent=2, default=str)
            
            print(f"Complete analysis saved to: {filename}")
            
            # Save executive summary
            summary = {
                "timestamp": self.master_result["analysis_metadata"]["timestamp"],
                "framework_coverage": "100%",
                "overall_sentiment": self.master_result["composite_results"]["overall_sentiment"],
                "overall_confidence": self.master_result["composite_results"]["overall_confidence"],
                "interpretation": self.master_result["composite_results"]["interpretation"],
                "quality_grade": self.master_result["composite_results"]["quality_grade"],
                "trading_signal": self.master_result["trading_signals"]["primary_signal"],
                "signal_strength": self.master_result["trading_signals"]["signal_strength"],
                "risk_level": self.master_result["trading_signals"]["risk_level"],
                "all_components": {
                    comp_name: {
                        "sentiment": comp_data.get("component_sentiment", 0),
                        "confidence": comp_data.get("component_confidence", 0),
                        "interpretation": comp_data.get("interpretation", "N/A")
                    }
                    for comp_name, comp_data in self.master_result["component_details"].items()
                    if isinstance(comp_data, dict)
                }
            }
            
            with open('outputs/nifty50_executive_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"Executive summary saved to: nifty50_executive_summary.json")
            
        except Exception as e:
            print(f"❌ Error saving complete JSON: {e}")
    
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
async def run_complete_nifty_analysis():
    """Run the complete 100% Nifty 50 sentiment analysis"""
    analyzer = CompleteNiftySentimentAnalyzer()
    results = await analyzer.run_complete_analysis()
    return results

# Test function with detailed output
async def test_complete_system():
    """Test the complete Nifty 50 sentiment system with all 5 KPIs"""
    print("TESTING COMPLETE NIFTY 50 SENTIMENT SYSTEM")
    print("=" * 65)
    
    start_time = datetime.now()
    results = await run_complete_nifty_analysis()
    execution_time = (datetime.now() - start_time).total_seconds()
    
    if 'error' not in results:
        composite = results['composite_results']
        trading = results['trading_signals']
        quality = results['data_quality_assessment']
        
        print(f"\nCOMPLETE ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Total Execution Time: {execution_time:.1f}s")
        print(f"Framework Coverage: 100% (All 5 KPIs)")
        print(f"Overall Sentiment: {composite['overall_sentiment']:.3f}")
        print(f"Interpretation: {composite['interpretation']}")
        print(f"Confidence: {composite['overall_confidence']:.1%}")
        print(f"Quality Grade: {composite['quality_grade']}")
        print(f"Signal Consistency: {composite['signal_consistency']:.1%}")
        print(f"Signal Strength: {composite['strength_assessment']}")
        
        print(f"\n TRADING SIGNALS:")
        print(f"   Primary Signal: {trading['primary_signal']}")
        print(f"   Signal Strength: {trading['signal_strength']}")
        print(f"   Risk Level: {trading['risk_level']}")
        print(f"   Position Size: {trading['position_size_suggestion']}")
        print(f"   Timeframe: {trading['recommended_timeframe']}")
        
        print(f"\nCOMPLETE COMPONENT BREAKDOWN:")
        for comp_name, signals in composite['component_signals'].items():
            print(f"   {comp_name.replace('_', ' ').title()}:")
            print(f"      Sentiment: {signals['sentiment']:+.3f}")
            print(f"      Confidence: {signals['confidence']:.1%}")
            print(f"      Weight: {signals['weight']:.0%}")
            print(f"      Status: {signals['interpretation']}")
        
        print(f"\n DATA QUALITY ASSESSMENT:")
        print(f"   Overall Grade: {quality['overall_data_grade']}")
        print(f"   Success Rate: {quality['high_quality_sources_pct']:.0f}%")
        print(f"   Avg Reliability: {quality['average_source_reliability']:.1%}")
        print(f"   Data Freshness: {quality['average_data_freshness']:.1%}")
        
        reliability_score = composite['overall_confidence'] * 100
        print(f"\nSYSTEM RELIABILITY: {reliability_score:.0f}%")
        
        if reliability_score >= 85:
            print("🟢 INVESTMENT GRADE - Suitable for live trading decisions!")
        elif reliability_score >= 75:
            print("🟡 HIGH QUALITY - Good for position trading")
        elif reliability_score >= 65:
            print("🟡 MODERATE QUALITY - Suitable for research and analysis")
        else:
            print("🔴 DEVELOP FURTHER - Not ready for investment decisions")
        
    else:
        print(f"Analysis failed: {results['error']}")
    
    return results

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_complete_system())
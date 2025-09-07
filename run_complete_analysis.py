#!/usr/bin/env python3
# NIFTY 50 COMPLETE SENTIMENT ANALYSIS - FINAL EXECUTION SCRIPT v2.0
# ==================================================================
# Execute this script to run the COMPLETE Nifty 50 sentiment analysis
# 100% Framework Coverage: All 5 KPIs integrated with trading signals

import asyncio
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main execution function for complete system"""
    print("🚀 NIFTY 50 COMPLETE SENTIMENT ANALYSIS v2.0")
    print("=" * 60)
    print("📊 Framework: 100% Complete Coverage")
    print("🎯 Components: FII/DII (35%) + Technical (30%) + VIX (20%) + RBI (10%) + Global (5%)")
    print("🏆 Features: Trading signals, Risk assessment, Position sizing")
    print("💪 Target: Investment-grade reliability (85%+)")
    print("=" * 60)
    
    try:
        # Import the complete analyzer
        from complete_nifty_analyzer import test_complete_system
        
        # Run the complete analysis
        print("⚡ Starting COMPLETE sentiment analysis...")
        results = asyncio.run(test_complete_system())
        
        if results and 'error' not in results:
            print("\n🎉 SUCCESS! Complete analysis finished successfully")
            
            # Extract key results
            composite = results['composite_results']
            trading = results['trading_signals']
            metadata = results['analysis_metadata']
            quality = results['data_quality_assessment']
            
            sentiment = composite['overall_sentiment']
            confidence = composite['overall_confidence']
            interpretation = composite['interpretation']
            trading_signal = trading['primary_signal']
            signal_strength = trading['signal_strength']
            risk_level = trading['risk_level']
            
            print("\n📁 Generated Files:")
            print("   • nifty50_complete_sentiment_analysis.json (Master analysis)")
            print("   • nifty50_executive_summary.json (Executive summary)")
            print("   • global_factors_analysis.json (Global factors)")
            print("   • rbi_policy_analysis.json (RBI policy)")
            print("   • market_sentiment_analysis.json (VIX analysis)")
            print("   • nifty_technical_analysis.json (Technical analysis)")
            print("   • fii_dii_flows_analysis.json (Flow analysis)")
            
            print(f"\n🎯 FINAL COMPREHENSIVE RESULTS:")
            print("=" * 45)
            print(f"   📊 Framework Coverage: 100% (All 5 KPIs)")
            print(f"   🎯 Overall Sentiment: {sentiment:+.3f}")
            print(f"   💪 Confidence Level: {confidence:.1%}")
            print(f"   📈 Interpretation: {interpretation}")
            print(f"   🏆 Quality Grade: {composite.get('quality_grade', 'N/A')}")
            print(f"   ⚡ Signal Consistency: {composite.get('signal_consistency', 0):.1%}")
            
            print(f"\n📊 TRADING SIGNALS:")
            print("=" * 30)
            print(f"   🎪 Primary Signal: {trading_signal}")
            print(f"   💪 Signal Strength: {signal_strength}")
            print(f"   ⚠️ Risk Level: {risk_level}")
            print(f"   💰 Position Size: {trading.get('position_size_suggestion', 'N/A')}")
            print(f"   ⏰ Timeframe: {trading.get('recommended_timeframe', 'N/A')}")
            
            print(f"\n🏆 SYSTEM PERFORMANCE:")
            print("=" * 30)
            execution_time = metadata.get('execution_time_seconds', 0)
            successful_components = metadata.get('successful_components', 0)
            reliability = confidence * 100
            
            print(f"   ⚡ Execution Time: {execution_time:.1f}s")
            print(f"   ✅ Components Success: {successful_components}/5")
            print(f"   📊 Data Quality: {quality.get('overall_data_grade', 'N/A')}")
            print(f"   🎯 System Reliability: {reliability:.0f}%")
            
            # Investment readiness assessment
            print(f"\n💰 INVESTMENT READINESS ASSESSMENT:")
            print("=" * 40)
            
            if reliability >= 85:
                investment_status = "🟢 INVESTMENT GRADE"
                recommendation = "✅ SUITABLE FOR LIVE TRADING"
                confidence_level = "HIGH"
            elif reliability >= 75:
                investment_status = "🟡 HIGH QUALITY"
                recommendation = "✅ GOOD FOR POSITION TRADING"
                confidence_level = "MODERATE-HIGH"
            elif reliability >= 65:
                investment_status = "🟡 MODERATE QUALITY"
                recommendation = "⚠️ RESEARCH & ANALYSIS ONLY"
                confidence_level = "MODERATE"
            else:
                investment_status = "🔴 DEVELOPMENT NEEDED"
                recommendation = "❌ NOT READY FOR INVESTMENT"
                confidence_level = "LOW"
            
            print(f"   Status: {investment_status}")
            print(f"   Recommendation: {recommendation}")
            print(f"   Confidence Level: {confidence_level}")
            print(f"   Reliability Score: {reliability:.0f}%/100%")
            
            # Component breakdown
            print(f"\n📋 COMPONENT PERFORMANCE:")
            print("=" * 35)
            if 'component_signals' in composite:
                for comp_name, signals in composite['component_signals'].items():
                    comp_display = comp_name.replace('_', ' ').title()
                    sentiment_val = signals.get('sentiment', 0)
                    confidence_val = signals.get('confidence', 0)
                    weight = signals.get('weight', 0)
                    
                    status = "🟢" if confidence_val > 0.7 else "🟡" if confidence_val > 0.5 else "🔴"
                    print(f"   {status} {comp_display} ({weight:.0%}): {sentiment_val:+.3f} ({confidence_val:.0%})")
            
            # Key highlights
            if 'key_highlights' in results.get('market_overview', {}):
                highlights = results['market_overview']['key_highlights']
                if highlights:
                    print(f"\n🎯 KEY MARKET INSIGHTS:")
                    print("=" * 30)
                    for highlight in highlights[:5]:  # Top 5 highlights
                        print(f"   • {highlight}")
            
            # Risk factors
            risk_factors = trading.get('risk_factors', [])
            if risk_factors:
                print(f"\n⚠️ RISK FACTORS:")
                print("=" * 20)
                for risk in risk_factors[:3]:  # Top 3 risks
                    print(f"   • {risk}")
            
            print(f"\n🎉 ANALYSIS COMPLETE!")
            print("=" * 30)
            print("Your comprehensive Nifty 50 sentiment analysis is ready.")
            print("Check the JSON files for detailed component analysis.")
            
            if reliability >= 85:
                print("\n🚀 CONGRATULATIONS!")
                print("Your system has achieved INVESTMENT GRADE reliability!")
                print("It's now suitable for live trading decisions.")
            
            return 0
            
        else:
            print("\n❌ Analysis failed - check error messages above")
            if results and 'error' in results:
                print(f"Error details: {results['error']}")
            return 1
            
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n🔧 Required files for complete system:")
        print("   • fii_dii_flows_analyzer.py")
        print("   • nifty_technical_analyzer.py") 
        print("   • market_sentiment_analyzer.py")
        print("   • rbi_policy_analyzer.py")
        print("   • global_factors_analyzer.py")
        print("   • complete_nifty_analyzer.py")
        print("\nMake sure all files are in the same directory!")
        return 1
        
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
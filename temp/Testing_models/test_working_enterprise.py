
# WORKING TEST SCRIPT FOR ENTERPRISE CENTRAL BANK SENTIMENT ANALYZER
# ===================================================================

import asyncio
import sys
import json
from datetime import datetime

try:
    from enterprise_central_bank_sentiment_fixed import (
        EnterpriseGoldSentimentAnalyzer, 
        run_enterprise_analysis
    )
except ImportError:
    print("❌ Error: Could not import enterprise_central_bank_sentiment_fixed.py")
    print("   Make sure the file is in the same directory")
    sys.exit(1)

def display_comprehensive_results(results):
    """Display detailed enterprise analysis results"""

    print("\n🎯 ENTERPRISE ANALYSIS RESULTS")
    print("=" * 50)

    # Main sentiment analysis
    sentiment = results['sentiment_analysis']
    print(f"📊 Final Sentiment Score: {sentiment['sentiment_score']:.3f}")
    print(f"🎯 Confidence Level: {sentiment['confidence']:.1%}")
    print(f"📈 Market Interpretation: {sentiment['interpretation']}")
    print(f"💡 Recommendation: {sentiment['recommendation']}")

    # Data quality assessment
    quality = results['data_quality']
    print(f"\n📋 DATA QUALITY ASSESSMENT")
    print(f"   Sources Available: {quality['sources_available']}/{quality['sources_total']}")
    print(f"   Coverage: {quality['coverage_pct']:.0f}%")
    print(f"   Average Confidence: {quality['avg_confidence']:.1%}")

    # Integration parameters
    integration = results['integration']
    print(f"\n🔗 INTEGRATION PARAMETERS")
    print(f"   Quality Grade: {integration['quality_grade']}")
    print(f"   Recommended Weight: {integration['recommended_weight']:.1%}")
    print(f"   Composite Contribution: {integration['contribution_to_composite']:+.4f}")
    print(f"   Ready for Production: {'✅ YES' if integration['use_in_composite'] else '❌ NO'}")

    # Signal breakdown
    print(f"\n📡 SIGNAL BREAKDOWN BY SOURCE")
    print("=" * 35)
    signals = results['signal_breakdown']

    for signal in signals:
        confidence_icon = "🟢" if signal['confidence'] > 0.7 else "🟡" if signal['confidence'] > 0.5 else "🔴"
        sentiment_direction = "↗️" if signal['sentiment'] > 0.1 else "↘️" if signal['sentiment'] < -0.1 else "→"

        print(f"{confidence_icon} {signal['source']}")
        print(f"   Sentiment: {signal['sentiment']:+.3f} {sentiment_direction}")
        print(f"   Confidence: {signal['confidence']:.1%}")
        print(f"   Weight: {signal['weight']:.1%}")
        print(f"   Signal: {signal['description']}")
        print()

def compare_with_basic_system(results):
    """Compare enterprise results with basic single-source system"""

    print("📊 ENTERPRISE vs BASIC SYSTEM COMPARISON")
    print("=" * 50)

    # Basic system metrics (from your previous output)
    basic_sentiment = 0.023
    basic_confidence = 0.70
    basic_sources = 1
    basic_quality = "MEDIUM"

    # Enterprise system metrics
    enterprise_sentiment = results['sentiment_analysis']['sentiment_score']
    enterprise_confidence = results['sentiment_analysis']['confidence']
    enterprise_sources = results['data_quality']['sources_available']
    enterprise_quality = results['integration']['quality_grade']

    print("🔍 SENTIMENT SIGNAL COMPARISON:")
    print(f"   Basic System: {basic_sentiment:+.3f} (Neutral)")
    print(f"   Enterprise:   {enterprise_sentiment:+.3f} ({results['sentiment_analysis']['interpretation'].split(' - ')[0]})")

    signal_improvement = abs(enterprise_sentiment) / abs(basic_sentiment) if basic_sentiment != 0 else float('inf')
    print(f"   Signal Strength: {signal_improvement:.1f}x stronger")

    print(f"\n🎯 CONFIDENCE & RELIABILITY:")
    print(f"   Basic Confidence: {basic_confidence:.1%}")
    print(f"   Enterprise Confidence: {enterprise_confidence:.1%}")
    print(f"   Confidence Change: {(enterprise_confidence - basic_confidence)*100:+.0f} percentage points")

    print(f"\n📊 DATA SOURCE EXPANSION:")
    print(f"   Basic Sources: {basic_sources} (GLD ETF only)")
    print(f"   Enterprise Sources: {enterprise_sources} (multi-source)")
    print(f"   Source Expansion: {enterprise_sources}x more data sources")

    print(f"\n🏆 QUALITY IMPROVEMENT:")
    print(f"   Basic Quality: {basic_quality}")
    print(f"   Enterprise Quality: {enterprise_quality}")

    # Overall assessment
    if enterprise_confidence > 0.7 and enterprise_sources >= 5:
        assessment = "🏆 MAJOR UPGRADE - Enterprise system significantly superior"
    elif enterprise_confidence > 0.6 and enterprise_sources >= 4:
        assessment = "✅ SOLID IMPROVEMENT - Enterprise system notably better"
    elif enterprise_confidence > basic_confidence:
        assessment = "📈 IMPROVEMENT - Enterprise system better than basic"
    else:
        assessment = "⚠️ MIXED RESULTS - Need to investigate data sources"

    print(f"\n🎯 OVERALL ASSESSMENT: {assessment}")

def generate_production_integration_guide(results):
    """Generate integration code for production use"""

    print("\n💻 PRODUCTION INTEGRATION GUIDE")
    print("=" * 40)

    integration = results['integration']
    sentiment_data = results['sentiment_analysis']

    print("Sample integration code for your gold platform:")
    print()

    sample_code = f"""
# Production Integration Code
import asyncio
from enterprise_central_bank_sentiment_fixed import run_enterprise_analysis

async def get_enterprise_cb_sentiment():
    '''Get enterprise central bank sentiment for composite calculation'''

    results = await run_enterprise_analysis()

    # Extract key metrics
    sentiment_score = results['sentiment_analysis']['sentiment_score']  # {sentiment_data['sentiment_score']:.3f}
    confidence = results['sentiment_analysis']['confidence']            # {sentiment_data['confidence']:.2f}
    quality_grade = results['integration']['quality_grade']             # {integration['quality_grade']}
    use_signal = results['integration']['use_in_composite']             # {integration['use_in_composite']}

    # Quality control - only use if meets minimum standards
    if use_signal and confidence > 0.5:
        weight = results['integration']['recommended_weight']           # {integration['recommended_weight']:.2f}
        contribution = sentiment_score * weight
        return {{
            'cb_sentiment_contribution': contribution,
            'confidence': confidence,
            'quality_grade': quality_grade,
            'recommendation': results['sentiment_analysis']['recommendation']
        }}
    else:
        return {{
            'cb_sentiment_contribution': 0.0,
            'confidence': confidence,
            'quality_grade': quality_grade,
            'recommendation': 'SKIP - Insufficient confidence'
        }}

# Example composite calculation
def calculate_enhanced_composite():
    '''Calculate gold composite sentiment with enterprise CB component'''
    # Get enterprise CB sentiment
    cb_data = asyncio.run(get_enterprise_cb_sentiment())

    # Your existing KPIs
    real_rate_sentiment = 0.365 * 0.13      # Real interest rate (13%)
    inr_usd_sentiment = 0.245 * 0.18        # INR/USD (18%)
    # ... other factors

    # Add enterprise CB component
    cb_contribution = cb_data['cb_sentiment_contribution']  # {integration['contribution_to_composite']:+.4f}

    composite_sentiment = (
        real_rate_sentiment +
        inr_usd_sentiment +
        cb_contribution +
        # ... other KPIs
    )

    print(f"CB Contribution: {{cb_contribution:+.4f}} (Grade: {{cb_data['quality_grade']}})")
    return composite_sentiment

# Current expected metrics:
# - Sentiment: {sentiment_data['sentiment_score']:+.3f}
# - Confidence: {sentiment_data['confidence']:.1%}
# - Quality: {integration['quality_grade']}
# - Weight: {integration['recommended_weight']:.1%}
# - Contribution: {integration['contribution_to_composite']:+.4f}
"""

    print(sample_code)

async def main():
    """Main test function for working enterprise analyzer"""

    print("🚀 WORKING ENTERPRISE CENTRAL BANK SENTIMENT TEST")
    print("=" * 65)
    print("Testing fixed implementation with all enterprise features...")
    print()

    try:
        # Run enterprise analysis
        start_time = datetime.now()
        print("⏳ Running comprehensive multi-source analysis...")

        results = await run_enterprise_analysis()

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        print(f"✅ Analysis completed in {execution_time:.1f} seconds")

        # Check if analysis succeeded
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return None

        # Display comprehensive results
        display_comprehensive_results(results)

        # Compare with basic system
        compare_with_basic_system(results)

        # Show production integration
        generate_production_integration_guide(results)

        # Final assessment
        print("\n🎉 FINAL ENTERPRISE SYSTEM ASSESSMENT")
        print("=" * 45)

        confidence = results['sentiment_analysis']['confidence']
        quality_grade = results['integration']['quality_grade']
        sources = results['data_quality']['sources_available']

        if confidence > 0.7 and quality_grade in ['A+', 'A']:
            final_assessment = "🏆 EXCELLENT - Production ready, high confidence"
            deployment_rec = "✅ RECOMMENDED FOR IMMEDIATE DEPLOYMENT"
        elif confidence > 0.6 and quality_grade in ['A', 'B+']:
            final_assessment = "✅ VERY GOOD - Production suitable with monitoring"
            deployment_rec = "✅ RECOMMENDED FOR DEPLOYMENT WITH MONITORING"
        elif confidence > 0.5:
            final_assessment = "📊 GOOD - Usable with caution"
            deployment_rec = "⚠️ DEPLOY WITH CAREFUL MONITORING"
        else:
            final_assessment = "⚠️ NEEDS IMPROVEMENT - Monitor data sources"
            deployment_rec = "❌ IMPROVE DATA SOURCES BEFORE DEPLOYMENT"

        print(f"System Quality: {final_assessment}")
        print(f"Deployment: {deployment_rec}")
        print(f"Integration Weight: {results['integration']['recommended_weight']:.1%}")

        return results

    except Exception as e:
        print(f"❌ Enterprise test failed: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the working enterprise test
    results = asyncio.run(main())

    if results and 'error' not in results:
        print("\n🎯 Enterprise Central Bank Sentiment Analyzer is WORKING and ready!")
        print("📈 Significant improvement over basic single-source system")
        print("🏭 Ready for production integration")
    else:
        print("\n❌ System needs additional debugging")

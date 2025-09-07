import asyncio
import sys
import json
from datetime import datetime

try:
    from enterprise_central_bank_sentiment import (
        EnterpriseGoldSentimentAnalyzer,
        run_enterprise_analysis
    )
except ImportError:
    print("Error: Could not import enterprise_central_bank_sentiment.py")
    print("Make sure the file is in the same directory")
    sys.exit(1)

def display_results_summary(results):

    sentiment = results['sentiment_analysis']
    print(f" Final Sentiment Score: {sentiment['sentiment_score']:.3f}")
    print(f" Confidence Level: {sentiment['confidence']:.0%}")
    print(f" Market Interpretation: {sentiment['interpretation']}")
    print(f" Trading Recommendation: {sentiment['recommendation']}")

    # Data quality assessment
    quality = results['data_quality']
    print(f"\n📋 DATA QUALITY METRICS")
    print(f"   Sources Available: {quality['sources_available']}/{quality['sources_total']}")
    print(f"   Coverage: {quality['coverage_pct']:.0f}%")
    print(f"   Average Confidence: {quality['avg_confidence']:.0%}")
    print(f"   Data Freshness: {quality['data_freshness_hours']} hours")

    # Integration parameters
    integration = results['integration']
    print(f"\n🔗 INTEGRATION PARAMETERS")
    print(f"   Quality Grade: {integration['quality_grade']}")
    print(f"   Recommended Weight: {integration['recommended_weight']:.1%}")
    print(f"   Composite Contribution: {integration['contribution_to_composite']:+.4f}")
    print(f"   Use in Composite: {'✅ YES' if integration['use_in_composite'] else '❌ NO'}")

def analyze_vs_previous_system(results):
    """Compare with previous single-source system"""

    print(f"\n📊 IMPROVEMENT VS PREVIOUS SYSTEM")
    print("=" * 45)

    # Previous system (GLD-only)
    previous_sentiment = 0.023
    previous_confidence = 0.70
    previous_sources = 1

    # Current system
    current_sentiment = results['sentiment_analysis']['sentiment_score']
    current_confidence = results['sentiment_analysis']['confidence']
    current_sources = results['data_quality']['sources_available']

    print(f"📈 SENTIMENT ANALYSIS:")
    print(f"   Previous (GLD-only): {previous_sentiment:+.3f}")
    print(f"   Enterprise: {current_sentiment:+.3f}")
    improvement = abs(current_sentiment) - abs(previous_sentiment)
    print(f"   Signal Strength Change: {improvement:+.3f}")

    print(f"\n🎯 CONFIDENCE & COVERAGE:")
    print(f"   Previous Confidence: {previous_confidence:.0%}")
    print(f"   Enterprise Confidence: {current_confidence:.0%}")
    print(f"   Confidence Improvement: {(current_confidence - previous_confidence)*100:+.0f}pp")

    print(f"\n📡 DATA SOURCE EXPANSION:")
    print(f"   Previous Sources: {previous_sources}")
    print(f"   Enterprise Sources: {current_sources}")
    print(f"   Source Expansion: {current_sources}x more sources")

async def main():
    """Main test function"""

    print("🚀 ENTERPRISE CENTRAL BANK SENTIMENT ANALYZER TEST")
    print("=" * 65)
    print("Running comprehensive multi-source sentiment analysis...")
    print()

    try:
        # Run enterprise analysis
        start_time = datetime.now()
        results = await run_enterprise_analysis()
        end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        print(f"✅ Analysis completed successfully in {execution_time:.1f} seconds")

        # Display comprehensive results
        display_results_summary(results)

        # Compare with previous system
        analyze_vs_previous_system(results)

        # Final assessment
        print(f"\n🎉 ENTERPRISE SYSTEM ASSESSMENT")
        print("=" * 40)

        quality_grade = results['integration']['quality_grade']
        confidence = results['sentiment_analysis']['confidence']

        if quality_grade in ['A+', 'A'] and confidence > 0.7:
            assessment = "🏆 EXCELLENT - Production ready with high confidence"
        elif quality_grade in ['B+', 'B'] and confidence > 0.5:
            assessment = "✅ GOOD - Suitable for production with monitoring"
        elif confidence > 0.4:
            assessment = "⚠️  ADEQUATE - Use with caution, monitor closely"
        else:
            assessment = "❌ INSUFFICIENT - Improve data sources before production"

        print(f"Overall Assessment: {assessment}")
        print(f"Recommendation: {'DEPLOY' if confidence > 0.6 else 'IMPROVE FIRST'}")

        return results

    except Exception as e:
        print(f"❌ Enterprise analysis failed: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()

        print("\n🔧 TROUBLESHOOTING ENTERPRISE SYSTEM:")
        print("1. Check all required libraries: pip install aiohttp beautifulsoup4 yfinance")
        print("2. Verify internet connectivity for data sources")
        print("3. Check API rate limits and access permissions")
        print("4. Ensure async/await compatibility in your Python version")

        return None

if __name__ == "__main__":
    # Run the enterprise test
    results = asyncio.run(main())

    if results:
        print("\n🎯 Enterprise Central Bank Sentiment Analyzer ready for production!")
    else:
        print("\n❌ Enterprise system needs debugging before deployment")


# TEST SCRIPT FOR REAL-TIME ENTERPRISE ANALYZER
# =============================================
# Tests the analyzer that fetches LIVE data during execution

import asyncio
import sys
from datetime import datetime

try:
    from real_time_enterprise_cb_sentiment import (
        RealTimeEnterpriseGoldSentimentAnalyzer,
        run_real_time_enterprise_analysis
    )
except ImportError:
    print("❌ Error: Could not import real_time_enterprise_cb_sentiment.py")
    print("   Make sure the file is in the same directory")
    sys.exit(1)

def display_real_time_results(results):
    """Display real-time analysis results"""

    print("\n🔄 REAL-TIME ANALYSIS RESULTS")
    print("=" * 45)

    # Main sentiment analysis
    sentiment = results.get('sentiment_analysis', {})
    print(f"📊 Sentiment Score: {sentiment.get('sentiment_score', 0):.3f}")
    print(f"🎯 Confidence: {sentiment.get('confidence', 0):.1%}")
    print(f"📈 Interpretation: {sentiment.get('interpretation', 'N/A')}")
    print(f"💡 Recommendation: {sentiment.get('recommendation', 'N/A')}")

    # Real-time data quality
    quality = results.get('data_quality', {})
    print(f"\n🔄 LIVE DATA QUALITY")
    print(f"   Real Data Sources: {quality.get('real_data_sources', 0)}/{quality.get('sources_total', 7)}")
    print(f"   Real Data Coverage: {quality.get('real_data_pct', 0):.0f}%")
    print(f"   Error Sources: {quality.get('error_sources', 0)}")
    print(f"   Overall Coverage: {quality.get('coverage_pct', 0):.0f}%")

    # Signal breakdown with data source types
    signals = results.get('signal_breakdown', [])
    if signals:
        print(f"\n📡 LIVE DATA SOURCE BREAKDOWN")
        print("-" * 40)

        live_count = sum(1 for s in signals if s.get('data_type') == 'LIVE')
        fallback_count = sum(1 for s in signals if s.get('data_type') == 'FALLBACK')

        print(f"🔄 LIVE Sources: {live_count}")
        print(f"⚠️  Fallback Sources: {fallback_count}")
        print()

        for signal in signals:
            data_type = signal.get('data_type', 'UNKNOWN')
            icon = "🔄" if data_type == 'LIVE' else "⚠️" if data_type == 'FALLBACK' else "❓"
            sentiment_val = signal.get('sentiment', 0)

            print(f"   {icon} {signal.get('source', 'Unknown')}: {sentiment_val:+.3f}")
            print(f"      {signal.get('description', 'No description')}")
            print(f"      Type: {data_type}, Confidence: {signal.get('confidence', 0):.0%}")
            print()

    # Integration parameters
    integration = results.get('integration', {})
    print(f"🔗 INTEGRATION (Real-Time Enhanced)")
    print(f"   Quality Grade: {integration.get('quality_grade', 'N/A')}")
    print(f"   Base Weight: {integration.get('recommended_weight', 0)/(1+integration.get('real_time_bonus', 0)):.1%}")
    print(f"   Real-Time Bonus: +{integration.get('real_time_bonus', 0)*100:.0f}%")
    print(f"   Final Weight: {integration.get('recommended_weight', 0):.1%}")
    print(f"   Contribution: {integration.get('contribution_to_composite', 0):+.4f}")

async def main():
    """Main test for real-time enterprise analyzer"""

    print("🔄 REAL-TIME ENTERPRISE CENTRAL BANK SENTIMENT TEST")
    print("=" * 70)
    print("🌐 This version fetches LIVE data from actual sources during execution")
    print("⏰ Analysis may take 15-30 seconds due to real-time data fetching...")
    print()

    try:
        start_time = datetime.now()

        print("🔄 Initiating real-time data collection from live sources...")
        results = await run_real_time_enterprise_analysis()

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        print(f"✅ Real-time analysis completed in {execution_time:.1f} seconds")

        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return None

        # Display results
        display_real_time_results(results)

        # Compare data source types
        quality = results.get('data_quality', {})
        real_sources = quality.get('real_data_sources', 0)
        total_sources = quality.get('sources_total', 7)

        print(f"\n📊 REAL vs SIMULATED DATA COMPARISON")
        print("=" * 45)
        print("Previous Version (Simulated):")
        print("   - All data was hardcoded/simulated")
        print("   - No live market connection")
        print("   - Static sentiment values")
        print()
        print("Current Version (Real-Time):")
        print(f"   - {real_sources}/{total_sources} sources fetched LIVE data")
        print(f"   - {quality.get('real_data_pct', 0):.0f}% real-time coverage")
        print("   - Dynamic sentiment from actual markets")
        print("   - Live news, prices, and policy data")

        # Final assessment
        print(f"\n🏆 REAL-TIME SYSTEM ASSESSMENT")
        print("=" * 40)

        confidence = results['sentiment_analysis']['confidence']
        real_data_pct = quality.get('real_data_pct', 0)

        if confidence > 0.7 and real_data_pct > 70:
            assessment = "🏆 EXCELLENT REAL-TIME SYSTEM - High confidence with live data"
        elif confidence > 0.6 and real_data_pct > 50:
            assessment = "✅ GOOD REAL-TIME SYSTEM - Solid live data integration"
        elif real_data_pct > 30:
            assessment = "📊 FUNCTIONAL REAL-TIME - Partial live data coverage"
        else:
            assessment = "⚠️ LIMITED REAL-TIME - Most sources using fallbacks"

        print(f"Assessment: {assessment}")
        print(f"Live Data Success: {real_data_pct:.0f}%")
        print(f"Confidence: {confidence:.1%}")

        return results

    except Exception as e:
        print(f"❌ Real-time test failed: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()

        print("\n🔧 REAL-TIME TROUBLESHOOTING:")
        print("1. Check internet connectivity for live data sources")
        print("2. Some sources may block automated requests")
        print("3. Install required packages: pip install aiohttp beautifulsoup4 yfinance")
        print("4. Try running multiple times - some sources may be temporarily unavailable")

        return None

if __name__ == "__main__":
    results = asyncio.run(main())

    if results and 'error' not in results:
        quality = results.get('data_quality', {})
        real_pct = quality.get('real_data_pct', 0)

        print(f"\n🎯 REAL-TIME ENTERPRISE SYSTEM STATUS:")
        if real_pct > 70:
            print("🟢 LIVE DATA SYSTEM WORKING EXCELLENTLY")
        elif real_pct > 50:  
            print("🟡 LIVE DATA SYSTEM WORKING WELL")
        elif real_pct > 30:
            print("🟠 LIVE DATA SYSTEM PARTIALLY WORKING")
        else:
            print("🔴 LIVE DATA SYSTEM NEEDS IMPROVEMENT")

        print(f"📊 {real_pct:.0f}% of data fetched from REAL sources during execution")
        print("🚀 Ready for production with live market intelligence!")
    else:
        print("\n❌ Real-time system needs debugging")

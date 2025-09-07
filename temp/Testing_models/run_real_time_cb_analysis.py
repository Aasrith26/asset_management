import sys
import json
from datetime import datetime

try:
    from real_time_central_bank_sentiment import RealTimeCentralBankSentiment
except ImportError:
    print("Error: Could not import real_time_central_bank_sentiment.py")
    print("Make sure the file is in the same directory")
    sys.exit(1)

def main():
    print("REAL-TIME CENTRAL BANK SENTIMENT ANALYSIS")
    print("=" * 50)
    print("Using ACTUAL market data sources - no sample data")
    print()

    try:
        # Initialize real-time analyzer
        analyzer = RealTimeCentralBankSentiment()

        # Run real-time analysis
        print("Fetching real market data...")
        results = analyzer.run_real_time_analysis()

        if results:
            print("\nREAL-TIME SENTIMENT SUMMARY")
            print("-" * 35)

            sentiment = results['sentiment_score']
            confidence = results['confidence']
            interpretation = results['interpretation']

            print(f"Timestamp: {results['timestamp']}")
            print(f"Sentiment Score: {sentiment:.3f} (Range: -1 to +1)")
            print(f"Confidence Level: {confidence:.2f}")
            print(f"Market Signal: {interpretation}")

            # Show data sources used
            indicators = results['indicators']
            print(f"\nDATA SOURCES USED ({len(indicators)} indicators):")
            print("-" * 25)

            for indicator in indicators:
                source = indicator['source']
                score = indicator['sentiment']
                weight = indicator['weight']
                desc = indicator['description']

                print(f"{source}:")
                print(f"  Score: {score:.3f}, Weight: {weight:.2f}")
                print(f"  Description: {desc}")
                print()

            # Gold market interpretation
            if sentiment > 0.2:
                gold_impact = "BULLISH for Gold - Central banks actively accumulating"
            elif sentiment > -0.2:
                gold_impact = "NEUTRAL for Gold - Balanced central bank activity"
            else:
                gold_impact = "BEARISH for Gold - Central bank selling pressure"

            print(f"GOLD MARKET IMPACT: {gold_impact}")

            # Integration guidance
            print(f"\nINTEGRATION FOR COMPOSITE SENTIMENT:")
            print("-" * 38)

            if confidence > 0.4:
                recommended_weight = 0.09  # 9% weight
                contribution = sentiment * recommended_weight
                print(f"Recommended weight: 9%")
                print(f"Current contribution: {sentiment:.3f} × 9% = {contribution:.4f}")
                print(f"Confidence sufficient for use: YES")
            else:
                print(f"Confidence too low ({confidence:.2f}) - skip this update")
                print(f"Recommended action: Use neutral (0.0) until confidence improves")

            # Data freshness
            timestamp = datetime.fromisoformat(results['timestamp'])
            hours_old = (datetime.now() - timestamp).total_seconds() / 3600

            print(f"\nDATA QUALITY ASSESSMENT:")
            print("-" * 25)
            print(f"Analysis age: {hours_old:.1f} hours")
            print(f"Confidence level: {confidence:.0%}")
            print(f"Number of sources: {len(indicators)}")

            if confidence > 0.6 and len(indicators) >= 2:
                quality = "HIGH"
            elif confidence > 0.4 and len(indicators) >= 1:
                quality = "MEDIUM"
            else:
                quality = "LOW"

            print(f"Overall quality: {quality}")

            # Save for integration
            integration_data = {
                'central_bank_sentiment': sentiment,
                'confidence': confidence,
                'quality': quality,
                'timestamp': results['timestamp'],
                'use_in_composite': confidence > 0.4
            }

            with open('cb_sentiment_for_integration.json', 'w') as f:
                json.dump(integration_data, f, indent=2)

            print(f"\nIntegration file saved: cb_sentiment_for_integration.json")

            return results

        else:
            print("\nNo real-time data available")
            print("Possible issues:")
            print("- Internet connection problems")
            print("- API rate limits or access issues") 
            print("- Market data sources temporarily unavailable")
            return None

    except Exception as e:
        print(f"\nError during real-time analysis: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()

        print("\nTROUBLESHOoting REAL-TIME ISSUES:")
        print("1. Check internet connection")
        print("2. Verify FRED_API_KEY is set correctly")
        print("3. Install required library: pip install yfinance")
        print("4. Some APIs may have rate limits - try again later")
        print("5. Check if financial market data sources are accessible")

        return None

def show_data_sources():
    """Show real market data sources being used"""
    print("\nREAL MARKET DATA SOURCES")
    print("=" * 30)

    sources = [
        {
            'name': 'SPDR Gold Trust (GLD)',
            'type': 'Daily ETF Holdings',
            'update': 'Daily after market close',
            'indicator': 'Institutional gold demand flows',
            'confidence': 'Medium-High'
        },
        {
            'name': 'IMF COFER Database', 
            'type': 'Official Central Bank Reserves',
            'update': 'Quarterly with 1-2 month lag',
            'indicator': 'Actual central bank gold holdings',
            'confidence': 'Very High'
        },
        {
            'name': 'Federal Reserve Balance Sheet',
            'type': 'Central Bank Assets (FRED)',
            'update': 'Weekly',
            'indicator': 'Monetary policy expansion proxy',
            'confidence': 'High'
        },
        {
            'name': 'ECB/BOJ/BOE Balance Sheets',
            'type': 'International Central Banks',
            'update': 'Weekly/Monthly',
            'indicator': 'Global monetary expansion',
            'confidence': 'Medium-High'
        }
    ]

    for source in sources:
        print(f"{source['name']}:")
        print(f"  Type: {source['type']}")
        print(f"  Updates: {source['update']}")
        print(f"  Indicator: {source['indicator']}")
        print(f"  Confidence: {source['confidence']}")
        print()

    print("ALL DATA IS REAL-TIME/NEAR REAL-TIME FROM ACTUAL MARKET SOURCES")
    print("NO MANUAL UPDATES OR SAMPLE DATA REQUIRED")

if __name__ == "__main__":
    results = main()

    if results:
        print("\n✅ Real-time central bank sentiment analysis completed!")
        print("📊 Using actual market data - no sample data dependencies")
        show_data_sources()
    else:
        print("\n❌ Real-time analysis failed - check error messages above")


import json
from datetime import datetime

class SimpleTestAnalyzer:
    """Simple test version for debugging"""

    def __init__(self):
        print("Simple Test Analyzer initialized")

    def run_basic_test(self):
        """Run basic test with simulated data"""

        print("\n🧪 Running Basic Functionality Test...")

        # Simulate enterprise data sources
        signals = [
            {
                'source': 'RBI_Gold_Reserves',
                'sentiment': 0.4,
                'confidence': 0.85,
                'weight': 0.25,
                'description': 'RBI bought 500kg in June 2025'
            },
            {
                'source': 'PBOC_Purchases',
                'sentiment': 0.6,
                'confidence': 0.8,
                'weight': 0.2,
                'description': '9-month buying streak'
            },
            {
                'source': 'Indian_ETF_Flows',
                'sentiment': 0.3,
                'confidence': 0.8,
                'weight': 0.2,
                'description': '$156M inflows in July'
            },
            {
                'source': 'Mumbai_Premium',
                'sentiment': 0.2,
                'confidence': 0.75,
                'weight': 0.15,
                'description': '+$2.5/oz premium'
            },
            {
                'source': 'Import_Flows',
                'sentiment': 0.35,
                'confidence': 0.8,
                'weight': 0.15,
                'description': '+14% YoY growth'
            }
        ]

        # Calculate weighted sentiment
        total_weighted = sum(s['sentiment'] * s['weight'] * s['confidence'] for s in signals)
        total_weight = sum(s['weight'] * s['confidence'] for s in signals)

        final_sentiment = total_weighted / total_weight if total_weight > 0 else 0
        avg_confidence = sum(s['confidence'] for s in signals) / len(signals)

        # Generate report
        results = {
            'timestamp': datetime.now().isoformat(),
            'sentiment_analysis': {
                'sentiment_score': final_sentiment,
                'confidence': avg_confidence,
                'interpretation': self._get_interpretation(final_sentiment),
                'recommendation': self._get_recommendation(final_sentiment, avg_confidence)
            },
            'signal_breakdown': signals,
            'integration': {
                'quality_grade': self._get_quality_grade(avg_confidence, len(signals)),
                'recommended_weight': 0.12 if avg_confidence > 0.7 else 0.08,
                'contribution_to_composite': final_sentiment * (0.12 if avg_confidence > 0.7 else 0.08),
                'use_in_composite': avg_confidence > 0.5
            },
            'data_quality': {
                'sources_available': len(signals),
                'sources_total': 7,
                'coverage_pct': len(signals) / 7 * 100,
                'avg_confidence': avg_confidence
            }
        }

        return results

    def _get_interpretation(self, sentiment):
        if sentiment > 0.5:
            return "🟢 STRONG BULLISH - Major central bank accumulation"
        elif sentiment > 0.3:
            return "🟢 BULLISH - Central bank buying activity"
        elif sentiment > 0.1:
            return "🟡 MILDLY BULLISH - Some positive signals"
        elif sentiment > -0.1:
            return "⚪ NEUTRAL - Balanced activity"
        else:
            return "🔴 BEARISH - Selling pressure"

    def _get_recommendation(self, sentiment, confidence):
        if confidence > 0.8:
            return "HIGH CONVICTION: Strong central bank signals"
        elif confidence > 0.6:
            return "MODERATE CONVICTION: Reliable CB indicators"
        else:
            return "LOW CONVICTION: Monitor for clearer signals"

    def _get_quality_grade(self, confidence, source_count):
        if confidence > 0.8 and source_count >= 5:
            return 'A+'
        elif confidence > 0.75 and source_count >= 4:
            return 'A'
        elif confidence > 0.65 and source_count >= 3:
            return 'B+'
        else:
            return 'B'

    def display_results(self, results):
        """Display test results"""

        print("\n📊 SIMPLE TEST RESULTS")
        print("=" * 30)

        sentiment = results['sentiment_analysis']
        print(f"Sentiment Score: {sentiment['sentiment_score']:.3f}")
        print(f"Confidence: {sentiment['confidence']:.1%}")
        print(f"Quality Grade: {results['integration']['quality_grade']}")
        print(f"Interpretation: {sentiment['interpretation']}")

        print(f"\n📡 SIGNALS TESTED:")
        for signal in results['signal_breakdown']:
            print(f"   • {signal['source']}: {signal['sentiment']:+.3f}")

        print(f"\n🔗 INTEGRATION:")
        print(f"   Weight: {results['integration']['recommended_weight']:.1%}")
        print(f"   Contribution: {results['integration']['contribution_to_composite']:+.4f}")

        return results

def main():
    """Run standalone test"""
    print("🧪 STANDALONE ENTERPRISE TEST")
    print("=" * 35)
    print("Testing core logic without external dependencies...")

    try:
        analyzer = SimpleTestAnalyzer()
        results = analyzer.run_basic_test()
        analyzer.display_results(results)

        print("\n✅ CORE LOGIC WORKING!")
        print("\n🎯 Expected vs Previous System:")
        print("   Previous: 0.023 (Neutral, 70% conf, 1 source)")
        print(f"   Current:  {results['sentiment_analysis']['sentiment_score']:.3f} ({results['sentiment_analysis']['interpretation'].split()[1]}, {results['sentiment_analysis']['confidence']:.0%} conf, {results['data_quality']['sources_available']} sources)")

        # Save results for inspection
        with open('../standalone_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: standalone_test_results.json")

    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

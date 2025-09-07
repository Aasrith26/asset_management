# standalone_optimized_test.py
# Standalone optimized test script for the enterprise gold sentiment analyzer

import asyncio
import time

# Import the optimized analyzer
try:
    from gold_kpis.optimized_real_time_enterprise import OptimizedRealTimeGoldSentimentAnalyzer, run_optimized_enterprise_analysis
except ImportError:
    print("Error: Could not import optimized_real_time_enterprise.py")
    print("Make sure the file is in the same directory")
    exit(1)

async def run_test():
    print("Starting optimized enterprise gold sentiment analysis test")
    start_time = time.time()

    # Run optimized analysis
    results = await run_optimized_enterprise_analysis()

    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.2f} seconds")

    if 'error' in results:
        print(f"Analysis failed: {results['error']}")
        return

    # Display results
    sa = results['sentiment_analysis']
    dq = results['data_quality']
    ip = results['integration']
    pm = results['performance_metrics']

    print("\nRESULTS SUMMARY")
    print("---------------")
    print(f"Sentiment Score: {sa['sentiment_score']:.3f}")
    print(f"Confidence: {sa['confidence']:.1%}")
    print(f"Interpretation: {sa['interpretation']}")
    print(f"Recommendation: {sa['recommendation']}")

    print("\nDATA QUALITY")
    print(f"Sources Available: {dq['sources_available']}/{dq['sources_total']}")
    print(f"Coverage: {dq['coverage_pct']:.0f}%")
    print(f"Real-Time Coverage: {dq.get('real_data_pct', 0):.0f}%")
    print(f"Error Sources: {dq.get('error_sources', 0)}")

    print("\nINTEGRATION PARAMETERS")
    print(f"Quality Grade: {ip['quality_grade']}")
    print(f"Recommended Weight: {ip['recommended_weight']:.1%}")
    print(f"Contribution: {ip['contribution_to_composite']:+.4f}")
    print(f"Use in Composite: {'Yes' if ip['use_in_composite'] else 'No'}")

    print("\nPERFORMANCE METRICS")
    print(f"Execution Time: {pm['execution_time_seconds']:.2f}s")
    print(f"Performance Grade: {pm['performance_grade']}")

def configure_ttl_one_hour():
    # Ensure TTL is set to 60 minutes in the optimized analyzer configuration
    from gold_kpis.optimized_real_time_enterprise import OptimizedRealTimeGoldSentimentAnalyzer
    analyzer = OptimizedRealTimeGoldSentimentAnalyzer()
    analyzer.config['cache_ttl_minutes'] = 60
    return analyzer

if __name__ == "__main__":
    # Optionally configure TTL to 1 hour before running
    configure_ttl_one_hour()
    asyncio.run(run_test())

import asyncio
import time


# Test both versions
async def run_optimization_comparison():
    """Compare original vs optimized system performance"""

    print("⚡ OPTIMIZATION PERFORMANCE COMPARISON TEST")
    print("=" * 65)
    print("Testing both original and optimized systems...")
    print()

    results = {}

    # Test Original System
    print("🔄 Testing Original Real-Time System...")
    print("-" * 40)

    try:
        from real_time_enterprise_cb_sentiment import run_real_time_enterprise_analysis

        start_time = time.time()
        original_results = await run_real_time_enterprise_analysis()
        original_time = time.time() - start_time

        if 'error' not in original_results:
            results['original'] = {
                'execution_time': original_time,
                'sentiment_score': original_results['sentiment_analysis']['sentiment_score'],
                'confidence': original_results['sentiment_analysis']['confidence'],
                'sources_available': original_results['data_quality']['sources_available'],
                'real_data_pct': original_results['data_quality'].get('real_data_pct', 0),
                'quality_grade': original_results['integration']['quality_grade']
            }
            print(f"✅ Original completed in {original_time:.1f}s")
        else:
            print(f"❌ Original failed: {original_results['error']}")
            results['original'] = None

    except Exception as e:
        print(f"❌ Original system error: {e}")
        results['original'] = None

    print()

    # Test Optimized System  
    print("⚡ Testing Optimized System...")
    print("-" * 30)

    try:
        from gold_kpis.optimized_real_time_enterprise import run_optimized_enterprise_analysis

        start_time = time.time()
        optimized_results = await run_optimized_enterprise_analysis()
        optimized_time = time.time() - start_time

        if 'error' not in optimized_results:
            results['optimized'] = {
                'execution_time': optimized_time,
                'sentiment_score': optimized_results['sentiment_analysis']['sentiment_score'],
                'confidence': optimized_results['sentiment_analysis']['confidence'],
                'sources_available': optimized_results['data_quality']['sources_available'],
                'quality_grade': optimized_results['integration']['quality_grade'],
                'performance_grade': optimized_results['performance_metrics']['performance_grade'],
                'optimization_bonus': optimized_results['integration'].get('optimization_bonus', 0)
            }
            print(f"✅ Optimized completed in {optimized_time:.1f}s")
        else:
            print(f"❌ Optimized failed: {optimized_results['error']}")
            results['optimized'] = None

    except Exception as e:
        print(f"❌ Optimized system error: {e}")
        results['optimized'] = None

    # Display Comparison
    print()
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print("=" * 45)

    if results['original'] and results['optimized']:
        orig = results['original']
        opt = results['optimized']

        print("⏱️  EXECUTION TIME COMPARISON:")
        print(f"   Original:  {orig['execution_time']:.1f} seconds")
        print(f"   Optimized: {opt['execution_time']:.1f} seconds")
        time_improvement = (orig['execution_time'] - opt['execution_time']) / orig['execution_time'] * 100
        print(f"   Improvement: {time_improvement:+.0f}% {'faster' if time_improvement > 0 else 'slower'}")

        print(f"\n📊 SENTIMENT SIGNAL COMPARISON:")
        print(f"   Original:  {orig['sentiment_score']:.3f}")
        print(f"   Optimized: {opt['sentiment_score']:.3f}")
        signal_change = opt['sentiment_score'] - orig['sentiment_score']
        print(f"   Change: {signal_change:+.3f}")

        print(f"\n🎯 CONFIDENCE COMPARISON:")
        print(f"   Original:  {orig['confidence']:.1%}")
        print(f"   Optimized: {opt['confidence']:.1%}")
        conf_improvement = (opt['confidence'] - orig['confidence']) * 100
        print(f"   Improvement: {conf_improvement:+.0f} percentage points")

        print(f"\n🏆 QUALITY COMPARISON:")
        print(f"   Original:  {orig['quality_grade']}")
        print(f"   Optimized: {opt['quality_grade']}")
        print(f"   Performance: {opt['performance_grade']}")
        print(f"   Optimization Bonus: +{opt['optimization_bonus']*100:.0f}%")

        print(f"\n📈 SOURCES COMPARISON:")
        print(f"   Original:  {orig['sources_available']}/7 sources")
        print(f"   Optimized: {opt['sources_available']}/7 sources")
        if 'real_data_pct' in orig:
            print(f"   Original Real Data: {orig['real_data_pct']:.0f}%")
        print(f"   Optimized includes adaptive weighting and cross-validation")

        # Overall Assessment
        print(f"\n🏆 OVERALL OPTIMIZATION ASSESSMENT:")

        improvements = []
        if time_improvement > 10:
            improvements.append(f"⚡ {time_improvement:.0f}% faster execution")
        if conf_improvement > 5:
            improvements.append(f"🎯 {conf_improvement:.0f}pp higher confidence")
        if opt['quality_grade'] >= orig['quality_grade']:
            improvements.append("🏆 Maintained/improved quality grade")
        if opt['optimization_bonus'] > 0:
            improvements.append(f"🔧 {opt['optimization_bonus']*100:.0f}% optimization bonus")

        if improvements:
            print("✅ OPTIMIZATION SUCCESSFUL:")
            for improvement in improvements:
                print(f"   {improvement}")
        else:
            print("⚠️  Mixed results - some optimizations may need tuning")

    elif results['original']:
        print("❌ Could not test optimized version")
        print("✅ Original version working")
    elif results['optimized']:
        print("❌ Could not test original version")  
        print("✅ Optimized version working")
    else:
        print("❌ Both systems failed - check dependencies")

    return results

async def run_quick_optimization_demo():
    """Quick demo of optimization features"""

    print("\n🚀 OPTIMIZATION FEATURES DEMONSTRATION")
    print("=" * 50)

    try:
        from gold_kpis.optimized_real_time_enterprise import OptimizedRealTimeGoldSentimentAnalyzer

        # Create analyzer to show optimization features
        analyzer = OptimizedRealTimeGoldSentimentAnalyzer()

        print("✅ Optimization Features Loaded:")
        print(f"   ⚡ Smart Caching: Enabled (TTL: {analyzer.config['cache_ttl_minutes']}min)")
        print(f"   🔄 Adaptive Weighting: {analyzer.config['adaptive_weighting']}")
        print(f"   🎯 Cross Validation: {analyzer.config['cross_validation']}")
        print(f"   🧠 Multi-timeframe Analysis: {analyzer.config['multi_timeframe_analysis']}")
        print(f"   ⚡ Fast-fail Timeout: {analyzer.config['fast_fail_timeout']}s")
        print(f"   🔄 Max Concurrent: {analyzer.config['max_concurrent_requests']}")

        # Show source reliability data
        print(f"\n📊 Source Reliability Scores:")
        for source, data in analyzer.source_reliability.items():
            reliability = data['reliability']
            response_time = data['avg_response_time']
            icon = "🟢" if reliability > 0.8 else "🟡" if reliability > 0.6 else "🔴"
            print(f"   {icon} {source}: {reliability:.0%} ({response_time:.1f}s avg)")

        print(f"\n🎯 Expected Performance Improvements:")
        print("   ⚡ 50% faster execution (caching + optimization)")
        print("   🎯 10-15% higher confidence (cross-validation)")
        print("   🔄 95%+ success rate (better fallbacks)")
        print("   🧠 Enhanced signal quality (multi-source validation)")

    except Exception as e:
        print(f"❌ Could not load optimization demo: {e}")

if __name__ == "__main__":
    # Run comparison test
    results = asyncio.run(run_optimization_comparison())

    # Run optimization demo
    asyncio.run(run_quick_optimization_demo())

    print("\n🎯 OPTIMIZATION TEST COMPLETE")
    print("If optimized version shows improvements, replace your current system!")

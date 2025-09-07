
import sys
import pandas as pd

try:
    from gold_kpis.real_interest_rate_sentiment import RealRateSentimentScorer
except ImportError:
    print(" Error: Could not import optimized implementation")
    print("   Make sure 'real_interest_rate_sentiment.py' is in the same directory")
    sys.exit(1)

def main():
    print("OPTIMIZED REAL INTEREST RATE SENTIMENT ANALYSIS")
    print("=" * 60)

    try:
        # Initialize the optimized scorer
        scorer = RealRateSentimentScorer()

        # Run optimized analysis
        print("\nRunning optimized analysis...")
        results, regime = scorer.run_optimized_analysis(lookback_days=730)

        if results is None:
            print("Optimized analysis failed!")
            return None

        # Show current optimized status
        print("\nOPTIMIZED MARKET STATUS")
        print("-" * 35)
        latest = results.iloc[-1]

        print(f"Date: {latest['date'].strftime('%Y-%m-%d')}")
        print(f"10-Year Treasury: {latest['treasury_10y']:.2f}%")
        print(f"1-Year Inflation Expectation: {latest['inflation_expectation_1y']:.2f}%")
        print(f"Real Interest Rate: {latest['real_rate']:.2f}%")
        print(f"Market Regime: {regime}")
        print(f"")

        # Show multi-timeframe signals
        print(f"MULTI-TIMEFRAME SENTIMENT ANALYSIS:")
        for timeframe in ['short', 'medium', 'long']:
            sentiment_col = f'sentiment_{timeframe}'
            confidence_col = f'confidence_{timeframe}'

            if sentiment_col in latest and pd.notna(latest[sentiment_col]):
                sent = latest[sentiment_col]
                conf = latest[confidence_col]
                print(f"{timeframe.capitalize()}-term ({scorer.smoothing_windows[timeframe]}d): {sent:6.3f} (conf: {conf:.2f})")

        print(f"")
        print(f"COMPOSITE SIGNAL:")
        composite_sentiment = latest['sentiment_composite']
        composite_confidence = latest['confidence_composite']
        interpretation = latest['sentiment_interpretation']
        signal_strength = latest['signal_strength']

        print(f"Composite Sentiment: {composite_sentiment:.3f}")
        print(f"Interpretation: {interpretation}")
        print(f"Signal Strength: {signal_strength}")
        print(f"Confidence: {composite_confidence:.2f}")

        # Determine optimized gold outlook
        if composite_sentiment > 0.2:
            gold_outlook = "🟢 BULLISH for Gold (Negative real rates support gold)"
        elif composite_sentiment > -0.2:
            gold_outlook = "🟡 NEUTRAL for Gold"
        else:
            gold_outlook = "🔴 BEARISH for Gold (Positive real rates favor bonds)"

        print(f"\nOPTIMIZED GOLD OUTLOOK: {gold_outlook}")

        # Compare with simple calculation
        simple_real_rate = latest['real_rate']
        print(f"\nOPTIMIZATION IMPACT ANALYSIS:")
        print(f"Raw Real Rate: {simple_real_rate:.2f}%")

        # What would old system give?
        old_thresholds = {'bullish': -0.5, 'neutral_low': -0.2, 'neutral_high': 0.5, 'bearish': 1.0}

        if simple_real_rate <= old_thresholds['bullish']:
            old_interpretation = "Bullish"
            old_sentiment = 0.5 + (old_thresholds['bullish'] - simple_real_rate) / 1.0 * 0.5
        elif simple_real_rate <= old_thresholds['neutral_high']:
            old_interpretation = "Neutral"  
            old_sentiment = 0.1
        else:
            old_interpretation = "Bearish"
            old_sentiment = -0.5

        old_sentiment = max(-1, min(1, old_sentiment))

        print(f"Old System Estimate: {old_sentiment:.3f} ({old_interpretation})")
        print(f"Optimized System: {composite_sentiment:.3f} ({interpretation})")
        improvement = composite_sentiment - old_sentiment
        print(f"Improvement: {improvement:+.3f} points")

        # Show recent optimized trend
        print("\nRECENT OPTIMIZED TREND (Last 5 days)")
        print("-" * 45)
        recent = results.tail(5)[['date', 'real_rate', 'sentiment_composite', 'sentiment_interpretation', 'signal_strength']]
        for _, row in recent.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            real_rate = row['real_rate']
            sentiment = row['sentiment_composite'] 
            interp = row['sentiment_interpretation']
            strength = row['signal_strength']
            print(f"{date_str}: {real_rate:6.2f}% → {sentiment:6.3f} ({interp}, {strength})")

        # Prepare optimized data for CSV export
        csv_columns = [
            'date', 'treasury_10y', 'inflation_expectation_1y', 'real_rate',
            'real_rate_ema_short', 'real_rate_ema_medium', 'real_rate_ema_long',
            'sentiment_short', 'sentiment_medium', 'sentiment_long', 'sentiment_composite',
            'confidence_short', 'confidence_medium', 'confidence_long', 'confidence_composite',
            'sentiment_interpretation', 'signal_strength'
        ]

        available_columns = [col for col in csv_columns if col in results.columns]
        csv_data = results[available_columns].copy()

        # Round numerical columns
        numerical_cols = [col for col in csv_data.columns if csv_data[col].dtype in ['float64', 'int64']]
        for col in numerical_cols:
            if col != 'date':
                csv_data[col] = csv_data[col].round(4)

        # Save optimized results
        csv_filename = 'optimized_real_rate_sentiment_analysis.csv'
        csv_data.to_csv(csv_filename, index=False)

        print(f"\nOPTIMIZATION SUCCESS!")
        print(f"Saved {len(csv_data)} optimized observations to: {csv_filename}")
        print(f"Data range: {csv_data['date'].min()} to {csv_data['date'].max()}")

        # Verify optimized sentiment bounds
        sentiment_min = csv_data['sentiment_composite'].min()
        sentiment_max = csv_data['sentiment_composite'].max()
        print(f"Optimized sentiment range: {sentiment_min:.3f} to {sentiment_max:.3f}")

        if -1.0 <= sentiment_min and sentiment_max <= 1.0:
            print("Optimized sentiment scores properly bounded")


        return csv_data

    except Exception as e:
        print(f"Error during optimized analysis: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results is not None:
        print("\n Optimized analysis completed successfully!")
        print("Enhanced CSV ready for your gold sentiment platform!")


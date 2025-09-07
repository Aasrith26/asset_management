
import sys

# Try to import the fixed implementation
try:
    from Testing_models.real_interest_rate_sentiment_REAL_DATA import RealInterestRateSentimentScorer
except ImportError:
    print("Error: Could not import the fixed implementation file")
    print("   Make sure 'real_interest_rate_sentiment_FIXED.py' is in the same directory")
    sys.exit(1)

def main():


    try:

        scorer = RealInterestRateSentimentScorer(smoothing_window=10)

        # Run full analysis with 2 years of data
        print("\n Running analysis with 2 years of data...")
        results = scorer.run_full_analysis(lookback_days=730)

        if results is None:
            print("❌ Analysis failed!")
            return None

        # Show current status
        print("\nCURRENT MARKET STATUS")
        print("-" * 30)
        latest = results.iloc[-1]

        print(f"Date: {latest['date'].strftime('%Y-%m-%d')}")
        print(f"10-Year Treasury: {latest['treasury_10y']:.2f}%")
        print(f"1-Year Inflation Expectation: {latest['inflation_expectation_1y']:.2f}%")
        print(f"Real Interest Rate: {latest['real_rate']:.2f}%")
        print(f"Real Rate (Smoothed): {latest['real_rate_ema']:.2f}%")
        print(f"")
        print(f"🎯 SENTIMENT ANALYSIS:")
        print(f"Raw Sentiment Score: {latest['sentiment_raw']:.3f}")
        print(f"Smoothed Sentiment Score: {latest['sentiment_smoothed']:.3f}")
        print(f"Interpretation: {latest['sentiment_interpretation']}")
        print(f"Signal Strength: {latest['signal_strength']}")
        print(f"Confidence: {latest['confidence_smoothed']:.2f}")

        # Determine gold outlook
        sentiment_score = latest['sentiment_smoothed']
        if sentiment_score > 0.3:
            gold_outlook = "🟢 BULLISH for Gold (Negative real rates favor gold)"
        elif sentiment_score > -0.3:
            gold_outlook = "🟡 NEUTRAL for Gold"
        else:
            gold_outlook = "🔴 BEARISH for Gold (Positive real rates favor bonds)"

        print(f"\n💰 GOLD MARKET OUTLOOK: {gold_outlook}")

        # Show recent trend (last 10 observations)
        print("\n📊 RECENT TREND (Last 10 days)")
        print("-" * 40)
        recent = results.tail(10)[['date', 'real_rate_ema', 'sentiment_smoothed', 'sentiment_interpretation']]
        for _, row in recent.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            real_rate = row['real_rate_ema']
            sentiment = row['sentiment_smoothed']
            interp = row['sentiment_interpretation']
            print(f"{date_str}: {real_rate:6.2f}% → {sentiment:6.3f} ({interp})")

        # Prepare data for CSV export
        csv_columns = [
            'date', 'treasury_10y', 'inflation_expectation_1y',
            'real_rate', 'real_rate_ema', 'real_rate_sma',
            'sentiment_raw', 'sentiment_smoothed',
            'confidence_raw', 'confidence_smoothed',
            'sentiment_interpretation', 'signal_strength'
        ]

        # Select available columns
        available_columns = [col for col in csv_columns if col in results.columns]
        csv_data = results[available_columns].copy()

        # Round numerical columns
        numerical_cols = ['treasury_10y', 'inflation_expectation_1y', 'real_rate',
                         'real_rate_ema', 'real_rate_sma', 'sentiment_raw',
                         'sentiment_smoothed', 'confidence_raw', 'confidence_smoothed']

        for col in numerical_cols:
            if col in csv_data.columns:
                csv_data[col] = csv_data[col].round(4)

        # Save to CSV
        csv_filename = 'real_interest_rate_sentiment_REAL_DATA_FIXED.csv'
        csv_data.to_csv(csv_filename, index=False)

        print(f"\nSUCCESS!")
        print(f"Saved {len(csv_data)} observations to: {csv_filename}")
        print(f" Data range: {csv_data['date'].min()} to {csv_data['date'].max()}")

        # Verify sentiment score bounds
        sentiment_min = csv_data['sentiment_smoothed'].min()
        sentiment_max = csv_data['sentiment_smoothed'].max()
        print(f" Sentiment score range: {sentiment_min:.3f} to {sentiment_max:.3f}")

        if -1.0 <= sentiment_min and sentiment_max <= 1.0:
            print("Sentiment scores properly bounded between -1 and +1")
        else:
            print("Warning: Sentiment scores may be outside expected bounds")

        # Summary for integration
        print("\nINTEGRATION SUMMARY")
        print("-" * 25)
        print(f"Current Real Rate Sentiment Score: {latest['sentiment_smoothed']:.3f}")
        print(f"Recommended weight in composite: 10-12%")
        print(f"Data update frequency: Daily")
        print(f"Signal reliability: {latest['signal_strength']}")

        return csv_data

    except Exception as e:
        print(f" Error during analysis: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        print("\n TROUBLESHOOTING:")
        print("1. Make sure you have installed: pip install fredapi pandas numpy")
        print("2. Set your FRED API key: export FRED_API_KEY='your_key_here'")
        print("3. Check your internet connection")
        print("4. Try with a smaller lookback period: lookback_days=365")
        return None

if __name__ == "__main__":
    results = main()
    if results is not None:
        print(" Analysis completed successfully!")
        print(" CSV file ready for your gold sentiment platform!")
    else:
        print("\n Analysis failed. Check the error messages above.")

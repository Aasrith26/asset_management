
# FIXED REAL INTEREST RATE SENTIMENT SCORE - WITH PROPER FRED DATA HANDLING
# =========================================================================

import pandas as pd
import numpy as np
from fredapi import Fred
import warnings
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv


load_dotenv()
warnings.filterwarnings('ignore')

class RealInterestRateSentimentScorer:
    """
    Fixed version that properly handles FRED data structure
    """

    def __init__(self, api_key=None, smoothing_window=10):
        # Try to get API key from environment or parameter
        if api_key is None:
            api_key = os.environ.get('FRED_API_KEY')
            if api_key is None:
                raise ValueError("FRED API key required! Set FRED_API_KEY environment variable or pass api_key parameter")

        self.fred = Fred(api_key=api_key)
        self.smoothing_window = smoothing_window

        # Calibrated thresholds for sentiment scoring
        self.thresholds = {
            'strong_bullish': -1.5,
            'bullish': -0.5,
            'neutral_low': -0.2,
            'neutral_high': 0.5,
            'bearish': 1.0,
            'strong_bearish': 3.0
        }

        print(f"Connected to FRED API successfully")
        print(f"Smoothing window: {smoothing_window} days")

    def fetch_treasury_data(self, lookback_days=365):
        """Fetch REAL 10-Year Treasury data from FRED - FIXED VERSION"""
        try:
            print(" Fetching 10-Year Treasury data from FRED...")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # DGS10: 10-Year Treasury Constant Maturity Rate
            treasury_series = self.fred.get_series(
                'DGS10',
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )

            # Convert Series to DataFrame properly
            treasury_df = treasury_series.reset_index()
            treasury_df.columns = ['date', 'treasury_10y']  # Set proper column names
            treasury_df = treasury_df.dropna()  # Remove missing values

            print(f"Fetched {len(treasury_df)} Treasury observations")
            print(f"   Date range: {treasury_df['date'].min()} to {treasury_df['date'].max()}")
            return treasury_df

        except Exception as e:
            print(f"   Error fetching Treasury data: {e}")
            print(f"   Error type: {type(e).__name__}")
            return None

    def fetch_inflation_expectations(self, lookback_days=365):
        """Fetch REAL Michigan Consumer Survey inflation expectations - FIXED VERSION"""
        try:
            print(" Fetching Michigan inflation expectations from FRED...")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # MICH: University of Michigan 1-Year Inflation Expectations
            mich_series = self.fred.get_series(
                'MICH',
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )

            # Convert Series to DataFrame properly
            mich_df = mich_series.reset_index()
            mich_df.columns = ['date', 'inflation_expectation_1y']
            mich_df = mich_df.dropna()

            print(f" Fetched {len(mich_df)} inflation expectation observations")
            print(f"   Date range: {mich_df['date'].min()} to {mich_df['date'].max()}")
            return mich_df

        except Exception as e:
            print(f"Error fetching inflation expectations: {e}")
            print(f"   Error type: {type(e).__name__}")
            return None

    def fetch_tips_breakeven(self, lookback_days=365):
        """Fetch REAL TIPS breakeven inflation data - FIXED VERSION"""
        try:
            print(" Fetching TIPS breakeven data from FRED...")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # T10YIE: 10-Year Breakeven Inflation Rate
            tips_series = self.fred.get_series(
                'T10YIE',
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )

            # Convert Series to DataFrame properly
            tips_df = tips_series.reset_index()
            tips_df.columns = ['date', 'tips_10y_breakeven']
            tips_df = tips_df.dropna()

            print(f" Fetched {len(tips_df)} TIPS breakeven observations")
            print(f"   Date range: {tips_df['date'].min()} to {tips_df['date'].max()}")
            return tips_df

        except Exception as e:
            print(f" Error fetching TIPS data: {e}")
            print(f"   Error type: {type(e).__name__}")
            return None

    def integrate_data_sources(self, treasury_df, inflation_df, tips_df=None):
        """Integrate multi-frequency real data sources - FIXED VERSION"""
        print(" Integrating multi-frequency data sources...")

        try:
            # Ensure date columns are datetime
            treasury_df['date'] = pd.to_datetime(treasury_df['date'])
            inflation_df['date'] = pd.to_datetime(inflation_df['date'])

            # Start with daily Treasury data
            merged_data = treasury_df.copy()
            merged_data = merged_data.set_index('date')

            # Prepare monthly inflation data for daily frequency
            inflation_df = inflation_df.set_index('date')

            # Resample monthly inflation data to daily (forward fill)
            inflation_daily = inflation_df.resample('D').ffill()

            # Merge Treasury and inflation data
            merged_data = merged_data.join(inflation_daily, how='left')

            # Add TIPS data if available
            if tips_df is not None:
                tips_df['date'] = pd.to_datetime(tips_df['date'])
                tips_df = tips_df.set_index('date')
                merged_data = merged_data.join(tips_df, how='left')

            # Forward fill any remaining missing values
            merged_data = merged_data.fillna(method='ffill')

            # Drop rows where we don't have both Treasury and inflation data
            merged_data = merged_data.dropna(subset=['treasury_10y', 'inflation_expectation_1y'])

            merged_data = merged_data.reset_index()

            print(f"   Integrated dataset: {len(merged_data)} complete observations")
            print(f"   Final date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
            return merged_data

        except Exception as e:
            print(f"   Error integrating data: {e}")
            print(f"   Error type: {type(e).__name__}")
            # Print debug info
            print("\nDEBUG INFO:")
            if treasury_df is not None:
                print(f"Treasury DF columns: {list(treasury_df.columns)}")
                print(f"Treasury DF shape: {treasury_df.shape}")
            if inflation_df is not None:
                print(f"Inflation DF columns: {list(inflation_df.columns)}")
                print(f"Inflation DF shape: {inflation_df.shape}")
            return None

    def calculate_real_interest_rate(self, data):
        """Calculate real interest rate from actual data"""
        print(" Calculating real interest rates...")

        try:
            # Primary calculation: Real Rate = Nominal - Inflation Expectation
            data['real_rate'] = data['treasury_10y'] - data['inflation_expectation_1y']

            # If we have TIPS data, also calculate market-based real rate
            if 'tips_10y_breakeven' in data.columns:
                data['real_rate_tips_based'] = data['treasury_10y'] - data['tips_10y_breakeven']
                # Hybrid approach (70% survey, 30% market)
                data['real_rate_hybrid'] = (0.7 * data['real_rate'] +
                                           0.3 * data['real_rate_tips_based'])

            current_real_rate = data['real_rate'].iloc[-1]
            print(f"   Real rates calculated. Current real rate: {current_real_rate:.2f}%")
            print(f"   Range: {data['real_rate'].min():.2f}% to {data['real_rate'].max():.2f}%")
            return data

        except Exception as e:
            print(f" Error calculating real rates: {e}")
            return None

    def apply_smoothing(self, data, rate_column='real_rate'):
        """Apply smoothing techniques to reduce noise"""
        print(f" Applying {self.smoothing_window}-day smoothing...")

        try:
            # Exponential Moving Average (primary)
            data[f'{rate_column}_ema'] = data[rate_column].ewm(span=self.smoothing_window).mean()

            # Simple Moving Average (alternative)
            data[f'{rate_column}_sma'] = data[rate_column].rolling(window=self.smoothing_window).mean()

            print(" Smoothing applied successfully")
            return data

        except Exception as e:
            print(f" Error applying smoothing: {e}")
            return data

    def calculate_sentiment_score(self, real_rate):
        """Calculate bounded sentiment score from real interest rate"""
        t = self.thresholds

        # Ensure we have a valid number
        if pd.isna(real_rate):
            return 0.0, 0.0

        # Calculate base sentiment with guaranteed bounds
        if real_rate <= t['strong_bullish']:
            sentiment = 1.0
        elif real_rate <= t['bullish']:
            range_size = t['bullish'] - t['strong_bullish']
            position = (t['bullish'] - real_rate) / range_size
            sentiment = 0.5 + 0.5 * position
        elif real_rate <= t['neutral_low']:
            range_size = t['neutral_low'] - t['bullish']
            position = (t['neutral_low'] - real_rate) / range_size
            sentiment = 0.5 * position
        elif real_rate <= t['neutral_high']:
            mid_neutral = (t['neutral_low'] + t['neutral_high']) / 2
            max_distance = (t['neutral_high'] - t['neutral_low']) / 2
            distance_from_mid = abs(real_rate - mid_neutral)
            sentiment = 0.1 * (1 - distance_from_mid / max_distance)
        elif real_rate <= t['bearish']:
            range_size = t['bearish'] - t['neutral_high']
            position = (real_rate - t['neutral_high']) / range_size
            sentiment = -0.5 * position
        elif real_rate <= t['strong_bearish']:
            range_size = t['strong_bearish'] - t['bearish']
            position = (real_rate - t['bearish']) / range_size
            sentiment = -0.5 - 0.5 * position
        else:
            sentiment = -1.0

        # ENFORCE BOUNDS (safety check)
        sentiment = max(-1.0, min(1.0, sentiment))

        # Calculate confidence
        if real_rate <= t['strong_bullish'] or real_rate >= t['strong_bearish']:
            confidence = 0.95
        elif real_rate <= t['bullish'] or real_rate >= t['bearish']:
            confidence = 0.80
        elif t['neutral_low'] <= real_rate <= t['neutral_high']:
            confidence = 0.40
        else:
            confidence = 0.65

        return sentiment, confidence

    def generate_sentiment_scores(self, data):
        """Generate sentiment scores for entire dataset"""
        print("Generating sentiment scores...")

        sentiment_scores = []
        confidence_scores = []
        sentiment_smoothed = []
        confidence_smoothed = []

        for _, row in data.iterrows():
            # Raw sentiment
            sentiment, confidence = self.calculate_sentiment_score(row['real_rate'])
            sentiment_scores.append(sentiment)
            confidence_scores.append(confidence)

            # Smoothed sentiment (if EMA available)
            if pd.notna(row.get('real_rate_ema')):
                sentiment_smooth, confidence_smooth = self.calculate_sentiment_score(row['real_rate_ema'])
                sentiment_smoothed.append(sentiment_smooth)
                confidence_smoothed.append(confidence_smooth)
            else:
                sentiment_smoothed.append(np.nan)
                confidence_smoothed.append(np.nan)

        data['sentiment_raw'] = sentiment_scores
        data['confidence_raw'] = confidence_scores
        data['sentiment_smoothed'] = sentiment_smoothed
        data['confidence_smoothed'] = confidence_smoothed

        print(f"Generated {len(data)} sentiment observations")
        return data

    def interpret_sentiment(self, score):
        """Convert sentiment score to interpretation"""
        if pd.isna(score):
            return "No Data"
        elif score > 0.7:
            return "Strongly Bullish"
        elif score > 0.3:
            return "Bullish"
        elif score > -0.3:
            return "Neutral"
        elif score > -0.7:
            return "Bearish"
        else:
            return "Strongly Bearish"

    def get_signal_strength(self, confidence):
        """Convert confidence to signal strength"""
        if pd.isna(confidence):
            return "No Signal"
        elif confidence > 0.8:
            return "Strong"
        elif confidence > 0.6:
            return "Medium"
        else:
            return "Weak"

    def run_full_analysis(self, lookback_days=365):
        """Run complete analysis with real FRED data"""
        print(" Starting Real Interest Rate Sentiment Analysis")
        print("=" * 60)

        # Step 1: Fetch real data
        treasury_data = self.fetch_treasury_data(lookback_days)
        inflation_data = self.fetch_inflation_expectations(lookback_days)
        tips_data = self.fetch_tips_breakeven(lookback_days)

        if treasury_data is None or inflation_data is None:
            print(" Failed to fetch required data")
            return None

        # Step 2: Integrate data sources
        integrated_data = self.integrate_data_sources(treasury_data, inflation_data, tips_data)
        if integrated_data is None:
            return None

        # Step 3: Calculate real interest rates
        data_with_rates = self.calculate_real_interest_rate(integrated_data)
        if data_with_rates is None:
            return None

        # Step 4: Apply smoothing
        smoothed_data = self.apply_smoothing(data_with_rates)

        # Step 5: Generate sentiment scores
        final_data = self.generate_sentiment_scores(smoothed_data)

        # Step 6: Add interpretations
        final_data['sentiment_interpretation'] = final_data['sentiment_smoothed'].apply(self.interpret_sentiment)
        final_data['signal_strength'] = final_data['confidence_smoothed'].apply(self.get_signal_strength)

        print("🎉 Analysis complete!")
        return final_data

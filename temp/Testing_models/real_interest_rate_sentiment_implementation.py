
# REAL INTEREST RATE SENTIMENT SCORE - PRODUCTION IMPLEMENTATION
# =================================================================
# This is a production-ready implementation for integrating Real Interest 
# Rate Sentiment Score into a gold momentum sentiment analysis platform

import pandas as pd
import numpy as np
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

class RealRateSentimentKPI:
    """
    Production implementation of Real Interest Rate Sentiment Score
    for Gold Momentum Sentiment Analysis Platform
    """

    def __init__(self, fred_api_key, smoothing_window=10):
        self.fred = Fred(api_key=fred_api_key)
        self.smoothing_window = smoothing_window

        # Calibrated thresholds based on historical analysis
        self.thresholds = {
            'strong_bullish': -1.5,    # Strong gold bullish signal
            'bullish': -0.5,           # Moderate gold bullish signal
            'neutral_low': -0.2,       # Lower neutral bound
            'neutral_high': 0.5,       # Upper neutral bound  
            'bearish': 1.0,            # Moderate gold bearish signal
            'strong_bearish': 3.0      # Strong gold bearish signal
        }

    def fetch_treasury_data(self, lookback_days=365):
        """Fetch 10-Year Treasury data from FRED"""
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=lookback_days)

            treasury_data = self.fred.get_series(
                'DGS10', 
                observation_start=start_date, 
                observation_end=end_date
            )
            return treasury_data.dropna()
        except Exception as e:
            print(f"Error fetching Treasury data: {e}")
            return None

    def fetch_inflation_expectations(self, lookback_days=365):
        """Fetch Michigan Consumer Survey inflation expectations"""
        try:
            end_date = pd.Timestamp.now()  
            start_date = end_date - pd.Timedelta(days=lookback_days)

            # Fetch both 1-year and 5-year expectations
            mich_1y = self.fred.get_series(
                'MICH', 
                observation_start=start_date,
                observation_end=end_date
            )

            # Forward-fill monthly data to daily frequency
            mich_1y_daily = mich_1y.resample('D').ffill()

            return mich_1y_daily.dropna()
        except Exception as e:
            print(f"Error fetching inflation expectations: {e}")
            return None

    def fetch_tips_breakeven(self, lookback_days=365):
        """Fetch TIPS breakeven inflation rate"""
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=lookback_days)

            tips_data = self.fred.get_series(
                'T10YIE',
                observation_start=start_date,
                observation_end=end_date  
            )
            return tips_data.dropna()
        except Exception as e:
            print(f"Error fetching TIPS data: {e}")
            return None

    def calculate_real_rate(self, nominal_rate, inflation_expectation):
        """Calculate real interest rate"""
        return nominal_rate - inflation_expectation

    def apply_smoothing(self, data, method='ema'):
        """Apply smoothing to reduce noise"""
        if method == 'ema':
            return data.ewm(span=self.smoothing_window).mean()
        elif method == 'sma':
            return data.rolling(window=self.smoothing_window).mean()
        else:
            return data

    def calculate_sentiment_score(self, real_rate):
        """
        Calculate sentiment score based on real interest rate
        Returns score (-1 to +1) and confidence (0 to 1)
        """
        t = self.thresholds

        # Base sentiment calculation
        if real_rate <= t['strong_bullish']:
            sentiment = 1.0
        elif real_rate <= t['bullish']:
            # Linear interpolation
            sentiment = 0.5 + 0.5 * (t['bullish'] - real_rate) / (t['bullish'] - t['strong_bullish'])
        elif real_rate <= t['neutral_low']:
            sentiment = 0.5 * (t['neutral_low'] - real_rate) / (t['neutral_low'] - t['bullish'])
        elif real_rate <= t['neutral_high']:
            # Neutral zone
            mid_neutral = (t['neutral_low'] + t['neutral_high']) / 2
            max_distance = (t['neutral_high'] - t['neutral_low']) / 2
            distance_from_mid = abs(real_rate - mid_neutral)
            sentiment = 0.1 * (1 - distance_from_mid / max_distance)
        elif real_rate <= t['bearish']:
            sentiment = -0.5 * (real_rate - t['neutral_high']) / (t['bearish'] - t['neutral_high'])
        elif real_rate <= t['strong_bearish']:
            sentiment = -0.5 - 0.5 * (real_rate - t['bearish']) / (t['strong_bearish'] - t['bearish'])
        else:
            sentiment = -1.0

        # Confidence calculation
        if real_rate <= t['strong_bullish'] or real_rate >= t['strong_bearish']:
            confidence = 0.95
        elif real_rate <= t['bullish'] or real_rate >= t['bearish']:
            confidence = 0.80
        elif t['neutral_low'] <= real_rate <= t['neutral_high']:
            confidence = 0.40
        else:
            confidence = 0.65

        return sentiment, confidence

    def get_current_sentiment(self, method='hybrid'):
        """
        Get current Real Interest Rate Sentiment Score

        Returns:
        dict with sentiment_score, confidence, real_rate, and metadata
        """

        # Fetch data
        treasury = self.fetch_treasury_data()
        inflation = self.fetch_inflation_expectations()

        if treasury is None or inflation is None:
            return {"error": "Failed to fetch required data"}

        # Align data to common dates
        combined = pd.DataFrame({
            'treasury_10y': treasury,
            'inflation_1y': inflation
        }).dropna()

        if len(combined) == 0:
            return {"error": "No overlapping data available"}

        # Calculate real rate
        combined['real_rate'] = self.calculate_real_rate(
            combined['treasury_10y'], 
            combined['inflation_1y']
        )

        # Apply smoothing
        combined['real_rate_smooth'] = self.apply_smoothing(combined['real_rate'])

        # Get latest values
        latest_raw = combined['real_rate'].iloc[-1]
        latest_smooth = combined['real_rate_smooth'].iloc[-1]

        # Calculate sentiment
        sentiment_raw, conf_raw = self.calculate_sentiment_score(latest_raw)
        sentiment_smooth, conf_smooth = self.calculate_sentiment_score(latest_smooth)

        return {
            'sentiment_score': sentiment_smooth,
            'confidence': conf_smooth,
            'real_rate_raw': latest_raw,
            'real_rate_smoothed': latest_smooth,
            'treasury_10y': combined['treasury_10y'].iloc[-1],
            'inflation_expectation': combined['inflation_1y'].iloc[-1],
            'data_date': combined.index[-1],
            'sentiment_interpretation': self._interpret_sentiment(sentiment_smooth),
            'signal_strength': self._interpret_confidence(conf_smooth)
        }

    def _interpret_sentiment(self, score):
        """Convert sentiment score to human-readable interpretation"""
        if score > 0.7:
            return "Strongly Bullish for Gold"
        elif score > 0.3:
            return "Bullish for Gold"
        elif score > -0.3:
            return "Neutral for Gold"
        elif score > -0.7:
            return "Bearish for Gold"
        else:
            return "Strongly Bearish for Gold"

    def _interpret_confidence(self, confidence):
        """Convert confidence to signal strength"""
        if confidence > 0.8:
            return "Strong Signal"
        elif confidence > 0.6:
            return "Medium Signal"
        else:
            return "Weak Signal"

# USAGE EXAMPLE:
# ==============
# 
# # Initialize with your FRED API key
# real_rate_kpi = RealRateSentimentKPI(fred_api_key='your_fred_api_key_here')
#
# # Get current sentiment score  
# current_sentiment = real_rate_kpi.get_current_sentiment()
# print(f"Current Real Rate Sentiment: {current_sentiment['sentiment_score']:.3f}")
# print(f"Interpretation: {current_sentiment['sentiment_interpretation']}")
# print(f"Confidence: {current_sentiment['signal_strength']}")
#
# # Integration into composite sentiment framework:
# composite_weights = {
#     'technical_analysis': 0.20,
#     'usd_index': 0.15,
#     'inflation_expectations': 0.10,
#     'fed_rate_expectations': 0.15,
#     'real_interest_rate': 0.12,  # This KPI
#     'etf_flows': 0.10,
#     'central_bank_buying': 0.08,
#     'geopolitical_risk': 0.10
# }


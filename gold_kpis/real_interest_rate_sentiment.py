
import pandas as pd
import numpy as np
from fredapi import Fred
import warnings
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

class RealRateSentimentScorer:
    """
    Optimized Real Interest Rate Sentiment Score with:
    - Calibrated thresholds based on -0.82 gold correlation
    - Multi-timeframe analysis (5, 10, 20-day)
    - Regime detection and dynamic adjustments
    - Enhanced confidence scoring
    """

    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get('FRED_API_KEY')
            if api_key is None:
                raise ValueError("FRED API key required!")

        self.fred = Fred(api_key=api_key)

        # OPTIMIZED THRESHOLDS (based on -0.82 correlation research)
        self.thresholds = {
            'strong_bullish': -1.0,    # More responsive (was -1.5)
            'bullish': -0.3,           # More responsive (was -0.5)  
            'neutral_low': -0.1,       # Narrower neutral (was -0.2)
            'neutral_high': 0.3,       # Narrower neutral (was 0.5)
            'bearish': 0.8,            # More responsive (was 1.0)
            'strong_bearish': 2.0      # More responsive (was 3.0)
        }

        # Multi-timeframe smoothing windows
        self.smoothing_windows = {
            'short': 5,    # Short-term signals
            'medium': 10,  # Medium-term (original)
            'long': 20     # Long-term trend
        }

    def fetch_data_sources(self, lookback_days=730):
        """Enhanced data fetching with better error handling"""
        print("Fetching data sources...")

        data_sources = {}

        try:
            # Treasury data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            treasury_series = self.fred.get_series('DGS10', 
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d'))
            data_sources['treasury'] = treasury_series.reset_index()
            data_sources['treasury'].columns = ['date', 'treasury_10y']

            # Michigan inflation expectations
            mich_series = self.fred.get_series('MICH',
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d'))
            data_sources['michigan'] = mich_series.reset_index()
            data_sources['michigan'].columns = ['date', 'inflation_expectation_1y']

            # TIPS breakeven (for validation)
            tips_series = self.fred.get_series('T10YIE',
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d'))
            data_sources['tips'] = tips_series.reset_index()
            data_sources['tips'].columns = ['date', 'tips_breakeven']

            try:
                mich5y_series = self.fred.get_series('MICH5Y',
                    observation_start=start_date.strftime('%Y-%m-%d'),
                    observation_end=end_date.strftime('%Y-%m-%d'))
                data_sources['michigan_5y'] = mich5y_series.reset_index()
                data_sources['michigan_5y'].columns = ['date', 'inflation_expectation_5y']
            except:
                print("⚠5Y Michigan data not available, using 1Y only")
                data_sources['michigan_5y'] = None

            print(f"Fetched all data sources successfully")
            return data_sources

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def integrate_and_prepare_data(self, data_sources):
        """Enhanced data integration with regime detection setup"""
        print("Integrating data with regime analysis...")

        # Start with daily Treasury data
        df = data_sources['treasury'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Add Michigan 1Y expectations (forward-filled daily)
        mich_df = data_sources['michigan'].copy()
        mich_df['date'] = pd.to_datetime(mich_df['date'])
        mich_df = mich_df.set_index('date')
        mich_daily = mich_df.resample('D').ffill()
        df = df.join(mich_daily, how='left')

        # Add TIPS data
        if data_sources['tips'] is not None:
            tips_df = data_sources['tips'].copy()
            tips_df['date'] = pd.to_datetime(tips_df['date'])
            tips_df = tips_df.set_index('date')
            df = df.join(tips_df, how='left')

        # Add 5Y expectations if available
        if data_sources['michigan_5y'] is not None:
            mich5y_df = data_sources['michigan_5y'].copy()
            mich5y_df['date'] = pd.to_datetime(mich5y_df['date'])
            mich5y_df = mich5y_df.set_index('date')
            mich5y_daily = mich5y_df.resample('D').ffill()
            df = df.join(mich5y_daily, how='left')

        # Forward fill and clean
        df = df.fillna(method='ffill')
        df = df.dropna(subset=['treasury_10y', 'inflation_expectation_1y'])
        df = df.reset_index()

        print(f"Integrated dataset: {len(df)} observations")
        return df

    def calculate_real_rates_enhanced(self, df):
        """Calculate real rates with multiple methodologies"""
        print("Calculating enhanced real interest rates...")

        # Primary real rate (survey-based)
        df['real_rate'] = df['treasury_10y'] - df['inflation_expectation_1y']

        # Market-based real rate (if TIPS available)
        if 'tips_breakeven' in df.columns:
            df['real_rate_market'] = df['treasury_10y'] - df['tips_breakeven']
            # Hybrid rate (70% survey, 30% market)
            df['real_rate_hybrid'] = 0.7 * df['real_rate'] + 0.3 * df['real_rate_market']
        else:
            df['real_rate_hybrid'] = df['real_rate']

        # Calculate volatility for regime detection
        df['real_rate_volatility'] = df['real_rate'].rolling(window=30).std()
        df['real_rate_momentum'] = df['real_rate'].diff(5)  # 5-day change

        current_real_rate = df['real_rate'].iloc[-1]
        print(f"Current real rate: {current_real_rate:.2f}%")

        return df

    def apply_multi_timeframe_smoothing(self, df):
        """Apply multiple timeframe smoothing for better signal quality"""
        print("Applying multi-timeframe smoothing...")

        for name, window in self.smoothing_windows.items():
            # Exponential Moving Averages
            df[f'real_rate_ema_{name}'] = df['real_rate'].ewm(span=window).mean()
            # Simple Moving Averages for comparison
            df[f'real_rate_sma_{name}'] = df['real_rate'].rolling(window=window).mean()

        print("Multi-timeframe smoothing applied")
        return df

    def detect_market_regime(self, df):
        """Detect current market regime for dynamic threshold adjustment"""

        # Simple regime detection based on volatility and trends
        recent_vol = df['real_rate_volatility'].tail(30).mean()
        recent_trend = df['real_rate'].tail(20).mean() - df['real_rate'].tail(60).mean()

        vol_threshold = df['real_rate_volatility'].quantile(0.7)

        if recent_vol > vol_threshold:
            regime = 'high_volatility'
            vol_adjustment = 1.2  # Amplify signals in volatile periods
        elif abs(recent_trend) > 0.5:
            regime = 'trending'
            vol_adjustment = 1.1  # Slight amplification in trending markets
        else:
            regime = 'normal'
            vol_adjustment = 1.0  # No adjustment

        return regime, vol_adjustment

    def calculate_optimized_sentiment(self, real_rate, regime_adjustment=1.0):
        """Enhanced sentiment calculation with regime adjustments"""

        if pd.isna(real_rate):
            return 0.0, 0.0

        t = self.thresholds

        # Base sentiment calculation with optimized thresholds
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
            # Narrower neutral zone
            mid_neutral = (t['neutral_low'] + t['neutral_high']) / 2
            max_distance = (t['neutral_high'] - t['neutral_low']) / 2
            distance_from_mid = abs(real_rate - mid_neutral)
            sentiment = 0.05 * (1 - distance_from_mid / max_distance)  # Smaller neutral bias
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

        # Apply regime adjustment
        sentiment = sentiment * regime_adjustment

        # Enforce bounds
        sentiment = max(-1.0, min(1.0, sentiment))

        # Enhanced confidence calculation
        confidence = self.calculate_enhanced_confidence(real_rate)

        return sentiment, confidence

    def calculate_enhanced_confidence(self, real_rate):
        """Enhanced confidence scoring based on distance from thresholds and regime stability"""

        t = self.thresholds

        # Base confidence from threshold distance
        if real_rate <= t['strong_bullish'] or real_rate >= t['strong_bearish']:
            base_confidence = 0.95
        elif real_rate <= t['bullish'] or real_rate >= t['bearish']:
            base_confidence = 0.85  # Higher than before
        elif t['neutral_low'] <= real_rate <= t['neutral_high']:
            base_confidence = 0.35  # Lower for narrow neutral zone
        else:
            base_confidence = 0.75

        return base_confidence

    def generate_multi_timeframe_signals(self, df):
        """Generate signals across multiple timeframes"""
        print("Generating multi-timeframe sentiment signals...")

        # Detect current regime
        regime, regime_adjustment = self.detect_market_regime(df)
        print(f"Market regime: {regime} (adjustment: {regime_adjustment:.1f}x)")

        # Generate signals for each timeframe
        for name, window in self.smoothing_windows.items():
            ema_col = f'real_rate_ema_{name}'

            if ema_col in df.columns:
                sentiment_scores = []
                confidence_scores = []

                for _, row in df.iterrows():
                    if pd.notna(row[ema_col]):
                        sentiment, confidence = self.calculate_optimized_sentiment(
                            row[ema_col], regime_adjustment)
                        sentiment_scores.append(sentiment)
                        confidence_scores.append(confidence)
                    else:
                        sentiment_scores.append(np.nan)
                        confidence_scores.append(np.nan)

                df[f'sentiment_{name}'] = sentiment_scores
                df[f'confidence_{name}'] = confidence_scores

        # Create composite signal (weighted average)
        weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}

        composite_sentiment = []
        composite_confidence = []

        for _, row in df.iterrows():
            weighted_sentiment = 0
            weighted_confidence = 0
            total_weight = 0

            for name, weight in weights.items():
                sentiment_col = f'sentiment_{name}'
                confidence_col = f'confidence_{name}'

                if pd.notna(row[sentiment_col]):
                    weighted_sentiment += row[sentiment_col] * weight
                    weighted_confidence += row[confidence_col] * weight
                    total_weight += weight

            if total_weight > 0:
                composite_sentiment.append(weighted_sentiment / total_weight)
                composite_confidence.append(weighted_confidence / total_weight)
            else:
                composite_sentiment.append(np.nan)
                composite_confidence.append(np.nan)

        df['sentiment_composite'] = composite_sentiment
        df['confidence_composite'] = composite_confidence

        print("Multi-timeframe signals generated")
        return df, regime

    def add_interpretations(self, df):
        """Add human-readable interpretations"""

        def interpret_sentiment(score):
            if pd.isna(score):
                return "No Data"
            elif score > 0.6:
                return "Strongly Bullish"
            elif score > 0.2:
                return "Bullish"
            elif score > -0.2:
                return "Neutral"
            elif score > -0.6:
                return "Bearish"
            else:
                return "Strongly Bearish"

        def signal_strength(confidence):
            if pd.isna(confidence):
                return "No Signal"
            elif confidence > 0.8:
                return "Very Strong"
            elif confidence > 0.6:
                return "Strong"
            elif confidence > 0.4:
                return "Medium"
            else:
                return "Weak"

        df['sentiment_interpretation'] = df['sentiment_composite'].apply(interpret_sentiment)
        df['signal_strength'] = df['confidence_composite'].apply(signal_strength)

        return df

    def run_optimized_analysis(self, lookback_days=730):
        """Run the complete optimized analysis"""
        # Fetch enhanced data
        data_sources = self.fetch_data_sources(lookback_days)
        if data_sources is None:
            return None

        # Integrate and prepare
        df = self.integrate_and_prepare_data(data_sources)

        # Calculate enhanced real rates
        df = self.calculate_real_rates_enhanced(df)

        # Apply multi-timeframe smoothing
        df = self.apply_multi_timeframe_smoothing(df)

        # Generate multi-timeframe signals
        df, regime = self.generate_multi_timeframe_signals(df)

        # Add interpretations
        df = self.add_interpretations(df)

        return df, regime

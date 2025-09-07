
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GoldMomentumSentiment:
    def __init__(self):
        self.gold_tickers = ["GC=F", "GLD", "XAUUSD=X", "IAU"]
        self.primary_ticker = "GC=F"

    def fetch_gold_data(self, days=60):
        """Fetch gold data with reasonable timeframe"""
        for ticker in self.gold_tickers:
            try:
                print(f"Fetching {days} days from {ticker}...")

                end_date = datetime.now()
                start_date = end_date - timedelta(days=days + 15)

                gold_data = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    progress=False
                )

                if not gold_data.empty:
                    if isinstance(gold_data.columns, pd.MultiIndex):
                        gold_data.columns = gold_data.columns.droplevel(1)

                    gold_data = gold_data.dropna()

                    # Ensure required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_columns:
                        if col not in gold_data.columns:
                            if col == 'Volume':
                                gold_data[col] = 0
                            else:
                                gold_data[col] = gold_data['Close']

                    gold_data = gold_data.tail(days)

                    if len(gold_data) >= 20:  # Minimum requirement
                        print(f"✅ Got {len(gold_data)} days from {ticker}")
                        self.data_source = ticker
                        return gold_data

            except Exception as e:
                print(f"Failed {ticker}: {e}")
                continue

        raise ValueError("Could not fetch gold data")

    def calculate_realistic_indicators(self, df):

        indicators = {}
        data_length = len(df)

        print(f"Calculating realistic indicators for {data_length} days...")

        # 1. Rate of Change - Multiple timeframes with REALISTIC thresholds
        for period in [7, 14, 21]:
            if data_length > period:
                roc = ((df['Close'].iloc[-1] - df['Close'].iloc[-period-1]) / df['Close'].iloc[-period-1] * 100)
                indicators[f'roc_{period}'] = roc

        # 2. MACD with actual values (not just signals)
        if data_length >= 26:
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()

            indicators['macd_line'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]

        # 3. RSI - Single, reliable calculation
        if data_length >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi_14'] = rsi.iloc[-1]

        # 4. Moving Averages with trend analysis
        for period in [20, 50]:
            if data_length >= period:
                ma = df['Close'].rolling(window=period).mean()
                indicators[f'ma_{period}'] = ma.iloc[-1]

                # MA slope (trend strength)
                if len(ma) >= 5:
                    ma_slope = (ma.iloc[-1] - ma.iloc[-5]) / ma.iloc[-5] * 100
                    indicators[f'ma_{period}_slope'] = ma_slope

        # 5. Bollinger Bands
        if data_length >= 20:
            bb_ma = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            bb_upper = bb_ma + (bb_std * 2)
            bb_lower = bb_ma - (bb_std * 2)

            # Bollinger %B (position within bands)
            bb_percent_b = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            indicators['bb_percent_b'] = bb_percent_b.iloc[-1]

        # 6. Volatility (essential for context)
        if data_length >= 10:
            returns = df['Close'].pct_change().dropna()
            indicators['volatility'] = returns.std() * np.sqrt(252)

        # 7. Volume trend (if available)
        if data_length >= 10 and df['Volume'].sum() > 0:
            vol_ma_short = df['Volume'].rolling(window=5).mean()
            vol_ma_long = df['Volume'].rolling(window=20).mean()
            if vol_ma_long.iloc[-1] > 0:
                indicators['volume_ratio'] = vol_ma_short.iloc[-1] / vol_ma_long.iloc[-1]

        # 8. Price momentum (simple but effective)
        if data_length >= 10:
            momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
            indicators['momentum_10d'] = momentum

        return indicators

    def realistic_scoring(self, indicators):

        scores = {}

        # 1. ROC Scoring - CONSERVATIVE thresholds
        roc_signals = []
        for period in [7, 14, 21]:
            roc_key = f'roc_{period}'
            if roc_key in indicators:
                roc_val = indicators[roc_key]
                # REALISTIC thresholds: 4% for strong, 2% for moderate
                if roc_val > 4.0:
                    roc_signals.append(0.8)  # Strong bullish (not perfect 1.0)
                elif roc_val > 2.0:
                    roc_signals.append(0.4)  # Moderate bullish
                elif roc_val < -4.0:
                    roc_signals.append(-0.8)  # Strong bearish
                elif roc_val < -2.0:
                    roc_signals.append(-0.4)  # Moderate bearish
                else:
                    roc_signals.append(0.0)  # Neutral

        scores['roc'] = np.mean(roc_signals) if roc_signals else 0.0

        # 2. MACD Scoring - Based on actual values, not just crossover
        if 'macd_line' in indicators and 'macd_signal' in indicators:
            macd_line = indicators['macd_line']
            macd_signal = indicators['macd_signal']
            macd_hist = indicators.get('macd_histogram', 0)

            # Calculate MACD score based on multiple factors
            macd_score = 0.0

            # Basic crossover (weighted 50%)
            if macd_line > macd_signal:
                macd_score += 0.3
            elif macd_line < macd_signal:
                macd_score -= 0.3

            # Histogram direction (weighted 30%)
            if macd_hist > 0:
                macd_score += 0.2
            elif macd_hist < 0:
                macd_score -= 0.2

            # Magnitude matters (weighted 20%)
            if abs(macd_hist) > abs(macd_signal) * 0.1:  # Significant histogram
                macd_score += 0.1 if macd_hist > 0 else -0.1

            scores['macd'] = max(-0.8, min(0.8, macd_score))  # Cap at 80%
        else:
            scores['macd'] = 0.0

        # 3. RSI Scoring - REALISTIC overbought/oversold levels
        if 'rsi_14' in indicators:
            rsi = indicators['rsi_14']

            if rsi >= 75:  # Very overbought
                scores['rsi'] = -0.3  # Negative because potential reversal
            elif rsi >= 65:  # Overbought but momentum
                scores['rsi'] = 0.2   # Slightly positive
            elif rsi >= 55:  # Bullish zone
                scores['rsi'] = 0.5
            elif rsi <= 25:  # Very oversold
                scores['rsi'] = 0.3   # Positive because potential bounce
            elif rsi <= 35:  # Oversold but momentum down
                scores['rsi'] = -0.2
            elif rsi <= 45:  # Bearish zone
                scores['rsi'] = -0.5
            else:  # Neutral zone (45-55)
                scores['rsi'] = 0.0
        else:
            scores['rsi'] = 0.0

        # 4. Moving Average Scoring - Trend and slope
        ma_scores = []
        for period in [20, 50]:
            ma_key = f'ma_{period}'
            slope_key = f'ma_{period}_slope'

            if ma_key in indicators:
                current_price = indicators.get('current_price', 0)
                ma_value = indicators[ma_key]

                # Position relative to MA
                if current_price > ma_value * 1.02:  # 2% above MA
                    ma_score = 0.4
                elif current_price > ma_value:  # Above MA but close
                    ma_score = 0.2
                elif current_price < ma_value * 0.98:  # 2% below MA
                    ma_score = -0.4
                elif current_price < ma_value:  # Below MA but close
                    ma_score = -0.2
                else:
                    ma_score = 0.0

                # Add slope component
                if slope_key in indicators:
                    slope = indicators[slope_key]
                    if slope > 1.0:  # Strong upward slope
                        ma_score += 0.2
                    elif slope > 0.3:  # Moderate upward slope
                        ma_score += 0.1
                    elif slope < -1.0:  # Strong downward slope
                        ma_score -= 0.2
                    elif slope < -0.3:  # Moderate downward slope
                        ma_score -= 0.1

                ma_scores.append(max(-0.6, min(0.6, ma_score)))

        scores['moving_average'] = np.mean(ma_scores) if ma_scores else 0.0

        # 5. Bollinger Band Scoring - Context matters
        if 'bb_percent_b' in indicators:
            bb_b = indicators['bb_percent_b']
            volatility = indicators.get('volatility', 0.2)

            # In low volatility, use mean reversion
            # In high volatility, use momentum
            if volatility < 0.2:  # Low volatility - mean reversion
                if bb_b > 0.9:
                    scores['bollinger'] = -0.3  # Overbought, expect reversion
                elif bb_b < 0.1:
                    scores['bollinger'] = 0.3   # Oversold, expect bounce
                else:
                    scores['bollinger'] = 0.0
            else:  # High volatility - momentum
                if bb_b > 0.8:
                    scores['bollinger'] = 0.4   # Breakout momentum
                elif bb_b < 0.2:
                    scores['bollinger'] = -0.4  # Breakdown momentum
                else:
                    scores['bollinger'] = 0.0
        else:
            scores['bollinger'] = 0.0

        # 6. Volume Scoring (if available)
        if 'volume_ratio' in indicators:
            vol_ratio = indicators['volume_ratio']
            if vol_ratio > 1.3:  # 30% above average
                scores['volume'] = 0.3
            elif vol_ratio > 1.1:  # 10% above average
                scores['volume'] = 0.1
            elif vol_ratio < 0.7:  # 30% below average
                scores['volume'] = -0.2
            else:
                scores['volume'] = 0.0
        else:
            scores['volume'] = 0.0

        # 7. Momentum Scoring
        if 'momentum_10d' in indicators:
            mom = indicators['momentum_10d']
            if mom > 5.0:
                scores['momentum'] = 0.6
            elif mom > 2.0:
                scores['momentum'] = 0.3
            elif mom < -5.0:
                scores['momentum'] = -0.6
            elif mom < -2.0:
                scores['momentum'] = -0.3
            else:
                scores['momentum'] = 0.0
        else:
            scores['momentum'] = 0.0

        return scores

    def confidence_calculation(self, scores, indicators):
        """Calculate REALISTIC confidence that accounts for uncertainty"""

        # Get non-zero scores
        active_scores = [score for score in scores.values() if abs(score) > 0.05]

        if not active_scores:
            return 0.2  # Very low confidence with no clear signals

        # 1. Signal Agreement
        positive_scores = [s for s in active_scores if s > 0]
        negative_scores = [s for s in active_scores if s < 0]

        total_signals = len(active_scores)
        agreement = max(len(positive_scores), len(negative_scores)) / total_signals

        # 2. Signal Strength (average magnitude)
        avg_strength = np.mean([abs(s) for s in active_scores])

        # 3. Conflicting Signals Penalty
        conflict_penalty = 1.0
        if len(positive_scores) > 0 and len(negative_scores) > 0:
            # Reduce confidence when signals conflict
            conflict_ratio = min(len(positive_scores), len(negative_scores)) / total_signals
            conflict_penalty = 1.0 - (conflict_ratio * 0.5)

        # 4. Market Uncertainty Factors
        uncertainty_penalty = 1.0

        # High volatility reduces confidence
        volatility = indicators.get('volatility', 0.2)
        if volatility > 0.25:
            uncertainty_penalty *= 0.85
        elif volatility > 0.35:
            uncertainty_penalty *= 0.7

        # Very low volatility also reduces confidence (lack of conviction)
        if volatility < 0.10:
            uncertainty_penalty *= 0.9

        # 5. Data Quality Factor
        data_quality = min(1.0, len(indicators) / 10)  # More indicators = better

        # 6. Final Confidence Calculation
        base_confidence = agreement * avg_strength * data_quality
        final_confidence = base_confidence * conflict_penalty * uncertainty_penalty

        # 7. Realistic Caps
        # Never exceed 90% confidence (markets are uncertain)
        # Never go below 10% confidence (some signal is better than none)
        final_confidence = max(0.1, min(0.9, final_confidence))

        return final_confidence

    def calculate_sentiment(self, scores, indicators):
        """Calculate weighted sentiment with REALISTIC weights and thresholds"""

        weights = {
            'roc': 0.25,
            'macd': 0.20,
            'rsi': 0.15,
            'moving_average': 0.20,
            'bollinger': 0.10,
            'volume': 0.05,
            'momentum': 0.05
        }

        # Calculate weighted score
        weighted_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())


        confidence = self.confidence_calculation(scores, indicators)

        # REALISTIC Classification Thresholds
        # Require higher thresholds AND minimum confidence
        if weighted_score >= 0.35 and confidence >= 0.6:
            return 1, "BULLISH", weighted_score, confidence
        elif weighted_score >= 0.20 and confidence >= 0.5:
            return 1, "WEAK BULLISH", weighted_score, confidence
        elif weighted_score <= -0.35 and confidence >= 0.6:
            return -1, "BEARISH", weighted_score, confidence
        elif weighted_score <= -0.20 and confidence >= 0.5:
            return -1, "WEAK BEARISH", weighted_score, confidence
        else:
            return 0, "NEUTRAL", weighted_score, confidence

    def analyze_sentiment(self, days=60):

        try:
            print("🎯 Starting Gold Sentiment Analysis...")

            # Fetch data
            df = self.fetch_gold_data(days=days)

            if df is None or len(df) < 20:
                return {"error": "Insufficient data for analysis"}

            print(f"📊 Analyzing {len(df)} days ...")

            # Add current price to indicators
            current_price = df['Close'].iloc[-1]

            # Calculate indicators
            indicators = self.calculate_realistic_indicators(df)
            indicators['current_price'] = current_price

            # Calculate scores
            scores = self.realistic_scoring(indicators)


            sentiment, label, weighted_score, confidence = self.calculate_sentiment(scores, indicators)

            # Market context
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)

            # Support/Resistance
            recent_data = df.tail(30)
            support = recent_data['Low'].quantile(0.15)
            resistance = recent_data['High'].quantile(0.85)

            return {
                "sentiment_score": sentiment,
                "sentiment_label": label,
                "weighted_score": round(weighted_score, 3),
                "confidence": round(confidence, 3),
                "current_price": round(current_price, 2),
                "price_change_period": round(price_change, 2),
                "support_level": round(support, 2),
                "resistance_level": round(resistance, 2),
                "key_metrics": {
                    "rsi_14": round(indicators.get('rsi_14', 50), 1),
                    "volatility": f"{indicators.get('volatility', 0.2):.1%}",
                    "momentum_10d": round(indicators.get('momentum_10d', 0), 2),
                    "bb_percent_b": round(indicators.get('bb_percent_b', 0.5), 3)
                },
                "component_scores": {k: round(v, 3) for k, v in scores.items()},
                "data_points": len(df),
                "data_source": getattr(self, 'data_source', 'Unknown'),
                "analysis_period": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                "timestamp": datetime.now().isoformat(),
                "note": "This analysis uses realistic thresholds and acknowledges market uncertainty"
            }

        except Exception as e:
            return {
                "error": f"analysis failed: {str(e)}",
                "sentiment_score": 0,
                "sentiment_label": "ERROR",
                "weighted_score": 0.0,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }

def main():
    print(" Gold Sentiment Analyzer")
    print(" Designed for realistic, truthful market analysis")

    analyzer = GoldMomentumSentiment()
    result = analyzer.analyze_sentiment(days=60)

    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return result


    sentiment_emoji = "🟢" if result['sentiment_score'] == 1 else "🔴" if result['sentiment_score'] == -1 else "🟡"

    print(f"\nSentiment: {sentiment_emoji} {result.get('sentiment_label', 'N/A')}")
    print(f"Score: {result.get('weighted_score', 'N/A')} (Range: -1.0 to +1.0)")
    print(f"Confidence: {result.get('confidence', 0)*100:.1f}% (Max 90% - markets are uncertain!)")
    print(f"Price: ${result.get('current_price', 'N/A')}")
    print(f"Period Change: {result.get('price_change_period', 0):+.2f}%")
    print(f"️Support: ${result.get('support_level', 'N/A')} | Resistance: ${result.get('resistance_level', 'N/A')}")

    # Key metrics
    if 'key_metrics' in result:
        km = result['key_metrics']
        print(f"\n Key Metrics:")
        print(f"   • RSI (14): {km.get('rsi_14', 'N/A')}")
        print(f"   • Volatility: {km.get('volatility', 'N/A')}")
        print(f"   • 10d Momentum: {km.get('momentum_10d', 'N/A')}%")
        print(f"   • Bollinger %B: {km.get('bb_percent_b', 'N/A')}")

    # Component scores (should be realistic)
    if 'component_scores' in result:
        print(f"\n Component Scores (Realistic Range):")
        scores = result['component_scores']
        for component, score in scores.items():
            emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "🟡"
            print(f"   {emoji} {component.replace('_', ' ').title()}: {score:+.3f}")

    print(f"\n Period: {result.get('analysis_period', 'N/A')}")
    print(f" Data Points: {result.get('data_points', 'N/A')} days")
    print(f"\n {result.get('note', '')}")
    print("="*70)

    return result

if __name__ == "__main__":
    main()

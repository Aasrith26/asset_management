import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GoldMomentumAnalyzer:
    def __init__(self):
        self.tickers = ["GC=F", "GLD", "XAUUSD=X"]
        self.primary_ticker = "GC=F"

    def fetch_price_data(self, days=60):
        """Fetch gold price data from Yahoo Finance"""
        for ticker in self.tickers:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days + 15)

                data = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    progress=False
                )

                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)

                    data = data.dropna().tail(days)

                    if len(data) >= 20:
                        print(f"Data fetched: {len(data)} days from {ticker}")
                        self.data_source = ticker
                        return data

            except Exception as e:
                continue

        raise ValueError("Unable to fetch gold price data")

    def calculate_technical_indicators(self, df):
        """Calculate standard technical indicators"""
        indicators = {}

        # Rate of Change (14-day)
        if len(df) >= 15:
            indicators['roc_14'] = ((df['Close'].iloc[-1] - df['Close'].iloc[-15]) / df['Close'].iloc[-15]) * 100

        # MACD (12, 26, 9)
        if len(df) >= 26:
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()

            indicators['macd_value'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_crossover'] = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1

        # RSI (14-day)
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))

        # Moving Averages
        for period in [20, 50]:
            if len(df) >= period:
                ma = df['Close'].rolling(window=period).mean()
                indicators[f'ma_{period}'] = ma.iloc[-1]
                indicators[f'price_vs_ma_{period}'] = (df['Close'].iloc[-1] / ma.iloc[-1] - 1) * 100

        # Volatility (20-day)
        if len(df) >= 20:
            returns = df['Close'].pct_change().dropna()
            indicators['volatility'] = returns.tail(20).std() * np.sqrt(252)

        # Price momentum
        if len(df) >= 10:
            indicators['momentum_10d'] = ((df['Close'].iloc[-1] / df['Close'].iloc[-10]) - 1) * 100

        return indicators

    def score_indicators(self, indicators):
        """Score each indicator based on standard technical analysis rules"""
        scores = {}

        # ROC scoring
        roc = indicators.get('roc_14', 0)
        if roc > 3:
            scores['roc'] = 1
        elif roc > 1:
            scores['roc'] = 0.5
        elif roc < -3:
            scores['roc'] = -1
        elif roc < -1:
            scores['roc'] = -0.5
        else:
            scores['roc'] = 0

        # MACD scoring
        if 'macd_crossover' in indicators:
            macd_cross = indicators['macd_crossover']
            macd_strength = abs(indicators['macd_value']) / (abs(indicators['macd_signal']) + 1e-10)

            if macd_cross == 1:
                scores['macd'] = min(1.0, 0.5 + (macd_strength - 1) * 0.5)
            else:
                scores['macd'] = max(-1.0, -0.5 - (macd_strength - 1) * 0.5)
        else:
            scores['macd'] = 0

        # RSI scoring
        rsi = indicators.get('rsi', 50)
        if rsi >= 70:
            scores['rsi'] = -0.5  # Overbought
        elif rsi >= 60:
            scores['rsi'] = 0.5   # Strong but not overbought
        elif rsi <= 30:
            scores['rsi'] = 0.5   # Oversold (bullish)
        elif rsi <= 40:
            scores['rsi'] = -0.5  # Weak
        else:
            scores['rsi'] = 0     # Neutral

        ma_scores = []
        for period in [20, 50]:
            price_vs_ma = indicators.get(f'price_vs_ma_{period}', 0)
            if price_vs_ma > 2:
                ma_scores.append(1)
            elif price_vs_ma > 0:
                ma_scores.append(0.5)
            elif price_vs_ma < -2:
                ma_scores.append(-1)
            elif price_vs_ma < 0:
                ma_scores.append(-0.5)
            else:
                ma_scores.append(0)

        scores['moving_average'] = np.mean(ma_scores) if ma_scores else 0

        # Momentum scoring
        momentum = indicators.get('momentum_10d', 0)
        if momentum > 5:
            scores['momentum'] = 1
        elif momentum > 2:
            scores['momentum'] = 0.5
        elif momentum < -5:
            scores['momentum'] = -1
        elif momentum < -2:
            scores['momentum'] = -0.5
        else:
            scores['momentum'] = 0

        return scores

    def calculate_composite_sentiment(self, scores):
        """Calculate weighted composite sentiment"""
        weights = {
            'roc': 0.25,
            'macd': 0.25,
            'rsi': 0.20,
            'moving_average': 0.20,
            'momentum': 0.10
        }

        weighted_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())

        # Calculate confidence based on signal consistency
        active_signals = [score for score in scores.values() if abs(score) > 0.1]

        if not active_signals:
            confidence = 0.3
        else:
            # Measure signal agreement
            positive_signals = len([s for s in active_signals if s > 0])
            negative_signals = len([s for s in active_signals if s < 0])
            total_signals = len(active_signals)

            agreement_ratio = max(positive_signals, negative_signals) / total_signals
            signal_strength = np.mean([abs(s) for s in active_signals])

            confidence = agreement_ratio * signal_strength
            confidence = min(0.95, max(0.3, confidence))

        # Sentiment classification
        if weighted_score >= 0.3:
            sentiment = 1
            label = "BULLISH"
        elif weighted_score <= -0.3:
            sentiment = -1
            label = "BEARISH"
        else:
            sentiment = 0
            label = "NEUTRAL"

        return sentiment, label, weighted_score, confidence

    def analyze_momentum(self, days=60):
        """Main analysis function"""
        try:
            # Fetch data
            df = self.fetch_price_data(days)

            if df is None or len(df) < 20:
                return {"error": "Insufficient price data"}

            # Calculate indicators
            indicators = self.calculate_technical_indicators(df)

            # Score indicators
            scores = self.score_indicators(indicators)

            # Calculate sentiment
            sentiment, label, weighted_score, confidence = self.calculate_composite_sentiment(scores)

            # Calculate additional metrics
            current_price = df['Close'].iloc[-1]
            period_change = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100

            # Support/Resistance levels
            recent_data = df.tail(30)
            support = recent_data['Low'].quantile(0.2)
            resistance = recent_data['High'].quantile(0.8)

            return {
                "sentiment": sentiment,
                "sentiment_label": label,
                "confidence": round(confidence, 3),
                "weighted_score": round(weighted_score, 3),
                "current_price": round(current_price, 2),
                "period_change_pct": round(period_change, 2),
                "support_level": round(support, 2),
                "resistance_level": round(resistance, 2),
                "indicators": {
                    "rsi": round(indicators.get('rsi', 50), 1),
                    "roc_14d": round(indicators.get('roc_14', 0), 2),
                    "momentum_10d": round(indicators.get('momentum_10d', 0), 2),
                    "volatility": round(indicators.get('volatility', 0.2), 3),
                    "price_vs_ma20": round(indicators.get('price_vs_ma_20', 0), 2),
                    "macd_signal": indicators.get('macd_crossover', 0)
                },
                "component_scores": scores,
                "data_points": len(df),
                "data_source": getattr(self, 'data_source', 'Unknown'),
                "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "sentiment": 0,
                "confidence": 0.0,
                "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

def main():
    analyzer = GoldMomentumAnalyzer()
    result = analyzer.analyze_momentum()

    if 'error' in result:
        print(f"Error: {result['error']}")
        return result

    # Main results
    print(f"\nSentiment: {result['sentiment_label']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Score: {result['weighted_score']} (Range: -1.0 to +1.0)")
    print(f"Current Price: ${result['current_price']}")
    print(f"Period Change: {result['period_change_pct']:+.2f}%")
    print(f"Support: ${result['support_level']} | Resistance: ${result['resistance_level']}")

    # Key indicators
    ind = result['indicators']
    print(f"\nKey Indicators:")
    print(f"  RSI (14): {ind['rsi']}")
    print(f"  ROC (14d): {ind['roc_14d']:.2f}%")
    print(f"  10d Momentum: {ind['momentum_10d']:.2f}%")
    print(f"  Price vs MA20: {ind['price_vs_ma20']:+.2f}%")
    print(f"  MACD Signal: {'Bullish' if ind['macd_signal'] == 1 else 'Bearish'}")
    print(f"  Volatility: {ind['volatility']:.1%}")

    # Component breakdown
    print(f"\nComponent Scores:")
    for component, score in result['component_scores'].items():
        print(f"  {component.replace('_', ' ').title()}: {score:+.2f}")

    print(f"\nAnalysis Period: {result['data_points']} days")
    print(f"Data Source: {result['data_source']}")
    print(f"Analysis Time: {result['analysis_date']}")
    print("="*60)

    return result

if __name__ == "__main__":
    main()

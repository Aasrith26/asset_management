
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GoldMomentumSentiment:
    def __init__(self):
        # Gold tickers available in yfinance
        self.gold_tickers = ["GC=F", "XAUUSD=X", "GLD"]  # Futures, Forex, ETF
        self.primary_ticker = "GC=F"  # Gold futures (most direct)

    def fetch_gold_data(self, days=30, ticker=None):
        """Fetch gold price data from Yahoo Finance using yfinance"""
        if ticker is None:
            ticker = self.primary_ticker

        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)  # Extra buffer for weekends/holidays

            # Fetch data using yfinance (removed unsupported parameters)
            gold_data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                progress=False
            )

            if gold_data.empty:
                print(f"No data found for {ticker}, trying alternative...")
                return self._try_alternative_tickers(days)

            # Clean and prepare data
            gold_data = gold_data.dropna()

            # Handle multi-level columns (sometimes yfinance returns multi-level columns)
            if isinstance(gold_data.columns, pd.MultiIndex):
                gold_data.columns = gold_data.columns.droplevel(1)

            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in gold_data.columns:
                    if col == 'Volume':
                        gold_data[col] = 0  # Set volume to 0 if not available
                    else:
                        # For price columns, use Close price as fallback
                        gold_data[col] = gold_data.get('Close', gold_data.iloc[:, 0])

            # Get the most recent data
            gold_data = gold_data.tail(days)

            if len(gold_data) < 2:
                raise ValueError(f"Insufficient data: only {len(gold_data)} days available")

            print(f"Successfully fetched {len(gold_data)} days of {ticker} data")
            return gold_data

        except Exception as e:
            print(f"Error fetching {ticker} data: {e}")
            return self._try_alternative_tickers(days)

    def _try_alternative_tickers(self, days):
        """Try alternative gold tickers if primary fails"""
        for ticker in self.gold_tickers[1:]:  # Skip primary ticker
            try:
                print(f"Trying alternative ticker: {ticker}")

                end_date = datetime.now()
                start_date = end_date - timedelta(days=days + 10)

                gold_data = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    progress=False
                )

                if not gold_data.empty:
                    # Handle multi-level columns
                    if isinstance(gold_data.columns, pd.MultiIndex):
                        gold_data.columns = gold_data.columns.droplevel(1)

                    gold_data = gold_data.dropna().tail(days)
                    if len(gold_data) >= 2:
                        print(f"Successfully fetched {len(gold_data)} days from {ticker}")
                        return gold_data

            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
                continue

        # Final fallback - try using Ticker object directly
        return self._try_ticker_object_fallback(days)

    def _try_ticker_object_fallback(self, days):
        """Final fallback using yfinance Ticker object"""
        for ticker_symbol in self.gold_tickers:
            try:
                print(f"Trying Ticker object for: {ticker_symbol}")

                ticker = yf.Ticker(ticker_symbol)

                # Try different period approaches
                periods = ['1mo', '3mo', '6mo']

                for period in periods:
                    try:
                        gold_data = ticker.history(period=period)

                        if not gold_data.empty and len(gold_data) >= 2:
                            # Take the last 'days' worth of data
                            gold_data = gold_data.tail(min(days, len(gold_data)))
                            print(f"Fallback success: {len(gold_data)} days from {ticker_symbol}")
                            return gold_data
                    except Exception as e:
                        print(f"Period {period} failed for {ticker_symbol}: {e}")
                        continue

            except Exception as e:
                print(f"Ticker object failed for {ticker_symbol}: {e}")
                continue

        raise ValueError("All methods failed to fetch gold data. Please check your internet connection and yfinance installation.")

    def get_current_gold_price(self):
        """Get the most recent gold price"""
        for ticker_symbol in self.gold_tickers:
            try:
                gold_ticker = yf.Ticker(ticker_symbol)
                current_data = gold_ticker.history(period="1d")

                if not current_data.empty:
                    return current_data['Close'].iloc[-1]

            except Exception as e:
                print(f"Error fetching current price from {ticker_symbol}: {e}")
                continue

        return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators with proper handling for various data sizes"""
        indicators = {}
        data_length = len(df)

        print(f"Calculating indicators for {data_length} data points")

        # 1. Rate of Change (adaptive period)
        roc_period = min(14, max(2, data_length - 1))
        if data_length >= 2:
            start_price = df['Close'].iloc[-roc_period-1] if data_length > roc_period else df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            indicators['roc_14'] = ((end_price - start_price) / start_price * 100)
        else:
            indicators['roc_14'] = 0

        # 2. Price Velocity (linear regression slope)
        if data_length >= 3:
            lookback = min(20, data_length)
            x = np.arange(lookback)
            y = df['Close'].tail(lookback).values
            if len(y) == len(x):
                slope = np.polyfit(x, y, 1)[0]
                indicators['price_velocity'] = slope
            else:
                indicators['price_velocity'] = 0
        else:
            indicators['price_velocity'] = 0

        # 3. MACD (adaptive periods)
        if data_length >= 26:
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['macd_signal'] = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
            indicators['macd_value'] = macd_line.iloc[-1]
            indicators['macd_signal_value'] = signal_line.iloc[-1]
        elif data_length >= 10:
            # Use shorter periods for small datasets
            ema_5 = df['Close'].ewm(span=5).mean()
            ema_10 = df['Close'].ewm(span=10).mean()
            indicators['macd_signal'] = 1 if ema_5.iloc[-1] > ema_10.iloc[-1] else -1
            indicators['macd_value'] = ema_5.iloc[-1] - ema_10.iloc[-1]
            indicators['macd_signal_value'] = ema_10.iloc[-1]
        else:
            indicators['macd_signal'] = 0
            indicators['macd_value'] = 0
            indicators['macd_signal_value'] = 0

        # 4. RSI (adaptive period)
        rsi_period = min(14, max(2, data_length - 1))
        if data_length >= 3:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()

            # Avoid division by zero
            rs = gain / loss.replace(0, 1e-10)
            rsi_value = 100 - (100 / (1 + rs.iloc[-1]))
            indicators['rsi_14'] = rsi_value if not np.isnan(rsi_value) else 50
        else:
            indicators['rsi_14'] = 50

        # 5. Moving Average Position (adaptive period)
        ma_period = min(50, max(2, data_length))
        if data_length >= 2:
            ma = df['Close'].rolling(window=ma_period, min_periods=1).mean()
            indicators['ma_position'] = 1 if df['Close'].iloc[-1] > ma.iloc[-1] else -1
            indicators['ma_value'] = ma.iloc[-1]
        else:
            indicators['ma_position'] = 0
            indicators['ma_value'] = df['Close'].iloc[-1] if not df.empty else 0

        # 6. Volume analysis (only if Volume column has meaningful data)
        if data_length >= 5 and 'Volume' in df.columns:
            if df['Volume'].sum() > 0:  # Check if volume data is meaningful
                recent_vol = df['Volume'].tail(5).mean()
                previous_vol = df['Volume'].head(-5).tail(5).mean() if data_length >= 10 else recent_vol
                indicators['volume_trend'] = 1 if recent_vol > previous_vol * 1.1 else -1 if recent_vol < previous_vol * 0.9 else 0
            else:
                indicators['volume_trend'] = 0
        else:
            indicators['volume_trend'] = 0

        # 7. Volatility indicator
        if data_length >= 10:
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            indicators['volatility'] = volatility
        else:
            indicators['volatility'] = 0.2  # Default moderate volatility

        # 8. Bollinger Bands
        if data_length >= 20:
            bb_period = min(20, data_length)
            bb_ma = df['Close'].rolling(window=bb_period).mean()
            bb_std = df['Close'].rolling(window=bb_period).std()
            bb_upper = bb_ma + (bb_std * 2)
            bb_lower = bb_ma - (bb_std * 2)

            current_price = df['Close'].iloc[-1]
            if current_price > bb_upper.iloc[-1]:
                indicators['bb_position'] = 1  # Above upper band
            elif current_price < bb_lower.iloc[-1]:
                indicators['bb_position'] = -1  # Below lower band
            else:
                indicators['bb_position'] = 0  # Between bands
        else:
            indicators['bb_position'] = 0

        return indicators

    def score_momentum_indicators(self, indicators):
        """Enhanced scoring with volatility consideration"""
        scores = {}

        # ROC Scoring (adjusted thresholds based on volatility)
        vol_multiplier = max(0.5, min(2.0, indicators.get('volatility', 0.2) / 0.2))
        roc_threshold = 3 * vol_multiplier

        if indicators['roc_14'] > roc_threshold:
            scores['roc'] = 1
        elif indicators['roc_14'] < -roc_threshold:
            scores['roc'] = -1
        else:
            scores['roc'] = 0

        # Price Velocity Scoring
        velocity = indicators['price_velocity']
        if abs(velocity) > 1:  # Significant slope
            scores['velocity'] = 1 if velocity > 0 else -1
        else:
            scores['velocity'] = 0

        # MACD Scoring
        scores['macd'] = indicators['macd_signal']

        # RSI Scoring (dynamic thresholds)
        rsi = indicators['rsi_14']
        if rsi > 65:
            scores['rsi'] = 0  # Potentially overbought
        elif rsi > 55:
            scores['rsi'] = 1  # Bullish momentum
        elif rsi < 35:
            scores['rsi'] = 0  # Potentially oversold
        elif rsi < 45:
            scores['rsi'] = -1  # Bearish momentum
        else:
            scores['rsi'] = 0  # Neutral zone

        # MA Position Scoring
        scores['ma'] = indicators['ma_position']

        # Volume Trend Scoring
        scores['volume'] = indicators.get('volume_trend', 0)

        # Bollinger Band Scoring
        scores['bb'] = indicators.get('bb_position', 0)

        return scores

    def calculate_weighted_sentiment(self, scores):
        """Enhanced weighted calculation with confidence scoring"""
        weights = {
            'roc': 0.25,        # Rate of change
            'velocity': 0.20,   # Price velocity
            'macd': 0.20,       # MACD signal
            'rsi': 0.15,        # RSI
            'ma': 0.10,         # Moving average
            'volume': 0.05,     # Volume trend
            'bb': 0.05          # Bollinger bands
        }

        weighted_score = sum(scores.get(key, 0) * weights[key] for key in weights)

        # Calculate confidence based on indicator agreement
        non_zero_scores = [score for score in scores.values() if score != 0]
        if non_zero_scores:
            # Confidence based on consistency of signals
            positive_signals = len([s for s in non_zero_scores if s > 0])
            negative_signals = len([s for s in non_zero_scores if s < 0])
            total_signals = len(non_zero_scores)

            # Higher confidence when signals agree
            dominant_signals = max(positive_signals, negative_signals)
            confidence = dominant_signals / total_signals if total_signals > 0 else 0
        else:
            confidence = 0

        # Classification with adjusted thresholds
        if weighted_score >= 0.25:
            return 1, "BULLISH", weighted_score, confidence
        elif weighted_score <= -0.25:
            return -1, "BEARISH", weighted_score, confidence
        else:
            return 0, "NEUTRAL", weighted_score, confidence

    def analyze_momentum_sentiment(self, days=30):
        """Enhanced analysis with detailed output"""
        try:
            # Fetch data
            df = self.fetch_gold_data(days=days)

            if df is None or df.empty:
                return {"error": "Could not fetch gold price data"}

            print(f"Analyzing {len(df)} days of gold price data")

            # Calculate indicators
            indicators = self.calculate_technical_indicators(df)

            # Score indicators
            scores = self.score_momentum_indicators(indicators)

            # Calculate weighted sentiment
            sentiment, label, weighted_score, confidence = self.calculate_weighted_sentiment(scores)

            # Additional analysis
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)

            # Support and resistance levels (simple calculation)
            recent_high = df['High'].tail(20).max()
            recent_low = df['Low'].tail(20).min()

            return {
                "sentiment_score": sentiment,
                "sentiment_label": label,
                "weighted_score": round(weighted_score, 3),
                "confidence": round(confidence, 3),
                "current_price": round(df['Close'].iloc[-1], 2),
                "price_change_period": round(price_change, 2),
                "support_level": round(recent_low, 2),
                "resistance_level": round(recent_high, 2),
                "indicators": {k: round(v, 4) if isinstance(v, (int, float)) else v
                             for k, v in indicators.items()},
                "individual_scores": scores,
                "data_points": len(df),
                "data_source": self.primary_ticker,
                "analysis_period": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "sentiment_score": 0,
                "sentiment_label": "ERROR",
                "weighted_score": 0.0,
                "confidence": 0.0,
                "current_price": 0.0,
                "timestamp": datetime.now().isoformat()
            }

# Usage Example with enhanced output
def main():
    print("Initializing Gold Momentum Sentiment Analyzer using Yahoo Finance...")
    analyzer = GoldMomentumSentiment()

    # Check if yfinance is available and show version
    try:
        import yfinance
        print(f" yfinance library detected (version: {yfinance.__version__})")
    except ImportError:
        print(" yfinance not found. Please install with: pip install yfinance")
        return
    except AttributeError:
        print("yfinance library detected (version info not available)")

    result = analyzer.analyze_momentum_sentiment(days=30)

    print("\n" + "="*70)
    print("           GOLD MOMENTUM SENTIMENT ANALYSIS (Yahoo Finance)")
    print("="*70)

    if 'error' in result and result.get('sentiment_score', 0) == 0:
        print(f"Error: {result['error']}")
        return result

    # Display main results
    sentiment_emoji = "🟢" if result['sentiment_score'] == 1 else "🔴" if result['sentiment_score'] == -1 else "🟡"
    print(f"\n Overall Sentiment: {sentiment_emoji} {result.get('sentiment_label', 'N/A')} ({result.get('sentiment_score', 'N/A')})")
    print(f"  Weighted Score: {result.get('weighted_score', 'N/A')}")
    print(f"Confidence Level: {result.get('confidence', 0)*100:.1f}%")
    print(f" Current Gold Price: ${result.get('current_price', 'N/A')}")
    print(f"Data Source: {result.get('data_source', 'N/A')}")

    if 'price_change_period' in result:
        change_emoji = "📈" if result['price_change_period'] > 0 else "📉" if result['price_change_period'] < 0 else "➡️"
        print(f"{change_emoji} Period Change: {result['price_change_period']:+.2f}%")

    if 'support_level' in result and 'resistance_level' in result:
        print(f"📉 Support Level: ${result['support_level']}")
        print(f"📈 Resistance Level: ${result['resistance_level']}")

    if 'analysis_period' in result:
        print(f" Analysis Period: {result['analysis_period']}")

    if 'data_points' in result:
        print(f" Data Points: {result['data_points']} days")

    # Display individual indicators
    if 'indicators' in result:
        print(f"\n📈 Technical Indicators:")
        indicators = result['indicators']
        scores = result.get('individual_scores', {})

        print(f"   • ROC (14-day): {indicators.get('roc_14', 'N/A'):.2f}% [{scores.get('roc', 'N/A')}]")
        print(f"   • Price Velocity: {indicators.get('price_velocity', 'N/A'):.2f} [{scores.get('velocity', 'N/A')}]")
        print(f"   • MACD Signal: {indicators.get('macd_signal', 'N/A')} [{scores.get('macd', 'N/A')}]")
        print(f"   • RSI (14): {indicators.get('rsi_14', 'N/A'):.1f} [{scores.get('rsi', 'N/A')}]")
        print(f"   • MA Position: {indicators.get('ma_position', 'N/A')} [{scores.get('ma', 'N/A')}]")
        print(f"   • Volume Trend: {indicators.get('volume_trend', 'N/A')} [{scores.get('volume', 'N/A')}]")
        print(f"   • Bollinger Position: {indicators.get('bb_position', 'N/A')} [{scores.get('bb', 'N/A')}]")

        if 'volatility' in indicators:
            print(f"   • Volatility: {indicators['volatility']:.1%}")


    return result

if __name__ == "__main__":
    main()

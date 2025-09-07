
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

class GoldMomentumSentiment:
    def __init__(self):
        # Prioritized gold tickers for maximum data reliability
        self.gold_tickers = ["GC=F", "GLD", "XAUUSD=X", "IAU"]
        self.primary_ticker = "GC=F"

    def fetch_gold_data(self, days=90):
        """Fetch comprehensive gold data with multiple timeframes"""
        # Fetch more data for better analysis
        for ticker in self.gold_tickers:
            try:
                print(f"Fetching {days} days of data from {ticker}...")

                end_date = datetime.now()
                start_date = end_date - timedelta(days=days + 20)  # Extra buffer

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

                    if len(gold_data) >= 30:  # Minimum data requirement
                        print(f"✅ Successfully fetched {len(gold_data)} days from {ticker}")
                        self.data_source = ticker
                        return gold_data

            except Exception as e:
                print(f"Failed {ticker}: {e}")
                continue

        raise ValueError("Failed to fetch sufficient gold data from all sources")

    def calculate_advanced_indicators(self, df):
        """Calculate comprehensive set of technical indicators"""
        indicators = {}
        data_length = len(df)

        print(f"Calculating {data_length} advanced indicators...")

        # 1. Multiple Rate of Change periods for trend confirmation
        for period in [7, 14, 21, 30]:
            if data_length > period:
                roc = ((df['Close'].iloc[-1] - df['Close'].iloc[-period-1]) / df['Close'].iloc[-period-1] * 100)
                indicators[f'roc_{period}'] = roc

        # 2. Advanced MACD with multiple timeframes
        if data_length >= 26:
            # Standard MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line

            indicators['macd_line'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = histogram.iloc[-1]
            indicators['macd_bullish'] = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1

            # MACD momentum (rate of change of MACD)
            if len(macd_line) >= 5:
                macd_momentum = (macd_line.iloc[-1] - macd_line.iloc[-5]) / abs(macd_line.iloc[-5]) if macd_line.iloc[-5] != 0 else 0
                indicators['macd_momentum'] = macd_momentum

        # 3. Multiple RSI periods with divergence detection
        for period in [9, 14, 21]:
            if data_length > period:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
                rs = gain / loss.replace(0, 1e-10)
                rsi = 100 - (100 / (1 + rs))
                indicators[f'rsi_{period}'] = rsi.iloc[-1]

        # 4. Stochastic Oscillator
        if data_length >= 14:
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(window=3).mean()

            indicators['stoch_k'] = k_percent.iloc[-1]
            indicators['stoch_d'] = d_percent.iloc[-1]
            indicators['stoch_signal'] = 1 if k_percent.iloc[-1] > d_percent.iloc[-1] else -1

        # 5. Williams %R
        if data_length >= 14:
            high_14 = df['High'].rolling(window=14).max()
            low_14 = df['Low'].rolling(window=14).min()
            williams_r = -100 * (high_14 - df['Close']) / (high_14 - low_14)
            indicators['williams_r'] = williams_r.iloc[-1]

        # 6. Multiple Moving Averages with trend strength
        ma_periods = [10, 20, 50, 100, 200]
        ma_signals = []

        for period in ma_periods:
            if data_length >= period:
                ma = df['Close'].rolling(window=period).mean()
                indicators[f'ma_{period}'] = ma.iloc[-1]
                ma_signals.append(1 if df['Close'].iloc[-1] > ma.iloc[-1] else -1)

                # MA slope (trend strength)
                if len(ma) >= 5:
                    ma_slope = (ma.iloc[-1] - ma.iloc[-5]) / ma.iloc[-5] * 100
                    indicators[f'ma_{period}_slope'] = ma_slope

        # Overall MA trend strength
        indicators['ma_consensus'] = sum(ma_signals) / len(ma_signals) if ma_signals else 0

        # 7. Bollinger Bands with bandwidth and %B
        if data_length >= 20:
            bb_ma = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            bb_upper = bb_ma + (bb_std * 2)
            bb_lower = bb_ma - (bb_std * 2)

            # Bollinger Band %B
            bb_percent_b = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            indicators['bb_percent_b'] = bb_percent_b.iloc[-1]

            # Bollinger Band Width (volatility measure)
            bb_width = (bb_upper - bb_lower) / bb_ma
            indicators['bb_width'] = bb_width.iloc[-1]

            # Position relative to bands
            current_price = df['Close'].iloc[-1]
            if current_price > bb_upper.iloc[-1]:
                indicators['bb_position'] = 1
            elif current_price < bb_lower.iloc[-1]:
                indicators['bb_position'] = -1
            else:
                indicators['bb_position'] = 0

        # 8. Advanced Volatility Analysis
        if data_length >= 20:
            returns = df['Close'].pct_change().dropna()

            # Multiple volatility measures
            indicators['volatility_20d'] = returns.tail(20).std() * np.sqrt(252)
            indicators['volatility_5d'] = returns.tail(5).std() * np.sqrt(252)

            # Volatility trend
            vol_recent = returns.tail(10).std()
            vol_previous = returns.tail(20).head(10).std()
            indicators['volatility_trend'] = 1 if vol_recent > vol_previous else -1

        # 9. Price Action Analysis
        if data_length >= 10:
            # Higher highs, higher lows analysis
            recent_highs = df['High'].tail(10)
            recent_lows = df['Low'].tail(10)

            # Trend consistency
            higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] > recent_highs.iloc[i-1])
            higher_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] > recent_lows.iloc[i-1])

            indicators['trend_consistency'] = (higher_highs + higher_lows) / 18  # Normalized to -1 to 1

        # 10. Volume Analysis (if available)
        if data_length >= 20 and df['Volume'].sum() > 0:
            # Volume trend
            vol_ma_short = df['Volume'].rolling(window=5).mean()
            vol_ma_long = df['Volume'].rolling(window=20).mean()
            indicators['volume_trend'] = 1 if vol_ma_short.iloc[-1] > vol_ma_long.iloc[-1] else -1

            # On-Balance Volume
            obv = (df['Volume'] * (df['Close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0))).cumsum()
            obv_ma = obv.rolling(window=10).mean()
            indicators['obv_signal'] = 1 if obv.iloc[-1] > obv_ma.iloc[-1] else -1
        else:
            indicators['volume_trend'] = 0
            indicators['obv_signal'] = 0

        # 11. Support/Resistance Analysis
        if data_length >= 30:
            # Recent support/resistance levels
            recent_data = df.tail(30)
            resistance = recent_data['High'].quantile(0.95)
            support = recent_data['Low'].quantile(0.05)

            current_price = df['Close'].iloc[-1]
            price_position = (current_price - support) / (resistance - support)
            indicators['sr_position'] = price_position

            # Breakthrough signals
            if current_price > resistance * 0.998:  # Close to resistance
                indicators['resistance_test'] = 1
            elif current_price < support * 1.002:  # Close to support
                indicators['resistance_test'] = -1
            else:
                indicators['resistance_test'] = 0

        # 12. Market Regime Detection
        if data_length >= 50:
            # Trending vs. Ranging market
            price_range = df['High'].tail(20).max() - df['Low'].tail(20).min()
            avg_range = (df['High'] - df['Low']).tail(20).mean()

            # ADX-like calculation for trend strength
            plus_dm = (df['High'].diff()).clip(lower=0)
            minus_dm = (-df['Low'].diff()).clip(lower=0)
            tr = pd.concat([df['High'] - df['Low'], 
                           abs(df['High'] - df['Close'].shift(1)), 
                           abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)

            if len(tr) >= 14:
                atr = tr.rolling(window=14).mean()
                plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(window=14).mean()

                indicators['adx'] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
                indicators['trend_strength'] = 1 if adx.iloc[-1] > 25 else 0
            else:
                indicators['adx'] = 25
                indicators['trend_strength'] = 0

        return indicators

    def advanced_scoring_system(self, indicators):
        """Advanced multi-factor scoring with regime detection"""
        scores = {}

        # 1. Multi-timeframe ROC scoring with confirmation
        roc_scores = []
        roc_weights = {'roc_7': 0.4, 'roc_14': 0.3, 'roc_21': 0.2, 'roc_30': 0.1}

        for roc_key, weight in roc_weights.items():
            if roc_key in indicators:
                roc_val = indicators[roc_key]
                # Dynamic thresholds based on volatility
                vol_adj = indicators.get('volatility_20d', 0.2) / 0.2
                threshold = 2.5 * vol_adj

                if roc_val > threshold:
                    roc_scores.append(1 * weight)
                elif roc_val < -threshold:
                    roc_scores.append(-1 * weight)
                else:
                    roc_scores.append(0)

        scores['roc_composite'] = sum(roc_scores) if roc_scores else 0

        # 2. Advanced MACD scoring
        if 'macd_line' in indicators:
            macd_score = 0
            # Basic MACD signal
            macd_score += indicators.get('macd_bullish', 0) * 0.5

            # MACD momentum
            if 'macd_momentum' in indicators:
                mom = indicators['macd_momentum']
                if mom > 0.05:
                    macd_score += 0.3
                elif mom < -0.05:
                    macd_score -= 0.3

            # MACD histogram direction
            if 'macd_histogram' in indicators:
                hist = indicators['macd_histogram']
                if hist > 0:
                    macd_score += 0.2
                elif hist < 0:
                    macd_score -= 0.2

            scores['macd'] = max(-1, min(1, macd_score))
        else:
            scores['macd'] = 0

        # 3. Multi-RSI consensus
        rsi_signals = []
        for period in [9, 14, 21]:
            rsi_key = f'rsi_{period}'
            if rsi_key in indicators:
                rsi_val = indicators[rsi_key]
                if rsi_val > 60:
                    rsi_signals.append(1)
                elif rsi_val < 40:
                    rsi_signals.append(-1)
                else:
                    rsi_signals.append(0)

        scores['rsi'] = sum(rsi_signals) / len(rsi_signals) if rsi_signals else 0
        scores['rsi'] = max(-1, min(1, scores['rsi']))

        # 4. Stochastic scoring
        if 'stoch_k' in indicators and 'stoch_d' in indicators:
            stoch_k = indicators['stoch_k']
            stoch_d = indicators['stoch_d']

            stoch_score = 0
            # Crossover signal
            stoch_score += indicators.get('stoch_signal', 0) * 0.6

            # Overbought/oversold with momentum
            if stoch_k < 20 and stoch_k > stoch_d:  # Oversold but turning up
                stoch_score += 0.4
            elif stoch_k > 80 and stoch_k < stoch_d:  # Overbought but turning down
                stoch_score -= 0.4

            scores['stochastic'] = max(-1, min(1, stoch_score))
        else:
            scores['stochastic'] = 0

        # 5. Moving Average consensus with trend strength
        ma_score = indicators.get('ma_consensus', 0)

        # Bonus for trend strength (slope analysis)
        slope_bonus = 0
        slope_count = 0
        for period in [10, 20, 50]:
            slope_key = f'ma_{period}_slope'
            if slope_key in indicators:
                slope = indicators[slope_key]
                if abs(slope) > 0.5:  # Significant slope
                    slope_bonus += 1 if slope > 0 else -1
                    slope_count += 1

        if slope_count > 0:
            ma_score += (slope_bonus / slope_count) * 0.3

        scores['moving_average'] = max(-1, min(1, ma_score))

        # 6. Bollinger Band advanced scoring
        if 'bb_percent_b' in indicators:
            bb_score = 0
            bb_b = indicators['bb_percent_b']
            bb_width = indicators.get('bb_width', 0.1)

            # Mean reversion vs. momentum based on volatility
            if bb_width < 0.05:  # Low volatility - mean reversion
                if bb_b < 0.2:
                    bb_score = 0.5  # Oversold, expect bounce
                elif bb_b > 0.8:
                    bb_score = -0.5  # Overbought, expect decline
            else:  # High volatility - momentum
                if bb_b > 0.8:
                    bb_score = 0.5  # Breakout above
                elif bb_b < 0.2:
                    bb_score = -0.5  # Breakdown below

            scores['bollinger'] = bb_score
        else:
            scores['bollinger'] = 0

        # 7. Volume confirmation
        vol_score = (indicators.get('volume_trend', 0) + indicators.get('obv_signal', 0)) / 2
        scores['volume'] = vol_score

        # 8. Trend consistency and momentum
        trend_score = indicators.get('trend_consistency', 0)
        if 'adx' in indicators:
            adx_val = indicators['adx']
            if adx_val > 25:  # Strong trend
                trend_score *= 1.5  # Amplify trend signals
            elif adx_val < 20:  # Weak trend
                trend_score *= 0.5  # Reduce trend signals

        scores['trend'] = max(-1, min(1, trend_score))

        # 9. Support/Resistance breakthrough
        scores['support_resistance'] = indicators.get('resistance_test', 0)

        return scores

    def calculate_super_weighted_sentiment(self, scores, indicators):
        """Advanced weighted calculation with regime-aware weighting"""

        # Base weights
        base_weights = {
            'roc_composite': 0.20,
            'macd': 0.15,
            'rsi': 0.12,
            'moving_average': 0.15,
            'stochastic': 0.10,
            'bollinger': 0.08,
            'volume': 0.05,
            'trend': 0.10,
            'support_resistance': 0.05
        }

        # Dynamic weight adjustment based on market regime
        weights = base_weights.copy()

        # If trending market, increase trend-following indicators
        if indicators.get('adx', 25) > 25:
            weights['roc_composite'] *= 1.2
            weights['moving_average'] *= 1.2
            weights['trend'] *= 1.3
            weights['bollinger'] *= 0.8  # Reduce mean reversion
        else:  # Ranging market
            weights['rsi'] *= 1.2
            weights['stochastic'] *= 1.2
            weights['bollinger'] *= 1.3  # Increase mean reversion
            weights['trend'] *= 0.7

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        # Calculate weighted score
        weighted_score = sum(scores.get(key, 0) * weight for key, weight in weights.items())

        # Advanced confidence calculation
        confidence = self._calculate_advanced_confidence(scores, indicators)

        # Enhanced classification with momentum confirmation
        momentum_strength = abs(weighted_score)

        if weighted_score >= 0.20 and confidence > 0.6:
            return 1, "BULLISH", weighted_score, confidence
        elif weighted_score <= -0.20 and confidence > 0.6:
            return -1, "BEARISH", weighted_score, confidence
        elif momentum_strength < 0.15 or confidence < 0.5:
            return 0, "NEUTRAL", weighted_score, confidence
        else:
            # Weak signal classification
            if weighted_score > 0:
                return 1, "WEAK BULLISH", weighted_score, confidence
            else:
                return -1, "WEAK BEARISH", weighted_score, confidence

    def _calculate_advanced_confidence(self, scores, indicators):
        """Calculate confidence based on signal agreement and strength"""

        # Signal consistency check
        non_zero_scores = [score for score in scores.values() if abs(score) > 0.1]

        if not non_zero_scores:
            return 0.0

        # Calculate signal agreement
        positive_signals = len([s for s in non_zero_scores if s > 0])
        negative_signals = len([s for s in non_zero_scores if s < 0])
        total_signals = len(non_zero_scores)

        # Base confidence from signal agreement
        dominant_signals = max(positive_signals, negative_signals)
        base_confidence = dominant_signals / total_signals

        # Adjust confidence based on signal strength
        avg_signal_strength = np.mean([abs(s) for s in non_zero_scores])
        strength_multiplier = min(1.2, 0.5 + avg_signal_strength)

        # Adjust for trend strength
        trend_multiplier = 1.0
        if 'adx' in indicators:
            adx_val = indicators['adx']
            if adx_val > 25:
                trend_multiplier = min(1.3, 1.0 + (adx_val - 25) / 100)
            elif adx_val < 15:
                trend_multiplier = max(0.7, 1.0 - (15 - adx_val) / 100)

        # Volatility adjustment
        vol_multiplier = 1.0
        if 'volatility_20d' in indicators:
            vol = indicators['volatility_20d']
            if vol > 0.3:  # High volatility reduces confidence
                vol_multiplier = max(0.8, 1.0 - (vol - 0.3) / 0.5)
            elif vol < 0.15:  # Very low volatility also reduces confidence
                vol_multiplier = max(0.9, 1.0 - (0.15 - vol) / 0.1)

        # Final confidence calculation
        final_confidence = base_confidence * strength_multiplier * trend_multiplier * vol_multiplier

        return min(1.0, max(0.0, final_confidence))

    def analyze_super_sentiment(self, days=90):
        """Super comprehensive sentiment analysis"""
        try:
            print("Starting Super-Optimized Gold Sentiment Analysis...")

            # Fetch comprehensive data
            df = self.fetch_gold_data(days=days)

            if df is None or len(df) < 30:
                return {"error": "Insufficient gold price data for super analysis"}

            print(f"Analyzing {len(df)} days of high-quality data...")

            # Calculate advanced indicators
            indicators = self.calculate_advanced_indicators(df)

            # Advanced scoring
            scores = self.advanced_scoring_system(indicators)

            # Super weighted sentiment calculation
            sentiment, label, weighted_score, confidence = self.calculate_super_weighted_sentiment(scores, indicators)

            # Comprehensive market analysis
            analysis = self._generate_market_analysis(df, indicators, scores)

            # Period performance
            price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)

            # Dynamic support/resistance
            support_resistance = self._calculate_dynamic_levels(df)

            return {
                "sentiment_score": sentiment,
                "sentiment_label": label,
                "weighted_score": round(weighted_score, 4),
                "confidence": round(confidence, 4),
                "current_price": round(df['Close'].iloc[-1], 2),
                "price_change_period": round(price_change, 2),
                "support_level": round(support_resistance['support'], 2),
                "resistance_level": round(support_resistance['resistance'], 2),
                "key_indicators": {
                    "trend_strength": round(indicators.get('adx', 25), 1),
                    "volatility": f"{indicators.get('volatility_20d', 0.2):.1%}",
                    "rsi_14": round(indicators.get('rsi_14', 50), 1),
                    "macd_signal": "BULLISH" if indicators.get('macd_bullish', 0) > 0 else "BEARISH",
                    "ma_consensus": f"{indicators.get('ma_consensus', 0)*100:.0f}%",
                    "bb_position": indicators.get('bb_percent_b', 0.5)
                },
                "individual_scores": {k: round(v, 3) for k, v in scores.items()},
                "market_analysis": analysis,
                "data_points": len(df),
                "data_source": getattr(self, 'data_source', 'Unknown'),
                "analysis_period": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "error": f"Super analysis failed: {str(e)}",
                "sentiment_score": 0,
                "sentiment_label": "ERROR",
                "weighted_score": 0.0,
                "confidence": 0.0,
                "current_price": 0.0,
                "timestamp": datetime.now().isoformat()
            }

    def _generate_market_analysis(self, df, indicators, scores):
        """Generate comprehensive market analysis"""
        analysis = []

        # Trend analysis
        adx = indicators.get('adx', 25)
        if adx > 25:
            analysis.append(f"Strong trending market (ADX: {adx:.1f})")
        else:
            analysis.append(f"Ranging/consolidating market (ADX: {adx:.1f})")

        # Momentum analysis
        roc_14 = indicators.get('roc_14', 0)
        if abs(roc_14) > 3:
            direction = "upward" if roc_14 > 0 else "downward"
            analysis.append(f"Strong {direction} momentum ({roc_14:+.1f}%)")

        # Volatility regime
        vol = indicators.get('volatility_20d', 0.2)
        if vol > 0.25:
            analysis.append(f"High volatility environment ({vol:.1%})")
        elif vol < 0.15:
            analysis.append(f"Low volatility environment ({vol:.1%})")

        # Overbought/oversold conditions
        rsi = indicators.get('rsi_14', 50)
        if rsi > 70:
            analysis.append("Potentially overbought conditions")
        elif rsi < 30:
            analysis.append("Potentially oversold conditions")

        return analysis

    def _calculate_dynamic_levels(self, df):
        """Calculate dynamic support and resistance levels"""
        # Use multiple methods for robust levels
        recent_data = df.tail(60)  # Last 60 days

        # Method 1: Statistical levels
        resistance_stat = recent_data['High'].quantile(0.95)
        support_stat = recent_data['Low'].quantile(0.05)

        # Method 2: Pivot points
        recent_high = recent_data['High'].max()
        recent_low = recent_data['Low'].min()
        recent_close = recent_data['Close'].iloc[-1]

        pivot = (recent_high + recent_low + recent_close) / 3
        resistance_pivot = pivot + (recent_high - recent_low) * 0.382
        support_pivot = pivot - (recent_high - recent_low) * 0.382

        # Combine methods
        resistance = (resistance_stat + resistance_pivot) / 2
        support = (support_stat + support_pivot) / 2

        return {'support': support, 'resistance': resistance}

# Enhanced main function
def main():
    print("Initializing Optimized Gold Sentiment Analyzer...")
    print("Using advanced multi-factor analysis with 90 days of data")

    analyzer = GoldMomentumSentiment()

    try:
        import yfinance
        print(f"yfinance ready for super analysis")
    except ImportError:
        print("yfinance required. Install with: pip install yfinance scipy")
        return

    result = analyzer.analyze_super_sentiment(days=90)

    if 'error' in result and result.get('sentiment_score', 0) == 0:
        print(f"❌ Error: {result['error']}")
        return result

    # Main results with enhanced display
    sentiment_emoji = "🟢" if result['sentiment_score'] == 1 else "🔴" if result['sentiment_score'] == -1 else "🟡"
    confidence_stars = "⭐" * min(5, int(result.get('confidence', 0) * 5))

    print(f"\n **SENTIMENT**: {sentiment_emoji} {result.get('sentiment_label', 'N/A')} ({result.get('sentiment_score', 'N/A')})")
    print(f"  **WEIGHTED SCORE**: {result.get('weighted_score', 'N/A')} (Range: -1.0 to +1.0)")
    print(f" **CONFIDENCE**: {result.get('confidence', 0)*100:.1f}% {confidence_stars}")
    print(f"**CURRENT PRICE**: ${result.get('current_price', 'N/A')}")
    print(f"**DATA SOURCE**: {result.get('data_source', 'N/A')}")

    if 'price_change_period' in result:
        change_emoji = "📈" if result['price_change_period'] > 0 else "📉" if result['price_change_period'] < 0 else "➡️"
        print(f"{change_emoji} **PERIOD CHANGE**: {result['price_change_period']:+.2f}%")

    # Key levels
    if 'support_level' in result:
        print(f" **SUPPORT**: ${result['support_level']} |  **RESISTANCE**: ${result['resistance_level']}")

    # Key indicators summary
    if 'key_indicators' in result:
        ki = result['key_indicators']
        print(f"\n📈 **KEY INDICATORS SUMMARY**:")
        print(f"   • Trend Strength (ADX): {ki.get('trend_strength', 'N/A')}")
        print(f"   • Volatility: {ki.get('volatility', 'N/A')}")
        print(f"   • RSI (14): {ki.get('rsi_14', 'N/A')}")
        print(f"   • MACD Signal: {ki.get('macd_signal', 'N/A')}")
        print(f"   • MA Consensus: {ki.get('ma_consensus', 'N/A')}")
        print(f"   • Bollinger %B: {ki.get('bb_position', 'N/A'):.2f}")

    # Market analysis
    if 'market_analysis' in result and result['market_analysis']:
        print(f"\n **MARKET ANALYSIS**:")
        for analysis_point in result['market_analysis']:
            print(f"   • {analysis_point}")

    # Individual scores breakdown
    if 'individual_scores' in result:
        print(f"\n **COMPONENT SCORES** (Range: -1.0 to +1.0):")
        scores = result['individual_scores']
        for component, score in scores.items():
            score_bar = "█" * int(abs(score) * 10) if abs(score) > 0.1 else "░"
            emoji = "🟢" if score > 0.3 else "🔴" if score < -0.3 else "🟡"
            print(f"   {emoji} {component.replace('_', ' ').title()}: {score:+.3f} {score_bar}")

    # Data quality
    print(f"\n📅 **ANALYSIS PERIOD**: {result.get('analysis_period', 'N/A')}")
    print(f"📊 **DATA POINTS**: {result.get('data_points', 'N/A')} days (High Quality)")



    return result

if __name__ == "__main__":
    main()

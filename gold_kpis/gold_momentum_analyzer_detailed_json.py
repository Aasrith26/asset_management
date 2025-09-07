# UPDATED GOLD MOMENTUM ANALYZER - JSON OUTPUT VERSION
# =====================================================
# Modified to return detailed context for explainable analysis
# All sub-indicators (ROC, MACD, RSI, MA, etc.) captured in JSON

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
                        self.data_source = ticker
                        return data
            except Exception as e:
                continue
        raise ValueError("Unable to fetch gold price data")

    def calculate_technical_indicators(self, df):
        """Calculate standard technical indicators with detailed context"""
        indicators = {}
        
        # Rate of Change (14-day)
        if len(df) >= 15:
            roc_value = ((df['Close'].iloc[-1] - df['Close'].iloc[-15]) / df['Close'].iloc[-15]) * 100
            indicators['roc_14'] = {
                'value': float(roc_value),
                'threshold_analysis': {
                    'strong_bullish': roc_value > 3,
                    'bullish': 1 < roc_value <= 3,
                    'neutral': -1 <= roc_value <= 1,
                    'bearish': -3 <= roc_value < -1,
                    'strong_bearish': roc_value < -3
                }
            }

        # MACD (12, 26, 9)
        if len(df) >= 26:
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            
            indicators['macd'] = {
                'macd_value': float(macd_line.iloc[-1]),
                'macd_signal': float(signal_line.iloc[-1]),
                'macd_histogram': float(macd_line.iloc[-1] - signal_line.iloc[-1]),
                'crossover_status': 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1,
                'strength': float(abs(macd_line.iloc[-1]) / (abs(signal_line.iloc[-1]) + 1e-10))
            }

        # RSI (14-day)
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_value = 100 - (100 / (1 + rs.iloc[-1]))
            
            indicators['rsi'] = {
                'value': float(rsi_value),
                'zone_analysis': {
                    'overbought': rsi_value >= 70,
                    'strong': 60 <= rsi_value < 70,
                    'neutral': 40 < rsi_value < 60,
                    'weak': 30 < rsi_value <= 40,
                    'oversold': rsi_value <= 30
                }
            }

        # Moving Averages
        ma_analysis = {}
        for period in [20, 50]:
            if len(df) >= period:
                ma = df['Close'].rolling(window=period).mean()
                price_vs_ma = (df['Close'].iloc[-1] / ma.iloc[-1] - 1) * 100
                
                ma_analysis[f'ma_{period}'] = {
                    'ma_value': float(ma.iloc[-1]),
                    'current_price': float(df['Close'].iloc[-1]),
                    'price_vs_ma_pct': float(price_vs_ma),
                    'trend_strength': 'strong' if abs(price_vs_ma) > 2 else 'moderate' if abs(price_vs_ma) > 1 else 'weak'
                }
        
        indicators['moving_averages'] = ma_analysis

        # Volatility (20-day)
        if len(df) >= 20:
            returns = df['Close'].pct_change().dropna()
            volatility = returns.tail(20).std() * np.sqrt(252)
            
            indicators['volatility'] = {
                'annualized_volatility': float(volatility),
                'volatility_regime': 'high' if volatility > 0.25 else 'normal' if volatility > 0.15 else 'low',
                'recent_20d_std': float(returns.tail(20).std())
            }

        # Price momentum
        if len(df) >= 10:
            momentum_10d = ((df['Close'].iloc[-1] / df['Close'].iloc[-10]) - 1) * 100
            indicators['momentum_10d'] = {
                'value': float(momentum_10d),
                'strength': 'strong' if abs(momentum_10d) > 5 else 'moderate' if abs(momentum_10d) > 2 else 'weak'
            }

        return indicators

    def score_indicators_with_context(self, indicators):
        """Score each indicator with detailed reasoning"""
        scores = {}
        scoring_details = {}
        
        # ROC scoring
        if 'roc_14' in indicators:
            roc = indicators['roc_14']['value']
            if roc > 3:
                scores['roc'] = 1
                scoring_details['roc'] = 'Strong positive momentum (>3%)'
            elif roc > 1:
                scores['roc'] = 0.5
                scoring_details['roc'] = 'Moderate positive momentum (1-3%)'
            elif roc < -3:
                scores['roc'] = -1
                scoring_details['roc'] = 'Strong negative momentum (<-3%)'
            elif roc < -1:
                scores['roc'] = -0.5
                scoring_details['roc'] = 'Moderate negative momentum (-3% to -1%)'
            else:
                scores['roc'] = 0
                scoring_details['roc'] = 'Neutral momentum (-1% to 1%)'

        # MACD scoring
        if 'macd' in indicators:
            macd_cross = indicators['macd']['crossover_status']
            macd_strength = indicators['macd']['strength']
            
            if macd_cross == 1:
                scores['macd'] = min(1.0, 0.5 + (macd_strength - 1) * 0.5)
                scoring_details['macd'] = f'Bullish crossover with strength {macd_strength:.2f}'
            else:
                scores['macd'] = max(-1.0, -0.5 - (macd_strength - 1) * 0.5)
                scoring_details['macd'] = f'Bearish crossover with strength {macd_strength:.2f}'

        # RSI scoring
        if 'rsi' in indicators:
            rsi = indicators['rsi']['value']
            if rsi >= 70:
                scores['rsi'] = -0.5
                scoring_details['rsi'] = f'Overbought territory ({rsi:.1f})'
            elif rsi >= 60:
                scores['rsi'] = 0.5
                scoring_details['rsi'] = f'Strong but not overbought ({rsi:.1f})'
            elif rsi <= 30:
                scores['rsi'] = 0.5
                scoring_details['rsi'] = f'Oversold - potential bounce ({rsi:.1f})'
            elif rsi <= 40:
                scores['rsi'] = -0.5
                scoring_details['rsi'] = f'Weak momentum ({rsi:.1f})'
            else:
                scores['rsi'] = 0
                scoring_details['rsi'] = f'Neutral territory ({rsi:.1f})'

        # Moving Average scoring
        if 'moving_averages' in indicators:
            ma_scores = []
            ma_details = []
            
            for ma_key, ma_data in indicators['moving_averages'].items():
                price_vs_ma = ma_data['price_vs_ma_pct']
                if price_vs_ma > 2:
                    ma_scores.append(1)
                    ma_details.append(f'{ma_key}: {price_vs_ma:+.1f}% (strong uptrend)')
                elif price_vs_ma > 0:
                    ma_scores.append(0.5)
                    ma_details.append(f'{ma_key}: {price_vs_ma:+.1f}% (above average)')
                elif price_vs_ma < -2:
                    ma_scores.append(-1)
                    ma_details.append(f'{ma_key}: {price_vs_ma:+.1f}% (strong downtrend)')
                elif price_vs_ma < 0:
                    ma_scores.append(-0.5)
                    ma_details.append(f'{ma_key}: {price_vs_ma:+.1f}% (below average)')
                else:
                    ma_scores.append(0)
                    ma_details.append(f'{ma_key}: {price_vs_ma:+.1f}% (neutral)')
            
            scores['moving_average'] = np.mean(ma_scores) if ma_scores else 0
            scoring_details['moving_average'] = '; '.join(ma_details)

        # Momentum scoring
        if 'momentum_10d' in indicators:
            momentum = indicators['momentum_10d']['value']
            if momentum > 5:
                scores['momentum'] = 1
                scoring_details['momentum'] = f'Strong 10d momentum ({momentum:+.1f}%)'
            elif momentum > 2:
                scores['momentum'] = 0.5
                scoring_details['momentum'] = f'Moderate 10d momentum ({momentum:+.1f}%)'
            elif momentum < -5:
                scores['momentum'] = -1
                scoring_details['momentum'] = f'Strong negative momentum ({momentum:+.1f}%)'
            elif momentum < -2:
                scores['momentum'] = -0.5
                scoring_details['momentum'] = f'Moderate negative momentum ({momentum:+.1f}%)'
            else:
                scores['momentum'] = 0
                scoring_details['momentum'] = f'Neutral momentum ({momentum:+.1f}%)'

        return scores, scoring_details

    def analyze_momentum_detailed(self, days=60):
        """Main analysis function returning detailed JSON context"""
        try:
            # Fetch data
            df = self.fetch_price_data(days)
            if df is None or len(df) < 20:
                return {"error": "Insufficient price data"}

            # Calculate indicators
            indicators = self.calculate_technical_indicators(df)
            
            # Score indicators
            scores, scoring_details = self.score_indicators_with_context(indicators)
            
            # Calculate sentiment
            sentiment, confidence = self.calculate_composite_sentiment(scores)
            
            # Build comprehensive context
            current_price = df['Close'].iloc[-1]
            period_change = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            
            # Support/Resistance levels
            recent_data = df.tail(30)
            support = recent_data['Low'].quantile(0.2)
            resistance = recent_data['High'].quantile(0.8)
            
            # Build detailed JSON response
            detailed_result = {
                "component_sentiment": float(sentiment),
                "component_confidence": float(confidence),
                "component_weight": 0.25,
                "weighted_contribution": float(sentiment * 0.25),
                
                "sub_indicators": {
                    "roc_14d": {
                        "value": indicators.get('roc_14', {}).get('value', 0),
                        "score": scores.get('roc', 0),
                        "interpretation": scoring_details.get('roc', 'No ROC data'),
                        "threshold_analysis": indicators.get('roc_14', {}).get('threshold_analysis', {})
                    },
                    "macd": {
                        "macd_value": indicators.get('macd', {}).get('macd_value', 0),
                        "macd_signal": indicators.get('macd', {}).get('macd_signal', 0),
                        "macd_histogram": indicators.get('macd', {}).get('macd_histogram', 0),
                        "crossover_status": indicators.get('macd', {}).get('crossover_status', 0),
                        "score": scores.get('macd', 0),
                        "interpretation": scoring_details.get('macd', 'No MACD data')
                    },
                    "rsi_14": {
                        "value": indicators.get('rsi', {}).get('value', 50),
                        "score": scores.get('rsi', 0),
                        "interpretation": scoring_details.get('rsi', 'No RSI data'),
                        "zone_analysis": indicators.get('rsi', {}).get('zone_analysis', {})
                    },
                    "moving_averages": {
                        "ma_20": indicators.get('moving_averages', {}).get('ma_20', {}),
                        "ma_50": indicators.get('moving_averages', {}).get('ma_50', {}),
                        "score": scores.get('moving_average', 0),
                        "interpretation": scoring_details.get('moving_average', 'No MA data')
                    },
                    "momentum_10d": {
                        "value": indicators.get('momentum_10d', {}).get('value', 0),
                        "score": scores.get('momentum', 0),
                        "interpretation": scoring_details.get('momentum', 'No momentum data'),
                        "strength": indicators.get('momentum_10d', {}).get('strength', 'unknown')
                    },
                    "volatility": {
                        "annualized_volatility": indicators.get('volatility', {}).get('annualized_volatility', 0.2),
                        "volatility_regime": indicators.get('volatility', {}).get('volatility_regime', 'normal'),
                        "score": 0,  # Volatility doesn't contribute to sentiment directly
                        "interpretation": f"Volatility regime: {indicators.get('volatility', {}).get('volatility_regime', 'normal')}"
                    }
                },
                
                "market_context": {
                    "current_price": float(current_price),
                    "period_change_pct": float(period_change),
                    "support_level": float(support),
                    "resistance_level": float(resistance),
                    "data_points": len(df),
                    "data_source": getattr(self, 'data_source', 'Unknown'),
                    "analysis_period_days": days,
                    "price_range": {
                        "high": float(df['High'].max()),
                        "low": float(df['Low'].min()),
                        "range_pct": float((df['High'].max() - df['Low'].min()) / df['Low'].min() * 100)
                    }
                },
                
                "scoring_breakdown": {
                    "individual_scores": scores,
                    "scoring_explanations": scoring_details,
                    "composite_calculation": {
                        "weights": {
                            'roc': 0.25,
                            'macd': 0.25,
                            'rsi': 0.20,
                            'moving_average': 0.20,
                            'momentum': 0.10
                        },
                        "weighted_sum": sum(scores.get(key, 0) * weight for key, weight in {
                            'roc': 0.25, 'macd': 0.25, 'rsi': 0.20, 'moving_average': 0.20, 'momentum': 0.10
                        }.items())
                    }
                },
                
                "analysis_timestamp": datetime.now().isoformat(),
                "data_freshness": "real_time"
            }
            
            return detailed_result
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "component_sentiment": 0.0,
                "component_confidence": 0.0,
                "analysis_timestamp": datetime.now().isoformat()
            }

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
            positive_signals = len([s for s in active_signals if s > 0])
            negative_signals = len([s for s in active_signals if s < 0])
            total_signals = len(active_signals)
            agreement_ratio = max(positive_signals, negative_signals) / total_signals
            signal_strength = np.mean([abs(s) for s in active_signals])
            confidence = agreement_ratio * signal_strength
            confidence = min(0.95, max(0.3, confidence))
        
        return weighted_score, confidence

# Test function
if __name__ == "__main__":
    analyzer = GoldMomentumAnalyzer()
    result = analyzer.analyze_momentum_detailed()
    
    # Save to JSON file instead of printing
    import json
    with open('technical_analysis_detailed.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("✅ Technical analysis saved to: technical_analysis_detailed.json")
    print(f"📊 Sentiment: {result.get('component_sentiment', 0):.3f}")
    print(f"🎯 Confidence: {result.get('component_confidence', 0):.1%}")
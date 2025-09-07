# NIFTY 50 TECHNICAL ANALYSIS SENTIMENT ANALYZER v1.0
# ===================================================
# Real-time technical analysis for Nifty 50 index
# 30% weight in composite sentiment - second highest priority KPI

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import asyncio
from typing import Dict, List, Optional, Tuple
import json
import hashlib
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class CachedTechData:
    data: any
    timestamp: datetime
    ttl_minutes: int = 15  # 15 minutes for technical data
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(minutes=self.ttl_minutes)

class NiftyTechnicalAnalyzer:
    """
    Nifty 50 Technical Analysis Sentiment Analyzer
    - Real-time price momentum analysis
    - Multiple technical indicators
    - Volume analysis integration
    - 30% weight in composite sentiment
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.cache = {}
        self.nifty_tickers = ["^NSEI", "NIFTY50.NS", "^NSEBANK"]  # Multiple fallbacks
        print("Nifty 50 Technical Analysis Analyzer v1.0 initialized")
        print("📈 Real-time technical momentum tracking enabled")
    
    def _default_config(self) -> Dict:
        return {
            'lookback_days': 100,  # 100 days for technical analysis
            'cache_ttl_minutes': 15,  # 15 minutes for technical data
            'confidence_threshold': 0.75,
            'technical_weights': {
                'trend_mas': 0.25,    # Moving averages trend
                'momentum_rsi': 0.20, # RSI momentum
                'trend_macd': 0.20,   # MACD trend
                'volume_trend': 0.15, # Volume analysis
                'bollinger': 0.10,    # Bollinger bands
                'support_resistance': 0.10  # S/R levels
            },
            'rsi_periods': [14, 21],
            'ma_periods': [20, 50, 200],
            'macd_params': {'fast': 12, 'slow': 26, 'signal': 9}
        }
    
    def _get_cache_key(self, source: str, params: str = "") -> str:
        """Generate cache key for technical data"""
        return hashlib.md5(f"tech_{source}_{params}".encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[any]:
        """Retrieve cached technical data if valid"""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if not cached.is_expired():
                return cached.data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: any, ttl_minutes: int = None):
        """Cache technical data with TTL"""
        ttl = ttl_minutes or self.config['cache_ttl_minutes']
        self.cache[cache_key] = CachedTechData(data, datetime.now(), ttl)
    
    async def fetch_nifty_data(self) -> pd.DataFrame:
        """Fetch Nifty 50 price data with multiple fallbacks"""
        cache_key = self._get_cache_key("nifty_price_data")
        cached_result = self._get_cached_data(cache_key)
        
        if cached_result is not None:
            print("📊 Nifty Data: Using cached result")
            return cached_result
        
        print("📊 Fetching real-time Nifty 50 data...")
        
        for ticker in self.nifty_tickers:
            try:
                nifty = yf.Ticker(ticker)
                # Get extended data for better technical analysis
                hist = nifty.history(period=f"{self.config['lookback_days']}d", interval="1d")
                
                if not hist.empty and len(hist) >= 50:  # Minimum data requirement
                    print(f"✅ Successfully fetched {len(hist)} days of data from {ticker}")
                    self._cache_data(cache_key, hist)
                    return hist
                    
            except Exception as e:
                print(f"Failed to fetch from {ticker}: {e}")
                continue
        
        raise ValueError("Unable to fetch Nifty 50 data from any source")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        print("📊 Calculating technical indicators...")
        
        indicators = {}
        
        # 1. Moving Averages Analysis
        indicators['moving_averages'] = self._calculate_moving_averages(df)
        
        # 2. RSI Analysis
        indicators['rsi_analysis'] = self._calculate_rsi_analysis(df)
        
        # 3. MACD Analysis
        indicators['macd_analysis'] = self._calculate_macd_analysis(df)
        
        # 4. Volume Analysis
        indicators['volume_analysis'] = self._calculate_volume_analysis(df)
        
        # 5. Bollinger Bands
        indicators['bollinger_bands'] = self._calculate_bollinger_bands(df)
        
        # 6. Support/Resistance
        indicators['support_resistance'] = self._calculate_support_resistance(df)
        
        # 7. Price Action Patterns
        indicators['price_patterns'] = self._analyze_price_patterns(df)
        
        return indicators
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict:
        """Calculate moving averages and trend analysis"""
        ma_data = {}
        current_price = df['Close'].iloc[-1]
        
        for period in self.config['ma_periods']:
            ma = df['Close'].rolling(window=period).mean()
            ma_data[f'ma_{period}'] = {
                'value': float(ma.iloc[-1]),
                'price_vs_ma_pct': float((current_price - ma.iloc[-1]) / ma.iloc[-1] * 100),
                'trend': 'above' if current_price > ma.iloc[-1] else 'below',
                'ma_slope': float(ma.iloc[-1] - ma.iloc[-5]) if len(ma) >= 5 else 0
            }
        
        # Overall MA trend score
        above_count = sum(1 for key, data in ma_data.items() if data['trend'] == 'above')
        ma_trend_score = (above_count / len(self.config['ma_periods']) - 0.5) * 2  # -1 to 1
        
        return {
            'individual_mas': ma_data,
            'trend_score': float(ma_trend_score),
            'trend_strength': 'strong' if abs(ma_trend_score) > 0.6 else 'moderate' if abs(ma_trend_score) > 0.3 else 'weak',
            'overall_trend': 'bullish' if ma_trend_score > 0.2 else 'bearish' if ma_trend_score < -0.2 else 'neutral',
            'current_price': float(current_price)
        }
    
    def _calculate_rsi_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate RSI with multiple periods and analysis"""
        rsi_data = {}
        
        for period in self.config['rsi_periods']:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            rsi_data[f'rsi_{period}'] = {
                'value': float(current_rsi),
                'level': self._classify_rsi_level(current_rsi),
                'trend': 'rising' if rsi.iloc[-1] > rsi.iloc[-3] else 'falling',
                'momentum': float(rsi.iloc[-1] - rsi.iloc[-5]) if len(rsi) >= 5 else 0
            }
        
        # Primary RSI (14-period)
        primary_rsi = rsi_data['rsi_14']['value']
        
        return {
            'individual_rsi': rsi_data,
            'primary_rsi': float(primary_rsi),
            'rsi_signal': self._get_rsi_signal(primary_rsi),
            'overbought_oversold': 'overbought' if primary_rsi > 70 else 'oversold' if primary_rsi < 30 else 'normal',
            'rsi_divergence': self._check_rsi_divergence(df, rsi)
        }
    
    def _calculate_macd_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD with signal analysis"""
        params = self.config['macd_params']
        
        ema_fast = df['Close'].ewm(span=params['fast']).mean()
        ema_slow = df['Close'].ewm(span=params['slow']).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=params['signal']).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # MACD signals
        bullish_crossover = current_macd > current_signal and macd_line.iloc[-2] <= signal_line.iloc[-2]
        bearish_crossover = current_macd < current_signal and macd_line.iloc[-2] >= signal_line.iloc[-2]
        
        return {
            'macd_line': float(current_macd),
            'signal_line': float(current_signal),
            'histogram': float(current_histogram),
            'crossover_status': 'bullish' if current_macd > current_signal else 'bearish',
            'recent_crossover': 'bullish' if bullish_crossover else 'bearish' if bearish_crossover else 'none',
            'histogram_trend': 'increasing' if current_histogram > histogram.iloc[-3] else 'decreasing',
            'macd_strength': float(abs(current_macd - current_signal)),
            'zero_line_cross': 'above' if current_macd > 0 else 'below'
        }
    
    def _calculate_volume_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns and trends"""
        volume = df['Volume']
        price = df['Close']
        
        # Volume moving averages
        vol_ma_20 = volume.rolling(window=20).mean()
        vol_ma_50 = volume.rolling(window=50).mean()
        
        current_volume = volume.iloc[-1]
        avg_volume_20 = vol_ma_20.iloc[-1]
        avg_volume_50 = vol_ma_50.iloc[-1]
        
        # Volume trend
        volume_ratio_20 = current_volume / avg_volume_20
        volume_ratio_50 = current_volume / avg_volume_50
        
        # Price-Volume relationship
        price_change = (price.iloc[-1] - price.iloc[-2]) / price.iloc[-2]
        volume_change = (current_volume - volume.iloc[-2]) / volume.iloc[-2]
        
        # On-Balance Volume approximation
        obv_trend = self._calculate_obv_trend(df)
        
        return {
            'current_volume': float(current_volume),
            'volume_vs_20d_avg': float(volume_ratio_20),
            'volume_vs_50d_avg': float(volume_ratio_50),
            'volume_trend': 'high' if volume_ratio_20 > 1.5 else 'normal' if volume_ratio_20 > 0.8 else 'low',
            'price_volume_relationship': 'positive' if (price_change > 0 and volume_change > 0) or (price_change < 0 and volume_change < 0) else 'negative',
            'obv_trend': obv_trend,
            'volume_momentum': float(volume_change)
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        """Calculate Bollinger Bands analysis"""
        period = 20
        std_dev = 2
        
        ma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        current_price = df['Close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_ma = ma.iloc[-1]
        
        # Bollinger position
        bb_position = (current_price - current_lower) / (current_upper - current_lower)
        
        return {
            'upper_band': float(current_upper),
            'middle_band': float(current_ma),
            'lower_band': float(current_lower),
            'current_price': float(current_price),
            'bb_position': float(bb_position),
            'price_position': 'above_upper' if current_price > current_upper else 'below_lower' if current_price < current_lower else 'within_bands',
            'band_width': float((current_upper - current_lower) / current_ma * 100),
            'squeeze_status': 'squeeze' if (current_upper - current_lower) / current_ma < 0.1 else 'expansion'
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        high_prices = df['High'].tail(50)
        low_prices = df['Low'].tail(50)
        current_price = df['Close'].iloc[-1]
        
        # Simple S/R calculation using recent highs/lows
        resistance_levels = []
        support_levels = []
        
        # Find local maxima and minima
        for i in range(2, len(high_prices) - 2):
            if (high_prices.iloc[i] > high_prices.iloc[i-1] and 
                high_prices.iloc[i] > high_prices.iloc[i+1] and
                high_prices.iloc[i] > high_prices.iloc[i-2] and 
                high_prices.iloc[i] > high_prices.iloc[i+2]):
                resistance_levels.append(high_prices.iloc[i])
        
        for i in range(2, len(low_prices) - 2):
            if (low_prices.iloc[i] < low_prices.iloc[i-1] and 
                low_prices.iloc[i] < low_prices.iloc[i+1] and
                low_prices.iloc[i] < low_prices.iloc[i-2] and 
                low_prices.iloc[i] < low_prices.iloc[i+2]):
                support_levels.append(low_prices.iloc[i])
        
        # Get nearest levels
        resistance_levels = sorted(resistance_levels, reverse=True)
        support_levels = sorted(support_levels, reverse=True)
        
        nearest_resistance = next((r for r in resistance_levels if r > current_price), None)
        nearest_support = next((s for s in support_levels if s < current_price), None)
        
        return {
            'nearest_resistance': float(nearest_resistance) if nearest_resistance else None,
            'nearest_support': float(nearest_support) if nearest_support else None,
            'resistance_distance_pct': float((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None,
            'support_distance_pct': float((current_price - nearest_support) / current_price * 100) if nearest_support else None,
            'all_resistance_levels': [float(r) for r in resistance_levels[:3]],
            'all_support_levels': [float(s) for s in support_levels[:3]]
        }
    
    def _analyze_price_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze price action patterns"""
        closes = df['Close'].tail(10)
        highs = df['High'].tail(10)
        lows = df['Low'].tail(10)
        
        # Higher highs, higher lows pattern
        recent_trend = self._identify_trend_pattern(closes)
        
        # Candlestick patterns (simplified)
        latest_candle = self._analyze_latest_candle(df)
        
        # Price momentum
        momentum_5d = (closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6] * 100 if len(closes) >= 6 else 0
        momentum_10d = (closes.iloc[-1] - closes.iloc[-10]) / closes.iloc[-10] * 100 if len(closes) >= 10 else 0
        
        return {
            'trend_pattern': recent_trend,
            'latest_candle': latest_candle,
            'momentum_5d_pct': float(momentum_5d),
            'momentum_10d_pct': float(momentum_10d),
            'volatility_10d': float(closes.pct_change().tail(10).std() * np.sqrt(252) * 100)
        }
    
    def _classify_rsi_level(self, rsi_value: float) -> str:
        """Classify RSI level"""
        if rsi_value >= 80:
            return "extremely_overbought"
        elif rsi_value >= 70:
            return "overbought"
        elif rsi_value >= 60:
            return "bullish"
        elif rsi_value >= 40:
            return "neutral"
        elif rsi_value >= 30:
            return "bearish"
        elif rsi_value >= 20:
            return "oversold"
        else:
            return "extremely_oversold"
    
    def _get_rsi_signal(self, rsi_value: float) -> str:
        """Get RSI trading signal"""
        if rsi_value >= 70:
            return "sell_signal"
        elif rsi_value <= 30:
            return "buy_signal"
        elif rsi_value >= 60:
            return "bullish_momentum"
        elif rsi_value <= 40:
            return "bearish_momentum"
        else:
            return "neutral"
    
    def _check_rsi_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> str:
        """Check for RSI divergence (simplified)"""
        if len(df) < 20:
            return "insufficient_data"
        
        recent_price_trend = df['Close'].iloc[-10:].is_monotonic_increasing
        recent_rsi_trend = rsi.iloc[-10:].is_monotonic_increasing
        
        if recent_price_trend and not recent_rsi_trend:
            return "bearish_divergence"
        elif not recent_price_trend and recent_rsi_trend:
            return "bullish_divergence"
        else:
            return "no_divergence"
    
    def _calculate_obv_trend(self, df: pd.DataFrame) -> str:
        """Calculate simplified OBV trend"""
        price_changes = df['Close'].diff()
        volume = df['Volume']
        
        obv = []
        obv_value = 0
        
        for i in range(len(price_changes)):
            if pd.isna(price_changes.iloc[i]):
                obv.append(obv_value)
            elif price_changes.iloc[i] > 0:
                obv_value += volume.iloc[i]
                obv.append(obv_value)
            elif price_changes.iloc[i] < 0:
                obv_value -= volume.iloc[i]
                obv.append(obv_value)
            else:
                obv.append(obv_value)
        
        # Trend of last 10 OBV values
        recent_obv = obv[-10:]
        if len(recent_obv) >= 5:
            return "rising" if recent_obv[-1] > recent_obv[-5] else "falling"
        return "neutral"
    
    def _identify_trend_pattern(self, closes: pd.Series) -> str:
        """Identify trend pattern from price series"""
        if len(closes) < 5:
            return "insufficient_data"
        
        recent_highs = closes.tail(5).max()
        recent_lows = closes.tail(5).min()
        older_highs = closes.head(5).max()
        older_lows = closes.head(5).min()
        
        if recent_highs > older_highs and recent_lows > older_lows:
            return "higher_highs_higher_lows"
        elif recent_highs < older_highs and recent_lows < older_lows:
            return "lower_highs_lower_lows"
        elif recent_highs > older_highs and recent_lows < older_lows:
            return "expanding_range"
        else:
            return "sideways"
    
    def _analyze_latest_candle(self, df: pd.DataFrame) -> Dict:
        """Analyze the latest candlestick"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        open_price = latest['Open']
        high_price = latest['High']
        low_price = latest['Low']
        close_price = latest['Close']
        
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        candle_type = "bullish" if close_price > open_price else "bearish" if close_price < open_price else "doji"
        
        return {
            'type': candle_type,
            'body_size_pct': float(body_size / open_price * 100),
            'range_size_pct': float(total_range / open_price * 100),
            'upper_shadow_pct': float((high_price - max(open_price, close_price)) / open_price * 100),
            'lower_shadow_pct': float((min(open_price, close_price) - low_price) / open_price * 100),
            'gap_from_prev': float((open_price - prev['Close']) / prev['Close'] * 100)
        }
    
    def calculate_composite_sentiment(self, indicators: Dict) -> Dict:
        """Calculate composite technical sentiment from all indicators"""
        scores = {}
        weights = self.config['technical_weights']
        
        # 1. Moving Averages Score
        ma_score = max(-1.0, min(1.0, indicators['moving_averages']['trend_score']))
        scores['trend_mas'] = ma_score
        
        # 2. RSI Score
        rsi_value = indicators['rsi_analysis']['primary_rsi']
        if rsi_value >= 70:
            rsi_score = -0.5  # Overbought
        elif rsi_value >= 60:
            rsi_score = 0.5   # Bullish momentum
        elif rsi_value <= 30:
            rsi_score = 0.5   # Oversold (potential bounce)
        elif rsi_value <= 40:
            rsi_score = -0.5  # Bearish momentum
        else:
            rsi_score = 0.0   # Neutral
        scores['momentum_rsi'] = rsi_score
        
        # 3. MACD Score
        macd_data = indicators['macd_analysis']
        if macd_data['recent_crossover'] == 'bullish':
            macd_score = 0.8
        elif macd_data['recent_crossover'] == 'bearish':
            macd_score = -0.8
        elif macd_data['crossover_status'] == 'bullish':
            macd_score = 0.4
        else:
            macd_score = -0.4
        scores['trend_macd'] = macd_score
        
        # 4. Volume Score
        vol_data = indicators['volume_analysis']
        if vol_data['volume_trend'] == 'high' and vol_data['price_volume_relationship'] == 'positive':
            vol_score = 0.6
        elif vol_data['volume_trend'] == 'high' and vol_data['price_volume_relationship'] == 'negative':
            vol_score = -0.6
        elif vol_data['obv_trend'] == 'rising':
            vol_score = 0.3
        elif vol_data['obv_trend'] == 'falling':
            vol_score = -0.3
        else:
            vol_score = 0.0
        scores['volume_trend'] = vol_score
        
        # 5. Bollinger Bands Score
        bb_data = indicators['bollinger_bands']
        bb_position = bb_data['bb_position']
        if bb_position > 0.8:
            bb_score = -0.3  # Near upper band
        elif bb_position < 0.2:
            bb_score = 0.3   # Near lower band
        else:
            bb_score = (bb_position - 0.5) * 0.4  # Linear scaling
        scores['bollinger'] = bb_score
        
        # 6. Support/Resistance Score
        sr_data = indicators['support_resistance']
        if sr_data['support_distance_pct'] and sr_data['support_distance_pct'] < 2:
            sr_score = 0.3  # Near support
        elif sr_data['resistance_distance_pct'] and sr_data['resistance_distance_pct'] < 2:
            sr_score = -0.3  # Near resistance
        else:
            sr_score = 0.0
        scores['support_resistance'] = sr_score
        
        # Calculate weighted composite score
        composite_score = sum(scores[key] * weights[key] for key in scores.keys())
        
        # Calculate confidence based on signal consistency
        signal_consistency = self._calculate_signal_consistency(scores)
        
        return {
            'individual_scores': scores,
            'composite_score': float(composite_score),
            'signal_consistency': float(signal_consistency),
            'score_breakdown': {key: {'score': scores[key], 'weight': weights[key], 'contribution': scores[key] * weights[key]} for key in scores.keys()}
        }
    
    def _calculate_signal_consistency(self, scores: Dict) -> float:
        """Calculate confidence based on how consistent the signals are"""
        score_values = list(scores.values())
        
        if not score_values:
            return 0.5
        
        # Count positive, negative, and neutral signals
        positive_signals = sum(1 for score in score_values if score > 0.1)
        negative_signals = sum(1 for score in score_values if score < -0.1)
        neutral_signals = sum(1 for score in score_values if -0.1 <= score <= 0.1)
        
        total_signals = len(score_values)
        
        # Higher consistency when signals agree
        if positive_signals >= total_signals * 0.7:
            consistency = 0.9  # Strong bullish consensus
        elif negative_signals >= total_signals * 0.7:
            consistency = 0.9  # Strong bearish consensus
        elif positive_signals >= total_signals * 0.5:
            consistency = 0.7  # Moderate bullish
        elif negative_signals >= total_signals * 0.5:
            consistency = 0.7  # Moderate bearish
        else:
            consistency = 0.5  # Mixed signals
        
        return consistency
    
    async def analyze_technical_sentiment(self) -> Dict:
        """Main analysis function for technical sentiment"""
        print("📈 Analyzing Nifty 50 Technical Sentiment...")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # Fetch Nifty data
            nifty_data = await self.fetch_nifty_data()
            
            if nifty_data.empty:
                return self._create_error_result("No Nifty 50 data available")
            
            # Calculate all technical indicators
            indicators = self.calculate_technical_indicators(nifty_data)
            
            # Calculate composite sentiment
            sentiment_analysis = self.calculate_composite_sentiment(indicators)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Build comprehensive result
            result = {
                'component_sentiment': sentiment_analysis['composite_score'],
                'component_confidence': sentiment_analysis['signal_consistency'],
                'component_weight': 0.30,
                'weighted_contribution': sentiment_analysis['composite_score'] * 0.30,
                
                'sub_indicators': {
                    'moving_averages': indicators['moving_averages'],
                    'rsi_analysis': indicators['rsi_analysis'],
                    'macd_analysis': indicators['macd_analysis'],
                    'volume_analysis': indicators['volume_analysis'],
                    'bollinger_bands': indicators['bollinger_bands'],
                    'support_resistance': indicators['support_resistance'],
                    'price_patterns': indicators['price_patterns']
                },
                
                'market_context': {
                    'current_price': float(nifty_data['Close'].iloc[-1]),
                    'previous_close': float(nifty_data['Close'].iloc[-2]),
                    'daily_change_pct': float((nifty_data['Close'].iloc[-1] - nifty_data['Close'].iloc[-2]) / nifty_data['Close'].iloc[-2] * 100),
                    'data_points': len(nifty_data),
                    'data_source': 'Yahoo_Finance_Nifty',
                    'analysis_period_days': self.config['lookback_days'],
                    'last_update': nifty_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                },
                
                'scoring_breakdown': sentiment_analysis['score_breakdown'],
                'technical_summary': {
                    'overall_trend': indicators['moving_averages']['overall_trend'],
                    'momentum_status': indicators['rsi_analysis']['rsi_signal'],
                    'macd_signal': indicators['macd_analysis']['crossover_status'],
                    'volume_confirmation': indicators['volume_analysis']['price_volume_relationship'],
                    'volatility_regime': 'high' if indicators['price_patterns']['volatility_10d'] > 20 else 'normal'
                },
                
                'execution_time_seconds': execution_time,
                'kpi_name': 'Technical_Analysis',
                'framework_weight': 0.30,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            }
            
            # Overall interpretation
            sentiment_score = result['component_sentiment']
            if sentiment_score > 0.6:
                interpretation = "🟢 STRONG BULLISH - Multiple technical confirmations"
            elif sentiment_score > 0.3:
                interpretation = "🟢 BULLISH - Net positive technical signals"
            elif sentiment_score > 0.1:
                interpretation = "🟡 MILDLY BULLISH - Slight upward bias"
            elif sentiment_score > -0.1:
                interpretation = "⚪ NEUTRAL - Mixed technical signals"
            elif sentiment_score > -0.3:
                interpretation = "🟡 MILDLY BEARISH - Slight downward bias"
            elif sentiment_score > -0.6:
                interpretation = "🔴 BEARISH - Net negative technical signals"
            else:
                interpretation = "🔴 STRONG BEARISH - Multiple bearish confirmations"
            
            result['interpretation'] = interpretation
            
            print(f"✅ Technical Analysis completed in {execution_time:.1f}s")
            print(f"📊 Sentiment: {sentiment_score:.3f} ({interpretation})")
            print(f"🎯 Confidence: {result['component_confidence']:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Technical analysis failed: {e}")
            return self._create_error_result(f"Analysis exception: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result with neutral default"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.30 * 0.5,  # Reduced weight for error
            'sub_indicators': {
                'error': {
                    'message': error_message,
                    'fallback_applied': True
                }
            },
            'market_context': {
                'data_source': 'Error_Fallback',
                'error': True
            },
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test function
async def test_technical_analyzer():
    """Test the technical analyzer"""
    analyzer = NiftyTechnicalAnalyzer()
    result = await analyzer.analyze_technical_sentiment()
    
    print("\n" + "="*60)
    print("NIFTY 50 TECHNICAL ANALYSIS TEST RESULTS")
    print("="*60)
    
    if result:
        print(f"Component Sentiment: {result.get('component_sentiment', 0):.3f}")
        print(f"Component Confidence: {result.get('component_confidence', 0):.1%}")
        print(f"Component Weight: {result.get('component_weight', 0):.0%}")
        print(f"Interpretation: {result.get('interpretation', 'N/A')}")
        
        # Save detailed JSON
        with open('nifty_technical_analysis.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print("\n✅ Detailed results saved to: nifty_technical_analysis.json")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_technical_analyzer())
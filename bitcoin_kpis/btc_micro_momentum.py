# BTC MICRO MOMENTUM ANALYZER - Modular Component
# =================================================
# Split from friend's btc_sentiment_composite_kraken_allinone.py
# KPI 1: Micro-Momentum (1m OHLC analysis)

import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any
import ssl
import certifi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class BitcoinMicroMomentumAnalyzer:
    """
    Bitcoin Micro-Momentum Analysis (1-minute OHLC)
    Features: ROC1/ROC5, MACD-hist z, RSI distance, Price-VWAP
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.df = pd.DataFrame()
        self.last_timestamp = None
        self.prev_score = None
        self.session = self._make_session()
        
    def _default_config(self) -> Dict:
        return {
            'pair': 'XBTUSD',
            'kraken_ohlc_url': 'https://api.kraken.com/0/public/OHLC',
            'alpha_smoothing': 0.20,
            'verify_ssl': True,
            'timeout': 15
        }
    
    def _make_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        retry = Retry(
            total=5, connect=5, read=5, backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504], 
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "btc-micro-momentum/1.0"})
        return session
    
    async def fetch_ohlc_data(self, since: Optional[int] = None) -> Tuple[pd.DataFrame, int]:
        """Fetch OHLC data from Kraken API"""
        params = {
            "pair": self.config['pair'],
            "interval": 1  # 1-minute intervals
        }
        if since:
            params["since"] = since
            
        try:
            response = self.session.get(
                self.config['kraken_ohlc_url'],
                params=params,
                timeout=self.config['timeout'],
                verify=self.config['verify_ssl']
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("error"):
                raise RuntimeError(f"Kraken API error: {data['error']}")
            
            result = data["result"]
            pair_key = next(k for k in result.keys() if k != "last")
            last_timestamp = int(result["last"])
            
            # Process OHLC data
            rows = []
            for candle in result[pair_key]:
                timestamp = datetime.fromtimestamp(int(candle[0]), tz=timezone.utc)
                rows.append({
                    "timestamp": timestamp,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "vwap": float(candle[5]),
                    "volume": float(candle[6])
                })
            
            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            return df, last_timestamp
            
        except Exception as e:
            print(f"❌ Error fetching OHLC data: {e}")
            return pd.DataFrame(), 0
    
    def _calculate_ema(self, series: pd.Series, span: int) -> pd.Series:
        """Calculate exponential moving average"""
        return series.ewm(span=span, adjust=False).mean()
    
    def _tanh_clip(self, x: float, scale: float, hard: float = 1.0) -> float:
        """Tanh clipping function"""
        v = math.tanh(x / (scale + 1e-12))
        return max(-hard, min(hard, v))
    
    def _ema_smoothing(self, prev: Optional[float], current: float, alpha: float) -> float:
        """EMA smoothing"""
        if prev is None:
            return current
        return alpha * current + (1 - alpha) * prev
    
    async def analyze_micro_momentum(self) -> Dict[str, Any]:
        """
        Main analysis function for micro-momentum
        Returns standardized result compatible with orchestrator
        """
        start_time = datetime.now()
        
        try:
            print("₿ Analyzing Bitcoin Micro-Momentum...")
            
            # Fetch OHLC data
            df, last_timestamp = await self.fetch_ohlc_data(since=self.last_timestamp)
            
            if df.empty:
                print("⚠️ No OHLC data available")
                return self._create_fallback_result()
            
            # Update internal data
            if not self.df.empty and not df.empty:
                self.df = pd.concat([self.df, df]).sort_index().drop_duplicates()
            else:
                self.df = df.copy()
            
            # Keep only recent data (last 400 minutes)
            cutoff = self.df.index[-1] - timedelta(minutes=400)
            self.df = self.df.loc[self.df.index >= cutoff]
            self.last_timestamp = last_timestamp
            
            # Ensure we have enough data
            if len(self.df) < 60:
                print("⚠️ Insufficient data for analysis")
                return self._create_fallback_result()
            
            # Calculate momentum indicators
            result = self._compute_momentum_score()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time_seconds'] = execution_time
            
            print(f"✅ Micro-momentum analysis completed in {execution_time:.1f}s")
            print(f"📊 Score: {result['component_sentiment']:.3f}")
            print(f"🎯 Confidence: {result['component_confidence']:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Micro-momentum analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _compute_momentum_score(self) -> Dict[str, Any]:
        """Compute momentum score from OHLC data"""
        close = self.df["close"].astype(float)
        price = float(close.iloc[-1])
        
        # 1. Calculate 1-minute return volatility (60-bar rolling)
        returns = close.pct_change()
        volatility = returns.rolling(60, min_periods=30).std().replace(0, np.nan).ffill()
        current_vol = float(volatility.iloc[-1]) if np.isfinite(volatility.iloc[-1]) else 1e-6
        
        # 2. Rate of Change features
        # ROC1: 1-minute return
        roc1 = (close.iloc[-1] / close.iloc[-2] - 1.0) if len(close) > 1 else 0.0
        f_roc1 = self._tanh_clip(roc1 / (current_vol + 1e-12), scale=3.0) / 3.0
        
        # ROC5: 5-minute return  
        roc5 = (close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) > 6 else 0.0
        f_roc5 = self._tanh_clip(roc5 / (math.sqrt(5) * (current_vol + 1e-12)), scale=3.0) / 3.0
        
        # 3. MACD histogram z-score
        macd_line = self._calculate_ema(close, 12) - self._calculate_ema(close, 26)
        signal_line = self._calculate_ema(macd_line, 9)
        macd_hist = macd_line - signal_line
        
        hist_window = macd_hist.iloc[-60:] if len(macd_hist) >= 60 else macd_hist
        hist_std = float(hist_window.std()) if len(hist_window) >= 20 else 0.0
        
        if hist_std > 0:
            macd_z = float(macd_hist.iloc[-1]) / (hist_std + 1e-12)
            f_macd = self._tanh_clip(macd_z, scale=3.0) / 3.0
        else:
            f_macd = 0.0
        
        # 4. RSI distance from 50
        delta = close.diff()
        up = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index).ewm(alpha=1/14, adjust=False).mean()
        down = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index).ewm(alpha=1/14, adjust=False).mean()
        rs = up / (down + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        rsi_current = float(rsi.iloc[-1])
        f_rsi = float(np.clip((rsi_current - 50.0) / 20.0, -1, 1))
        
        # 5. Price vs Intraday VWAP
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        intraday_data = self.df[self.df.index >= start_of_day]
        
        if not intraday_data.empty and intraday_data["volume"].sum() > 0:
            vwap_current = float(
                (intraday_data["vwap"] * intraday_data["volume"]).sum() / 
                (intraday_data["volume"].sum() + 1e-12)
            )
        else:
            vwap_current = price
        
        price_std = float(close.rolling(60, min_periods=30).std().iloc[-1] or 1e-6)
        f_vwap = self._tanh_clip((price - vwap_current) / (price_std + 1e-12), scale=2.0) / 2.0
        
        # 6. Combine features with weights
        weights = np.array([0.25, 0.20, 0.25, 0.20, 0.10])
        features = np.array([f_roc1, f_roc5, f_macd, f_rsi, f_vwap], dtype=float)
        
        raw_score = float(np.dot(weights, features) / weights.sum())
        raw_score = max(-1.0, min(1.0, raw_score))
        
        # Apply EMA smoothing
        smoothed_score = self._ema_smoothing(
            self.prev_score, 
            raw_score, 
            self.config['alpha_smoothing']
        )
        self.prev_score = smoothed_score
        
        # Calculate confidence based on data quality and signal strength
        confidence = self._calculate_confidence(current_vol, hist_std, len(self.df))
        
        return {
            'component_sentiment': float(smoothed_score),
            'component_confidence': float(confidence),
            'component_weight': 0.30,  # 30% weight in Bitcoin framework
            'weighted_contribution': float(smoothed_score * 0.30),
            
            'sub_indicators': {
                'roc_features': {
                    'f_roc1': round(f_roc1, 3),
                    'f_roc5': round(f_roc5, 3),
                    'roc1_pct': round(roc1 * 100, 4),
                    'roc5_pct': round(roc5 * 100, 4)
                },
                'momentum_indicators': {
                    'f_macd_hist': round(f_macd, 3),
                    'f_rsi_distance': round(f_rsi, 3),
                    'rsi_value': round(rsi_current, 2),
                    'macd_hist_z': round(macd_z if 'macd_z' in locals() else 0.0, 3)
                },
                'price_analysis': {
                    'f_vwap_distance': round(f_vwap, 3),
                    'current_price': round(price, 2),
                    'intraday_vwap': round(vwap_current, 2),
                    'volatility_1h': round(current_vol * 100, 4)
                }
            },
            
            'market_context': {
                'data_source': 'Kraken_OHLC_1min',
                'data_quality': 'high',
                'data_points': len(self.df),
                'last_update': self.df.index[-1].isoformat(),
                'price_currency': 'USD'
            },
            
            'kpi_name': 'Bitcoin_Micro_Momentum',
            'framework_weight': 0.30,
            'analysis_timestamp': datetime.now().isoformat(),
            'raw_score': round(raw_score, 3),
            'smoothed_score': round(smoothed_score, 3)
        }
    
    def _calculate_confidence(self, volatility: float, hist_std: float, data_points: int) -> float:
        """Calculate confidence score based on data quality"""
        # Data quantity confidence
        data_confidence = min(0.95, 0.5 + (data_points / 300) * 0.5)
        
        # Volatility confidence (prefer moderate volatility)
        if 0.001 <= volatility <= 0.01:  # 0.1% to 1% daily vol
            vol_confidence = 0.9
        elif volatility > 0.02:  # Very high volatility
            vol_confidence = 0.6
        else:  # Very low volatility
            vol_confidence = 0.7
        
        # Signal strength confidence
        signal_confidence = min(0.9, max(0.5, hist_std / 100))
        
        return (data_confidence + vol_confidence + signal_confidence) / 3
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when data is insufficient"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.30 * 0.5,  # Reduced weight
            'sub_indicators': {
                'fallback': {
                    'reason': 'insufficient_data',
                    'message': 'Using neutral fallback due to data issues'
                }
            },
            'market_context': {
                'data_source': 'Kraken_OHLC_Fallback',
                'data_quality': 'low'
            },
            'kpi_name': 'Bitcoin_Micro_Momentum',
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.3,
            'component_weight': 0.30 * 0.3,  # Heavily reduced weight
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
            'kpi_name': 'Bitcoin_Micro_Momentum',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test the component
if __name__ == "__main__":
    import asyncio
    
    async def test_micro_momentum():
        print("🧪 Testing Bitcoin Micro-Momentum Analyzer")
        print("=" * 50)
        
        analyzer = BitcoinMicroMomentumAnalyzer()
        result = await analyzer.analyze_micro_momentum()
        
        print(f"\n📊 Test Results:")
        print(f"Sentiment: {result.get('component_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('component_confidence', 'N/A')}")
        print(f"Status: {'✅ Success' if 'error' not in result else '❌ Error'}")
        
        return result
    
    # Run test
    asyncio.run(test_micro_momentum())
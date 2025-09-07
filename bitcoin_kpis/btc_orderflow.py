# BTC ORDERFLOW ANALYZER - Modular Component
# ===========================================
# Split from friend's btc_sentiment_composite_kraken_allinone.py  
# KPI 4: Orderflow & CVD (trade flow analysis)

import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from collections import deque, defaultdict
import math

class BitcoinOrderflowAnalyzer:
    """
    Bitcoin Orderflow & CVD Analysis (Trade Flow)
    Features: 5m imbalance, 15m CVD slope, 1m intensity booster
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.prev_score = None
        self.last_trade_id = None
        self.minute_buckets = defaultdict(lambda: {"buy": 0.0, "sell": 0.0, "trades": 0})
        self.trade_rate_history = deque(maxlen=60)  # 60 minutes of trade rate data
        self.session = self._make_session()
        
    def _default_config(self) -> Dict:
        return {
            'pair': 'XBTUSD',
            'kraken_trades_url': 'https://api.kraken.com/0/public/Trades',
            'alpha_smoothing': 0.20,
            'verify_ssl': True,
            'timeout': 15
        }
    
    def _make_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        session.headers.update({"User-Agent": "btc-orderflow/1.0"})
        return session
    
    def _tanh_clip(self, x: float, scale: float, hard: float = 1.0) -> float:
        """Tanh clipping function"""
        v = math.tanh(x / (scale + 1e-12))
        return max(-hard, min(hard, v))
    
    def _ema_smoothing(self, prev: Optional[float], current: float, alpha: float) -> float:
        """EMA smoothing"""
        if prev is None:
            return current
        return alpha * current + (1 - alpha) * prev
    
    def _floor_minute(self, timestamp: float) -> int:
        """Floor timestamp to minute boundary"""
        return int(timestamp - (timestamp % 60))
    
    async def fetch_recent_trades(self, since: Optional[int] = None) -> tuple[List, int]:
        """Fetch recent trades from Kraken"""
        try:
            params = {"pair": self.config['pair']}
            if since:
                params["since"] = since
            
            response = self.session.get(
                self.config['kraken_trades_url'],
                params=params,
                timeout=self.config['timeout'],
                verify=self.config['verify_ssl']
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("error"):
                raise RuntimeError(f"Kraken trades API error: {data['error']}")
            
            result = data["result"]
            pair_key = next(k for k in result.keys() if k != "last")
            last_id = int(result["last"])
            trades = result[pair_key]
            
            return trades, last_id
            
        except Exception as e:
            print(f"❌ Error fetching trades: {e}")
            return [], 0
    
    def _process_trades(self, trades: List):
        """Process trades into minute buckets"""
        for trade in trades:
            price = float(trade[0])
            volume = float(trade[1])
            timestamp = float(trade[2])
            side = trade[3]  # 'b' for buy, 's' for sell
            
            # Calculate notional value
            notional = price * volume
            
            # Floor to minute
            minute_key = self._floor_minute(timestamp)
            
            # Add to appropriate bucket
            if side == 'b':  # Buy (market buy = bullish)
                self.minute_buckets[minute_key]["buy"] += notional
            else:  # Sell (market sell = bearish)
                self.minute_buckets[minute_key]["sell"] += notional
            
            self.minute_buckets[minute_key]["trades"] += 1
        
        # Clean old buckets (keep only last 120 minutes)
        if self.minute_buckets:
            current_time = time.time()
            cutoff = self._floor_minute(current_time) - 60 * 120
            
            old_keys = [k for k in self.minute_buckets.keys() if k < cutoff]
            for key in old_keys:
                del self.minute_buckets[key]
    
    def _create_minute_series(self) -> pd.DataFrame:
        """Create minute-level time series from trade buckets"""
        if not self.minute_buckets:
            return pd.DataFrame()
        
        current_time = time.time()
        current_minute = self._floor_minute(current_time)
        
        # Create minute range (last 120 minutes)
        minutes = list(range(current_minute - 60 * 120 + 60, current_minute + 60, 60))
        
        rows = []
        for minute in minutes:
            bucket = self.minute_buckets.get(minute, {"buy": 0.0, "sell": 0.0, "trades": 0})
            timestamp = datetime.fromtimestamp(minute, tz=timezone.utc)
            
            rows.append({
                "timestamp": timestamp,
                "buy": bucket["buy"],
                "sell": bucket["sell"], 
                "trades": bucket["trades"]
            })
        
        df = pd.DataFrame(rows).set_index("timestamp")
        df["total"] = df["buy"] + df["sell"]
        df["delta"] = df["buy"] - df["sell"]
        
        return df
    
    async def analyze_orderflow(self) -> Dict[str, Any]:
        """
        Main analysis function for orderflow & CVD
        Returns standardized result compatible with orchestrator
        """
        start_time = datetime.now()
        
        try:
            print("₿ Analyzing Bitcoin Orderflow...")
            
            # Fetch recent trades
            trades, last_id = await self.fetch_recent_trades(since=self.last_trade_id)
            
            if trades:
                self._process_trades(trades)
                self.last_trade_id = last_id
            
            # Create minute series
            df = self._create_minute_series()
            
            if df.empty or len(df) < 16:  # Need at least 16 minutes of data
                print("⚠️ Insufficient trade data for analysis")
                return self._create_fallback_result()
            
            # Calculate orderflow metrics
            result = self._compute_orderflow_score(df)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time_seconds'] = execution_time
            
            print(f"✅ Orderflow analysis completed in {execution_time:.1f}s")
            print(f"📊 Score: {result['component_sentiment']:.3f}")
            print(f"🎯 Confidence: {result['component_confidence']:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Orderflow analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _compute_orderflow_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute orderflow sentiment score"""
        
        # 1. 5-minute imbalance
        df_5m = df.iloc[-5:] if len(df) >= 5 else df
        buy_5m = df_5m["buy"].sum()
        sell_5m = df_5m["sell"].sum()
        total_5m = buy_5m + sell_5m
        
        if total_5m > 0:
            imbalance_5m = (buy_5m - sell_5m) / total_5m
            f_imb5 = self._tanh_clip(imbalance_5m, scale=0.7)
        else:
            imbalance_5m = 0.0
            f_imb5 = 0.0
        
        # 2. 15-minute CVD (Cumulative Volume Delta) slope
        cvd = df["delta"].cumsum()
        
        if len(cvd) >= 16:
            cvd_slope_15m = float(cvd.iloc[-1] - cvd.iloc[-16])  # 15-minute change
        else:
            cvd_slope_15m = float(cvd.iloc[-1] - cvd.iloc[0]) if len(cvd) > 1 else 0.0
        
        # Calculate CVD scale from recent changes
        if len(cvd) >= 60:
            cvd_changes = np.diff(cvd.iloc[-60:].values)
            cvd_scale = float(np.median(np.abs(cvd_changes))) if len(cvd_changes) > 0 else 1.0
        else:
            cvd_scale = 1.0
        
        cvd_scale = max(cvd_scale, 1.0)  # Minimum scale
        
        f_cvd15 = self._tanh_clip(cvd_slope_15m / (15.0 * cvd_scale), scale=1.0)
        
        # 3. 1-minute intensity booster
        latest_minute = df.iloc[-1]
        buy_1m = float(latest_minute["buy"])
        sell_1m = float(latest_minute["sell"])
        total_1m = buy_1m + sell_1m
        trades_1m = float(latest_minute["trades"])
        
        if total_1m > 0:
            imbalance_1m = (buy_1m - sell_1m) / total_1m
        else:
            imbalance_1m = 0.0
        
        # Track trade rate for intensity calculation
        self.trade_rate_history.append(trades_1m)
        
        # Calculate z-score of trade rate
        if len(self.trade_rate_history) >= 10:
            mean_rate = float(np.mean(self.trade_rate_history))
            std_rate = float(np.std(self.trade_rate_history))
            
            if std_rate > 0:
                z_rate = (trades_1m - mean_rate) / std_rate
            else:
                z_rate = 0.0
        else:
            z_rate = 0.0
        
        # Intensity booster = imbalance * intensity z-score
        f_intensity = self._tanh_clip(imbalance_1m * z_rate, scale=2.0)
        
        # 4. Combine features
        raw_score = 0.45 * f_imb5 + 0.40 * f_cvd15 + 0.15 * f_intensity
        raw_score = max(-1.0, min(1.0, raw_score))
        
        # Apply EMA smoothing
        smoothed_score = self._ema_smoothing(
            self.prev_score,
            raw_score,
            self.config['alpha_smoothing']
        )
        self.prev_score = smoothed_score
        
        # Calculate confidence
        confidence = self._calculate_confidence(total_5m, total_1m, len(df))
        
        return {
            'component_sentiment': float(smoothed_score),
            'component_confidence': float(confidence),
            'component_weight': 0.20,  # 20% weight in Bitcoin framework
            'weighted_contribution': float(smoothed_score * 0.20),
            
            'sub_indicators': {
                'imbalance_analysis': {
                    'imbalance_5m': round(imbalance_5m, 3),
                    'buy_notional_5m': round(buy_5m, 2),
                    'sell_notional_5m': round(sell_5m, 2),
                    'total_notional_5m': round(total_5m, 2),
                    'f_imb5': round(f_imb5, 3),
                    'imbalance_interpretation': self._interpret_imbalance(imbalance_5m)
                },
                'cvd_analysis': {
                    'cvd_slope_15m': round(cvd_slope_15m, 2),
                    'cvd_scale': round(cvd_scale, 2),
                    'f_cvd15': round(f_cvd15, 3),
                    'cvd_current': round(float(cvd.iloc[-1]), 2) if len(cvd) > 0 else 0,
                    'cvd_interpretation': self._interpret_cvd_slope(cvd_slope_15m)
                },
                'intensity_analysis': {
                    'imbalance_1m': round(imbalance_1m, 3),
                    'trades_1m': int(trades_1m),
                    'trade_rate_z': round(z_rate, 2),
                    'f_intensity': round(f_intensity, 3),
                    'mean_trade_rate': round(float(np.mean(self.trade_rate_history)), 1) if len(self.trade_rate_history) > 0 else 0
                }
            },
            
            'market_context': {
                'data_source': 'Kraken_Trades',
                'data_quality': 'high',
                'minute_buckets': len(self.minute_buckets),
                'trade_rate_history_length': len(self.trade_rate_history),
                'data_timespan_minutes': len(df),
                'price_currency': 'USD'
            },
            
            'kpi_name': 'Bitcoin_Orderflow',
            'framework_weight': 0.20,
            'analysis_timestamp': datetime.now().isoformat(),
            'raw_score': round(raw_score, 3),
            'smoothed_score': round(smoothed_score, 3)
        }
    
    def _interpret_imbalance(self, imbalance: float) -> str:
        """Interpret imbalance level"""
        if imbalance > 0.3:
            return "Strong buy pressure - bullish flow"
        elif imbalance > 0.1:
            return "Moderate buy pressure - slight bullish"
        elif imbalance < -0.3:
            return "Strong sell pressure - bearish flow"
        elif imbalance < -0.1:
            return "Moderate sell pressure - slight bearish"
        else:
            return "Balanced flow - neutral"
    
    def _interpret_cvd_slope(self, slope: float) -> str:
        """Interpret CVD slope"""
        if slope > 50000:
            return "Strong cumulative buying"
        elif slope > 10000:
            return "Moderate cumulative buying"
        elif slope < -50000:
            return "Strong cumulative selling"
        elif slope < -10000:
            return "Moderate cumulative selling"
        else:
            return "Neutral cumulative flow"
    
    def _calculate_confidence(self, volume_5m: float, volume_1m: float, data_points: int) -> float:
        """Calculate confidence score based on data quality and volume"""
        # Data quantity confidence
        data_confidence = min(0.9, 0.3 + (data_points / 120) * 0.6)
        
        # Volume confidence (higher volume = higher confidence)
        if volume_5m > 1000000:  # > $1M in 5 minutes
            volume_confidence = 0.9
        elif volume_5m > 100000:  # > $100k in 5 minutes
            volume_confidence = 0.7
        else:
            volume_confidence = 0.5
        
        # Trade activity confidence
        if volume_1m > 50000:  # > $50k in 1 minute
            activity_confidence = 0.9
        elif volume_1m > 10000:  # > $10k in 1 minute  
            activity_confidence = 0.7
        else:
            activity_confidence = 0.5
        
        return (data_confidence + volume_confidence + activity_confidence) / 3
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when data is insufficient"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.20 * 0.5,  # Reduced weight
            'sub_indicators': {
                'fallback': {
                    'reason': 'insufficient_trade_data',
                    'message': 'Using neutral fallback due to trade data issues'
                }
            },
            'market_context': {
                'data_source': 'Kraken_Trades_Fallback',
                'data_quality': 'low'
            },
            'kpi_name': 'Bitcoin_Orderflow',
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.3,
            'component_weight': 0.20 * 0.3,  # Heavily reduced weight
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
            'kpi_name': 'Bitcoin_Orderflow',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test the component
if __name__ == "__main__":
    import asyncio
    
    async def test_orderflow():
        print("🧪 Testing Bitcoin Orderflow Analyzer")
        print("=" * 50)
        
        analyzer = BitcoinOrderflowAnalyzer()
        
        # First fetch some trades to populate buckets
        print("📡 Fetching initial trade data...")
        trades, last_id = await analyzer.fetch_recent_trades()
        if trades:
            analyzer._process_trades(trades)
            analyzer.last_trade_id = last_id
            print(f"✅ Processed {len(trades)} trades")
        
        # Now run analysis
        result = await analyzer.analyze_orderflow()
        
        print(f"\n📊 Test Results:")
        print(f"Sentiment: {result.get('component_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('component_confidence', 'N/A')}")
        print(f"Status: {'✅ Success' if 'error' not in result.get('sub_indicators', {}) else '❌ Error'}")
        
        # Display detailed metrics
        sub_indicators = result.get('sub_indicators', {})
        if 'imbalance_analysis' in sub_indicators:
            imb = sub_indicators['imbalance_analysis']
            print(f"\n⚖️ Imbalance Analysis:")
            print(f"  5m Imbalance: {imb.get('imbalance_5m', 'N/A')}")
            print(f"  Total Volume 5m: ${imb.get('total_notional_5m', 'N/A'):,.0f}")
            print(f"  Interpretation: {imb.get('imbalance_interpretation', 'N/A')}")
        
        if 'cvd_analysis' in sub_indicators:
            cvd = sub_indicators['cvd_analysis']
            print(f"\n📈 CVD Analysis:")
            print(f"  15m Slope: {cvd.get('cvd_slope_15m', 'N/A')}")
            print(f"  Current CVD: {cvd.get('cvd_current', 'N/A')}")
            print(f"  Interpretation: {cvd.get('cvd_interpretation', 'N/A')}")
        
        return result
    
    # Run test
    asyncio.run(test_orderflow())
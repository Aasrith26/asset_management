# REIT TECHNICAL MOMENTUM - Original Logic Preserved
# ==================================================
# Split from friend's original code, keeping exact same logic
# KPI1: Technical Momentum (daily) — from Yahoo OHLC

import os, json, math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from typing import Dict, Any, Tuple

class REITTechnicalMomentumAnalyzer:
    """
    REIT Technical Momentum Analysis - ORIGINAL LOGIC PRESERVED
    KPI1: Technical Momentum (daily) — from Yahoo OHLC
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        return {
            'reit_symbols': ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "BRIGADE.NS", "SOBHA.NS"],  # Keep original symbols
            'lookback_days': 500
        }
    
    def tanh_clip(self, x, scale=1.0, hard=1.0):
        """Original tanh_clip function - unchanged"""
        v = math.tanh(x / (scale + 1e-12))
        return max(-hard, min(hard, v))
    
    async def fetch_reit_data(self) -> Tuple[str, pd.DataFrame]:
        """Fetch REIT data - original logic preserved"""
        tickers = self.config['reit_symbols']
        data = None
        chosen = None
        end = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        start = end - timedelta(days=self.config['lookback_days'] + 30)
        
        for t in tickers:
            try:
                print(f"📡 Trying to fetch {t}...")
                df = yf.download(t, start=start.isoformat(), end=end.isoformat(),
                               interval="1d", progress=False, auto_adjust=False)
                if df is not None and not df.empty and "Close" in df.columns:
                    data = df.copy()
                    chosen = t
                    break
            except Exception:
                continue
        
        if data is None:
            raise RuntimeError("Could not fetch Mindspace REIT from Yahoo Finance. Tried: " + ", ".join(tickers))
        
        # Original data processing - unchanged
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.rename(columns=str.lower)
        data = data.dropna(subset=["close", "high", "low", "volume"])
        data.index = pd.to_datetime(data.index, utc=True)
        
        return chosen, data
    
    def kpi_technical_momentum(self, price_df: pd.DataFrame):
        """ORIGINAL KPI1 LOGIC - UNCHANGED FROM FRIEND'S CODE"""
        close = price_df["close"].astype(float).copy().dropna()
        ret = close.pct_change()
        vol = ret.rolling(63, min_periods=30).std().ffill().replace(0, np.nan).ffill()
        vol_now = float(vol.iloc[-1]) if len(vol) else 1e-4
        
        # ROC20 feature
        roc20 = (close / close.shift(20) - 1.0)
        f_roc = self.tanh_clip((float(roc20.iloc[-1]) / (math.sqrt(20) * (vol_now + 1e-12))), scale=1.5)
        
        # RSI14 feature
        delta = close.diff()
        up = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index).ewm(alpha=1/14, adjust=False).mean()
        dn = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index).ewm(alpha=1/14, adjust=False).mean()
        rs = up / (dn + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        f_rsi = float(np.clip((float(rsi.iloc[-1]) - 50.0) / 25.0, -1.0, 1.0))
        
        # EMA alignment & slope
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        align = 1.0 if ema20.iloc[-1] > ema50.iloc[-1] else -1.0
        slope = float((ema20.iloc[-1] - ema20.iloc[-5]) / (5 * (vol_now + 1e-12)))
        f_ema = self.tanh_clip(0.6*align + 0.4*self.tanh_clip(slope, 1.0), scale=1.0)
        
        # Original weighting and combination - unchanged
        w = np.array([0.4, 0.3, 0.3])
        f = np.array([f_roc, f_rsi, f_ema], dtype=float)
        s_raw = float(np.dot(w, f) / w.sum())
        s = max(-1.0, min(1.0, s_raw))
        
        # Original diagnostics format - unchanged
        diag = {
            "score": round(s, 3),
            "features": {"f_roc20": round(f_roc, 3), "f_rsi14": round(f_rsi, 3), "f_emaAlignSlope": round(f_ema, 3)},
            "metrics": {"price": round(float(close.iloc[-1]), 2),
                        "roc20": round(float(roc20.iloc[-1]), 4),
                        "rsi14": round(float(rsi.iloc[-1]), 2)}
        }
        
        return s, diag
    
    async def analyze_technical_momentum(self) -> Dict[str, Any]:
        """
        Main analysis function - wrapper around original logic
        """
        start_time = datetime.now()
        
        try:
            print("🏢 Analyzing REIT Technical Momentum...")
            
            # Fetch data using original logic
            symbol, price_df = await self.fetch_reit_data()
            
            # Apply original KPI calculation
            sentiment_score, diagnostics = self.kpi_technical_momentum(price_df)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to standardized format for orchestrator
            result = {
                'component_sentiment': float(sentiment_score),
                'component_confidence': 0.8,  # High confidence for original logic
                'component_weight': 0.35,  # 35% weight as in original
                'weighted_contribution': float(sentiment_score * 0.35),
                
                'sub_indicators': {
                    'original_diagnostics': diagnostics,
                    'technical_features': diagnostics['features'],
                    'price_metrics': diagnostics['metrics']
                },
                
                'market_context': {
                    'symbol': symbol,
                    'data_source': 'Yahoo_Finance_Original_Logic',
                    'data_quality': 'high',
                    'price_currency': 'INR'
                },
                
                'execution_time_seconds': execution_time,
                'kpi_name': 'REIT_Technical_Momentum_Original',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Technical momentum analysis completed in {execution_time:.1f}s")
            print(f"📊 Score: {result['component_sentiment']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Technical momentum analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.3,
            'component_weight': 0.35 * 0.3,
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
            'kpi_name': 'REIT_Technical_Momentum_Original',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test the component with original logic
if __name__ == "__main__":
    import asyncio
    
    async def test_original_technical():
        print("🧪 Testing REIT Technical Momentum - Original Logic")
        print("=" * 50)
        
        analyzer = REITTechnicalMomentumAnalyzer()
        result = await analyzer.analyze_technical_momentum()
        
        print(f"\n📊 Test Results:")
        print(f"Sentiment: {result.get('component_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('component_confidence', 'N/A')}")
        print(f"Status: {'✅ Success' if 'error' not in result.get('sub_indicators', {}) else '❌ Error'}")
        
        # Show original diagnostics
        original_diag = result.get('sub_indicators', {}).get('original_diagnostics', {})
        if original_diag:
            print(f"\n📈 Original Diagnostics:")
            print(f"  Score: {original_diag.get('score', 'N/A')}")
            print(f"  Features: {original_diag.get('features', {})}")
            print(f"  Metrics: {original_diag.get('metrics', {})}")
        
        return result
    
    # Run test
    asyncio.run(test_original_technical())
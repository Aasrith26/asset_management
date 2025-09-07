# REIT ACCUMULATION FLOW - Original Logic Preserved
# =================================================
# Split from friend's original code, keeping exact same logic
# KPI3: Accumulation & Ownership Flow — Up/Down Vol, A/D slope, OBV slope

import os, json, math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from typing import Dict, Any, Tuple

class REITAccumulationFlowAnalyzer:
    """
    REIT Accumulation Flow Analysis - ORIGINAL LOGIC PRESERVED
    KPI3: Accumulation & Ownership Flow — Up/Down Vol, A/D slope, OBV slope
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        return {
            'reit_symbols': ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "BRIGADE.NS", "SOBHA.NS"],
  # Keep original symbols
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
                print(f"📡 Trying to fetch {t} for flow analysis...")
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
    
    def kpi_flow(self, price_df: pd.DataFrame):
        """ORIGINAL KPI3 LOGIC - UNCHANGED FROM FRIEND'S CODE"""
        df = price_df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low  = df["low"].astype(float)
        vol  = df["volume"].astype(float)
        
        # Up/Down Volume (20D) - original logic
        prev_close = close.shift(1)
        up_mask = close > prev_close
        down_mask = ~up_mask
        up_vol_20 = vol.where(up_mask, 0.0).rolling(20, min_periods=5).sum()
        down_vol_20 = vol.where(down_mask, 0.0).rolling(20, min_periods=5).sum()
        ratio = (up_vol_20.iloc[-1] + 1.0) / (down_vol_20.iloc[-1] + 1.0)
        f_updown = self.tanh_clip((ratio - 1.0) / 0.3, scale=1.0)
        
        # Accumulation/Distribution (A/D) line and 20D slope - original logic
        clv = ((close - low) - (high - close)) / (high - low + 1e-12)  # [-1, +1]
        ad  = (clv * vol).fillna(0.0).cumsum()
        if len(ad) >= 21:
            ad_slope = float(ad.iloc[-1] - ad.iloc[-21])
        else:
            ad_slope = 0.0
        ad_scale = float(np.median(np.abs(np.diff(ad.values[-60:]))) if len(ad) >= 3 else 1.0) or 1.0
        f_ad = self.tanh_clip(ad_slope / (20.0 * ad_scale), scale=1.0)
        
        # On-Balance Volume (OBV) slope - original logic
        sign = np.sign((close - prev_close).fillna(0.0))
        obv = (vol * sign).cumsum()
        if len(obv) >= 21:
            obv_slope = float(obv.iloc[-1] - obv.iloc[-21])
        else:
            obv_slope = 0.0
        vol_scale = float(np.median(vol.tail(60))) or 1.0
        f_obv = self.tanh_clip(obv_slope / (20.0 * vol_scale), scale=1.0)
        
        # Combine — emphasize Up/Down and A/D; OBV as supplemental - original weighting
        S_raw = 0.45*f_updown + 0.35*f_ad + 0.20*f_obv
        S = max(-1.0, min(1.0, S_raw))
        
        # Original diagnostics format - unchanged
        diag = {
            "score": round(S, 3),
            "features": {"f_updown": round(f_updown, 3), "f_ad_slope": round(f_ad, 3), "f_obv_slope": round(f_obv, 3)},
            "metrics": {
                "up_down_ratio_20d": round(ratio, 3),
                "ad_slope_20d": round(ad_slope, 2),
                "obv_slope_20d": round(obv_slope, 2)
            }
        }
        
        return S, diag
    
    async def analyze_accumulation_flow(self) -> Dict[str, Any]:
        """
        Main analysis function - wrapper around original logic
        """
        start_time = datetime.now()
        
        try:
            print("🏢 Analyzing REIT Accumulation Flow...")
            
            # Fetch data using original logic
            symbol, price_df = await self.fetch_reit_data()
            
            # Apply original KPI calculation
            sentiment_score, diagnostics = self.kpi_flow(price_df)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to standardized format for orchestrator
            result = {
                'component_sentiment': float(sentiment_score),
                'component_confidence': 0.8,  # High confidence for original logic
                'component_weight': 0.20,  # 20% weight as in original
                'weighted_contribution': float(sentiment_score * 0.20),
                
                'sub_indicators': {
                    'original_diagnostics': diagnostics,
                    'flow_features': diagnostics['features'],
                    'flow_metrics': diagnostics['metrics']
                },
                
                'market_context': {
                    'symbol': symbol,
                    'data_source': 'Yahoo_Finance_Original_Logic',
                    'data_quality': 'high',
                    'price_currency': 'INR'
                },
                
                'execution_time_seconds': execution_time,
                'kpi_name': 'REIT_Accumulation_Flow_Original',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Accumulation flow analysis completed in {execution_time:.1f}s")
            print(f"📊 Score: {result['component_sentiment']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Accumulation flow analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.3,
            'component_weight': 0.20 * 0.3,
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
            'kpi_name': 'REIT_Accumulation_Flow_Original',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test the component with original logic
if __name__ == "__main__":
    import asyncio
    
    async def test_original_flow():
        print("🧪 Testing REIT Accumulation Flow - Original Logic")
        print("=" * 50)
        
        analyzer = REITAccumulationFlowAnalyzer()
        result = await analyzer.analyze_accumulation_flow()
        
        print(f"\n📊 Test Results:")
        print(f"Sentiment: {result.get('component_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('component_confidence', 'N/A')}")
        print(f"Status: {'✅ Success' if 'error' not in result.get('sub_indicators', {}) else '❌ Error'}")
        
        # Show original diagnostics
        original_diag = result.get('sub_indicators', {}).get('original_diagnostics', {})
        if original_diag:
            print(f"\n📊 Original Diagnostics:")
            print(f"  Score: {original_diag.get('score', 'N/A')}")
            print(f"  Features: {original_diag.get('features', {})}")
            print(f"  Metrics: {original_diag.get('metrics', {})}")
        
        return result
    
    # Run test
    asyncio.run(test_original_flow())
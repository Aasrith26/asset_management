# REIT YIELD SPREAD - Original Logic Preserved  
# ==============================================
# Split from friend's original code, keeping exact same logic
# KPI2: Yield-Spread vs India 10Y G-Sec — 10Y via clean-price index proxy + TE snapshot anchor

import os, json, math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from typing import Dict, Any, Tuple, Optional

class REITYieldSpreadAnalyzer:
    """
    REIT Yield Spread Analysis - ORIGINAL LOGIC PRESERVED
    KPI2: Yield-Spread vs India 10Y G-Sec — 10Y via clean-price index proxy + TE snapshot anchor
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.session = self._make_session()
        
    def _default_config(self) -> Dict:
        return {
            'reit_symbols': ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "BRIGADE.NS", "SOBHA.NS"],
  # Keep original symbols
            'lookback_days_price': 500,
            'lookback_days_yield': 500,
            # Original constants from friend's code
            'in10y_clean_ticker': "NIFTYGS10YRCLN.NS",   # NIFTY 10 Yr Benchmark G-Sec (Clean Price) Index
            'assumed_mod_duration': 6.5,                 # conservative modified duration for 10Y benchmark
            'te_api_key': os.environ.get("TE_API_KEY", "guest:guest"),
            'te_base': "https://api.tradingeconomics.com",
            'use_true_te_history': False  # set True ONLY if you have a paid TE key with history access
        }
    
    def _make_session(self):
        """Original session creation - unchanged"""
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "mindspace-reit-four-kpi/1.0"})
        return s
    
    def tanh_clip(self, x, scale=1.0, hard=1.0):
        """Original tanh_clip function - unchanged"""
        v = math.tanh(x / (scale + 1e-12))
        return max(-hard, min(hard, v))
    
    async def fetch_mindspace_history(self):
        """Original fetch logic - unchanged"""
        tickers = self.config['reit_symbols']
        data = None
        chosen = None
        end = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        start = end - timedelta(days=self.config['lookback_days_price'] + 30)
        
        for t in tickers:
            try:
                print(f"📡 Trying to fetch REIT data for {t}...")
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
        
        # Dividends (TTM) - original logic
        div = None
        try:
            d = yf.Ticker(chosen).dividends
            if d is not None and not d.empty:
                div = d.copy()
        except Exception:
            pass
        
        return chosen, data, div
    
    def get_in10y_snapshot(self):
        """ORIGINAL India 10Y snapshot logic - unchanged"""
        # Option A: Markets snapshot — often allowed on guest
        try:
            url = f"{self.config['te_base']}/markets/government-bond/india:10y?c={self.config['te_api_key']}&format=json"
            r = self.session.get(url, timeout=20)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, list) and len(j) > 0:
                    y = j[0].get("Close") or j[0].get("Last") or j[0].get("Price")
                    if y is not None:
                        y = float(y)
                        return y/100.0 if y > 3 else y
        except Exception:
            pass
        
        # Option B: Indicators snapshot — may or may not work on guest
        try:
            url = f"{self.config['te_base']}/indicators/india/government-bond-10y?c={self.config['te_api_key']}&format=json"
            r = self.session.get(url, timeout=20)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, list) and len(j) > 0:
                    y = j[-1].get("Value")
                    if y is not None:
                        y = float(y)
                        return y/100.0 if y > 3 else y
        except Exception:
            pass
        
        return None
    
    def build_in10y_proxy_history(self, anchor_yield_frac: float, lookback_days: int = 400):
        """ORIGINAL 10Y proxy logic - unchanged"""
        end = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        start = end - timedelta(days=lookback_days + 30)
        
        idx_df = yf.download(self.config['in10y_clean_ticker'], 
                           start=start.isoformat(), end=end.isoformat(),
                           interval="1d", progress=False, auto_adjust=False)
        
        if idx_df is None or idx_df.empty:
            raise RuntimeError(f"Could not fetch {self.config['in10y_clean_ticker']} from Yahoo.")
        
        if isinstance(idx_df.columns, pd.MultiIndex):
            idx_df.columns = idx_df.columns.get_level_values(0)
        idx_df = idx_df.rename(columns=str.lower).dropna(subset=["close"])
        idx_df.index = pd.to_datetime(idx_df.index, utc=True)
        
        px = idx_df["close"].astype(float)
        ret = px.pct_change().fillna(0.0)
        y = pd.Series(index=px.index, dtype=float)
        y.iloc[-1] = anchor_yield_frac if anchor_yield_frac is not None else 0.065
        
        # Original duration calculation - unchanged
        inv_dur = 1.0 / self.config['assumed_mod_duration']
        for i in range(len(px)-1, 0, -1):
            r = float(ret.iloc[i])
            y.iloc[i-1] = y.iloc[i] + (r * inv_dur) * (-1.0)  # price up => yield down
        
        y = y.clip(lower=0.03, upper=0.10)
        y.name = "yield_frac"
        return y
    
    def kpi_yield_spread(self, price_df: pd.DataFrame, dividends: pd.Series, in10y_series: pd.Series):
        """ORIGINAL KPI2 LOGIC - UNCHANGED FROM FRIEND'S CODE"""
        close = price_df["close"].astype(float).dropna()
        px = float(close.iloc[-1])
        
        # TTM distribution yield from dividends (last 365d) - original logic
        y_end = close.index[-1]
        y_start = y_end - pd.Timedelta(days=365)
        div_ttm = 0.0
        
        if dividends is not None and not dividends.empty:
            dv = dividends.copy()
            dv.index = pd.to_datetime(dv.index, utc=True)
            dv = dv[(dv.index > y_start) & (dv.index <= y_end)]
            div_ttm = float(dv.sum())
        
        reit_yield = (div_ttm / px) if px > 0 else 0.0  # fraction
        
        # Align 10Y series to price date - original logic
        in10y = in10y_series.copy().dropna()
        in10y = in10y[in10y.index <= y_end]
        if in10y.empty:
            raise RuntimeError("India 10Y yield series empty/not aligned.")
        yld10 = float(in10y.iloc[-1])  # fraction
        
        spread = reit_yield - yld10  # fraction
        
        # Build spread history (last ~252d) for robust scaling - original logic
        idx = close.index[-252:] if len(close) >= 252 else close.index
        ttm_series = pd.Series(reit_yield, index=idx)  # carry last known TTM between distributions
        
        if dividends is not None and not dividends.empty:
            dv = dividends.copy()
            dv.index = pd.to_datetime(dv.index, utc=True)
            for ts in idx:
                dv1y = dv[(dv.index > ts - pd.Timedelta(days=365)) & (dv.index <= ts)].sum() if not dv.empty else 0.0
                px_t = float(close.loc[:ts].iloc[-1])
                ttm_series.loc[ts] = (dv1y / px_t) if px_t > 0 else np.nan
            ttm_series = ttm_series.fillna(method="ffill").fillna(reit_yield)
        
        in10y_aligned = in10y.reindex(idx, method="ffill")
        spread_hist = ttm_series - in10y_aligned
        dspread = spread_hist.diff().dropna()
        scale = float(np.median(np.abs(dspread))) if len(dspread) >= 10 else 0.001
        scale = max(scale, 0.0005)
        
        # Original scoring - unchanged
        s_raw = self.tanh_clip(spread / (5 * scale), scale=1.0)
        s = max(-1.0, min(1.0, s_raw))
        
        # Original diagnostics format - unchanged
        diag = {
            "score": round(s, 3),
            "features": {
                "spread_frac": round(spread, 4),
                "reit_yield_ttm": round(reit_yield, 4),
                "gsec10y_yield": round(yld10, 4)
            },
            "metrics": {"div_ttm": round(div_ttm, 4), "price": round(px, 2), "scale_proxy": round(scale, 6)}
        }
        
        return s, diag
    
    async def analyze_yield_spread(self) -> Dict[str, Any]:
        """
        Main analysis function - wrapper around original logic
        """
        start_time = datetime.now()
        
        try:
            print("🏢 Analyzing REIT Yield Spread...")
            
            # 1) Mindspace price & dividends - original logic
            ticker, px_df, div_series = await self.fetch_mindspace_history()
            
            # 2) India 10Y series - original logic
            y_snap = self.get_in10y_snapshot()  # fractional (e.g., 0.065) or None
            
            if self.config['use_true_te_history']:
                # Only with paid key - not implemented for guest
                raise RuntimeError("True TE history requires paid key")
            else:
                in10y_series = self.build_in10y_proxy_history(y_snap, lookback_days=self.config['lookback_days_yield'])
            
            # 3) Apply original KPI calculation
            sentiment_score, diagnostics = self.kpi_yield_spread(px_df, div_series, in10y_series)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to standardized format for orchestrator
            result = {
                'component_sentiment': float(sentiment_score),
                'component_confidence': 0.8,  # High confidence for original logic
                'component_weight': 0.30,  # 30% weight as in original
                'weighted_contribution': float(sentiment_score * 0.30),
                
                'sub_indicators': {
                    'original_diagnostics': diagnostics,
                    'yield_features': diagnostics['features'],
                    'yield_metrics': diagnostics['metrics']
                },
                
                'market_context': {
                    'symbol': ticker,
                    'data_source': 'Yahoo_Finance_TradingEconomics_Original_Logic',
                    'yield_anchor': 'TE_snapshot' if y_snap is not None else 'assumed_6.5%',
                    'data_quality': 'high',
                    'price_currency': 'INR'
                },
                
                'execution_time_seconds': execution_time,
                'kpi_name': 'REIT_Yield_Spread_Original',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Yield spread analysis completed in {execution_time:.1f}s")
            print(f"📊 Score: {result['component_sentiment']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Yield spread analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.3,
            'component_weight': 0.30 * 0.3,
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
            'kpi_name': 'REIT_Yield_Spread_Original',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test the component with original logic
if __name__ == "__main__":
    import asyncio
    
    async def test_original_yield():
        print("🧪 Testing REIT Yield Spread - Original Logic")
        print("=" * 50)
        
        analyzer = REITYieldSpreadAnalyzer()
        result = await analyzer.analyze_yield_spread()
        
        print(f"\n📊 Test Results:")
        print(f"Sentiment: {result.get('component_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('component_confidence', 'N/A')}")
        print(f"Status: {'✅ Success' if 'error' not in result.get('sub_indicators', {}) else '❌ Error'}")
        
        # Show original diagnostics
        original_diag = result.get('sub_indicators', {}).get('original_diagnostics', {})
        if original_diag:
            print(f"\n💰 Original Diagnostics:")
            print(f"  Score: {original_diag.get('score', 'N/A')}")
            print(f"  Features: {original_diag.get('features', {})}")
            print(f"  Metrics: {original_diag.get('metrics', {})}")
        
        return result
    
    # Run test
    asyncio.run(test_original_yield())
# REIT LIQUIDITY RISK - Original Logic Preserved
# ===============================================
# Split from friend's original code, keeping exact same logic
# KPI4: Liquidity & Slippage Risk — Range-based spread proxy & Amihud illiquidity

import os, json, math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from typing import Dict, Any, Tuple

class REITLiquidityRiskAnalyzer:
    """
    REIT Liquidity Risk Analysis - ORIGINAL LOGIC PRESERVED
    KPI4: Liquidity & Slippage Risk — Range-based spread proxy & Amihud illiquidity
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
    
    def robust_z(self, latest_value: float, series: pd.Series):
        """ORIGINAL robust_z function - unchanged"""
        s = pd.Series(series).dropna()
        if len(s) < 10:
            return 0.0
        # exclude last observation if it equals latest_value
        s_hist = s.iloc[:-1] if float(s.iloc[-1]) == float(latest_value) and len(s) > 1 else s
        med = float(s_hist.median())
        mad = float((s_hist - med).abs().median())
        denom = 1.4826 * mad if mad > 1e-12 else max(abs(med) * 0.05, 1e-6)  # fallback
        return (latest_value - med) / denom
    
    async def fetch_reit_data(self) -> Tuple[str, pd.DataFrame]:
        """Fetch REIT data - original logic preserved"""
        tickers = self.config['reit_symbols']
        data = None
        chosen = None
        end = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        start = end - timedelta(days=self.config['lookback_days'] + 30)
        
        for t in tickers:
            try:
                print(f"📡 Trying to fetch {t} for liquidity analysis...")
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
    
    def kpi_liquidity(self, price_df: pd.DataFrame):
        """ORIGINAL KPI4 LOGIC - UNCHANGED FROM FRIEND'S CODE"""
        df = price_df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low  = df["low"].astype(float)
        vol  = df["volume"].astype(float)
        
        # Range-based effective spread proxy (today vs 60D median) - original logic
        rng_today = float((high.iloc[-1] - low.iloc[-1]) / (close.iloc[-1] + 1e-12))
        rng_60 = ((high - low) / (close + 1e-12)).tail(60)
        med_rng_60 = float(np.median(rng_60)) if len(rng_60) else rng_today
        rel_rng = (rng_today / (med_rng_60 + 1e-12)) - 1.0
        f_spread = self.tanh_clip(-rel_rng / 0.25, scale=1.0)  # narrower than usual => positive
        
        # Amihud illiquidity (|ret| / value_traded) — 20D average, then robust-z vs 60D - original logic
        ret = close.pct_change()
        value_traded = (close * vol).replace(0, np.nan)
        amihud = (ret.abs() / (value_traded + 1e-12)).rolling(20, min_periods=10).mean()
        amihud_60 = amihud.tail(60).dropna()
        
        if len(amihud_60) >= 10:
            z_ami = self.robust_z(float(amihud.iloc[-1]), amihud_60)
        else:
            z_ami = 0.0
        
        f_amihud = self.tanh_clip(-z_ami / 2.0, scale=1.0)     # lower illiq than usual => positive
        
        # Original combination - unchanged
        S_raw = 0.5*f_spread + 0.5*f_amihud
        S = max(-1.0, min(1.0, S_raw))
        
        # Original diagnostics format - unchanged
        diag = {
            "score": round(S, 3),
            "features": {"f_spread_proxy": round(f_spread, 3), "f_amihud": round(f_amihud, 3)},
            "metrics": {
                "range_today": round(rng_today, 5),
                "median_range_60d": round(med_rng_60, 5),
                "amihud_20d": round(float(amihud.iloc[-1]) if pd.notna(amihud.iloc[-1]) else 0.0, 10)
            }
        }
        
        return S, diag
    
    async def analyze_liquidity_risk(self) -> Dict[str, Any]:
        """
        Main analysis function - wrapper around original logic
        """
        start_time = datetime.now()
        
        try:
            print("🏢 Analyzing REIT Liquidity Risk...")
            
            # Fetch data using original logic
            symbol, price_df = await self.fetch_reit_data()
            
            # Apply original KPI calculation
            sentiment_score, diagnostics = self.kpi_liquidity(price_df)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to standardized format for orchestrator
            result = {
                'component_sentiment': float(sentiment_score),
                'component_confidence': 0.8,  # High confidence for original logic
                'component_weight': 0.15,  # 15% weight as in original
                'weighted_contribution': float(sentiment_score * 0.15),
                
                'sub_indicators': {
                    'original_diagnostics': diagnostics,
                    'liquidity_features': diagnostics['features'],
                    'liquidity_metrics': diagnostics['metrics']
                },
                
                'market_context': {
                    'symbol': symbol,
                    'data_source': 'Yahoo_Finance_Original_Logic',
                    'data_quality': 'high',
                    'price_currency': 'INR'
                },
                
                'execution_time_seconds': execution_time,
                'kpi_name': 'REIT_Liquidity_Risk_Original',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Liquidity risk analysis completed in {execution_time:.1f}s")
            print(f"📊 Score: {result['component_sentiment']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Liquidity risk analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.3,
            'component_weight': 0.15 * 0.3,
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
            'kpi_name': 'REIT_Liquidity_Risk_Original',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test the component with original logic
if __name__ == "__main__":
    import asyncio
    
    async def test_original_liquidity():
        print("🧪 Testing REIT Liquidity Risk - Original Logic")
        print("=" * 50)
        
        analyzer = REITLiquidityRiskAnalyzer()
        result = await analyzer.analyze_liquidity_risk()
        
        print(f"\n📊 Test Results:")
        print(f"Sentiment: {result.get('component_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('component_confidence', 'N/A')}")
        print(f"Status: {'✅ Success' if 'error' not in result.get('sub_indicators', {}) else '❌ Error'}")
        
        # Show original diagnostics
        original_diag = result.get('sub_indicators', {}).get('original_diagnostics', {})
        if original_diag:
            print(f"\n💧 Original Diagnostics:")
            print(f"  Score: {original_diag.get('score', 'N/A')}")
            print(f"  Features: {original_diag.get('features', {})}")
            print(f"  Metrics: {original_diag.get('metrics', {})}")
        
        return result
    
    # Run test
    asyncio.run(test_original_liquidity())
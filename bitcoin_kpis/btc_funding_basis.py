# BTC FUNDING & BASIS ANALYZER - Modular Component
# =================================================
# Split from friend's btc_sentiment_composite_kraken_allinone.py
# KPI 2: Funding/Basis & Positioning (perpetual futures)

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from collections import deque
import math

class BitcoinFundingBasisAnalyzer:
    """
    Bitcoin Funding & Basis Analysis (Perpetual Futures)
    Features: Funding rate contrarian, basis stretch, OI momentum
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.prev_score = None
        self.oi_history = deque(maxlen=90)  # 90 minutes of OI data
        self.session = self._make_session()
        
    def _default_config(self) -> Dict:
        return {
            'spot_pair': 'XBTUSD',
            'perp_symbol': 'PI_XBTUSD',
            'kraken_ticker_url': 'https://api.kraken.com/0/public/Ticker',
            'kraken_futures_url': 'https://futures.kraken.com/derivatives/api/v3/tickers',
            'alpha_smoothing': 0.20,
            'verify_ssl': True,
            'timeout': 15
        }
    
    def _make_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        session.headers.update({"User-Agent": "btc-funding-basis/1.0"})
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
    
    async def fetch_spot_price(self) -> float:
        """Fetch spot price from Kraken"""
        try:
            response = self.session.get(
                self.config['kraken_ticker_url'],
                params={"pair": self.config['spot_pair']},
                timeout=self.config['timeout'],
                verify=self.config['verify_ssl']
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("error"):
                raise RuntimeError(f"Kraken spot API error: {data['error']}")
            
            result = data["result"]
            pair_key = next(iter(result.keys()))
            spot_price = float(result[pair_key]["c"][0])  # Last price
            
            return spot_price
            
        except Exception as e:
            print(f"❌ Error fetching spot price: {e}")
            return 0.0
    
    async def fetch_perpetual_data(self) -> Dict[str, float]:
        """Fetch perpetual futures data from Kraken"""
        try:
            response = self.session.get(
                self.config['kraken_futures_url'],
                timeout=self.config['timeout'],
                verify=self.config['verify_ssl']
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("result") != "success":
                raise RuntimeError(f"Kraken futures API error: {data}")
            
            # Find the perpetual contract
            tickers = data.get("tickers", [])
            perp_data = None
            
            for ticker in tickers:
                if ticker.get("symbol") == self.config['perp_symbol']:
                    perp_data = ticker
                    break
            
            if not perp_data:
                raise RuntimeError(f"Perpetual symbol {self.config['perp_symbol']} not found")
            
            # Extract relevant data
            mark_price = float(perp_data.get("markPrice") or perp_data.get("last") or 0)
            open_interest = float(perp_data.get("openInterest") or 0)
            
            # Funding rate (8h rate, convert to hourly)
            funding_rate_8h = perp_data.get("fundingRate") or perp_data.get("predFundingRate") or 0
            funding_rate = float(funding_rate_8h) / 8 if funding_rate_8h else 0.0
            
            return {
                "mark_price": mark_price,
                "open_interest": open_interest,
                "funding_rate": funding_rate,
                "funding_rate_8h": float(funding_rate_8h) if funding_rate_8h else 0.0
            }
            
        except Exception as e:
            print(f"❌ Error fetching perpetual data: {e}")
            return {
                "mark_price": 0.0,
                "open_interest": 0.0,
                "funding_rate": 0.0,
                "funding_rate_8h": 0.0
            }
    
    async def analyze_funding_basis(self) -> Dict[str, Any]:
        """
        Main analysis function for funding & basis
        Returns standardized result compatible with orchestrator
        """
        start_time = datetime.now()
        
        try:
            print("₿ Analyzing Bitcoin Funding & Basis...")
            
            # Fetch spot and perpetual data
            spot_price = await self.fetch_spot_price()
            perp_data = await self.fetch_perpetual_data()
            
            if spot_price <= 0 or perp_data["mark_price"] <= 0:
                print("⚠️ Invalid price data received")
                return self._create_fallback_result()
            
            # Calculate indicators
            result = self._compute_funding_basis_score(
                spot_price,
                perp_data["mark_price"],
                perp_data["open_interest"],
                perp_data["funding_rate_8h"]
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time_seconds'] = execution_time
            
            print(f"✅ Funding & basis analysis completed in {execution_time:.1f}s")
            print(f"📊 Score: {result['component_sentiment']:.3f}")
            print(f"🎯 Confidence: {result['component_confidence']:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Funding & basis analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _compute_funding_basis_score(self, spot: float, mark: float, oi: float, funding: float) -> Dict[str, Any]:
        """Compute funding and basis sentiment score"""
        
        # Track open interest history
        self.oi_history.append(oi)
        
        # 1. Funding rate contrarian signal
        # High positive funding = too much long pressure = bearish
        # High negative funding = too much short pressure = bullish
        f_funding = -self._tanh_clip(funding, scale=0.0005)  # ~5 bps scale
        
        # 2. Basis stretch signal
        # basis = (mark_price - spot_price) / spot_price
        basis = (mark - spot) / (spot + 1e-12)
        f_basis = -self._tanh_clip(basis, scale=0.001)  # ~10 bps scale
        
        # 3. Open Interest build momentum (5-minute change)
        if len(self.oi_history) >= 6:  # At least 6 minutes of data
            oi_change_5m = self.oi_history[-1] - self.oi_history[-6]
        else:
            oi_change_5m = 0.0
        
        # Calculate OI z-score if we have enough history
        if len(self.oi_history) >= 30:
            oi_diffs = np.diff(np.array(list(self.oi_history), dtype=float))
            oi_std = float(np.std(oi_diffs[-60:])) if len(oi_diffs) >= 2 else 0.0
        else:
            oi_std = 0.0
        
        if oi_std > 0:
            oi_z = oi_change_5m / (oi_std + 1e-12)
            oi_z = max(-3.0, min(3.0, oi_z))
            f_oi = oi_z / 3.0  # Normalize to [-1, 1]
        else:
            f_oi = 0.0
        
        # 4. Combine base features
        raw_score = 0.4 * f_funding + 0.3 * f_basis + 0.3 * f_oi
        
        # 5. Apply crowding adjustments
        # If funding is very high and OI is building, reduce bullishness
        if funding > 0.001 and oi_change_5m > 0:  # High funding + OI increase
            crowding_penalty = min(0.30, funding / 0.003)
            raw_score -= crowding_penalty
        elif funding < -0.0005 and oi_change_5m > 0:  # Negative funding + OI increase
            crowding_bonus = min(0.20, abs(funding) / 0.002)
            raw_score += crowding_bonus
        
        # Clip final score
        raw_score = max(-1.0, min(1.0, raw_score))
        
        # Apply EMA smoothing
        smoothed_score = self._ema_smoothing(
            self.prev_score,
            raw_score,
            self.config['alpha_smoothing']
        )
        self.prev_score = smoothed_score
        
        # Calculate confidence
        confidence = self._calculate_confidence(funding, basis, len(self.oi_history))
        
        return {
            'component_sentiment': float(smoothed_score),
            'component_confidence': float(confidence),
            'component_weight': 0.30,  # 30% weight in Bitcoin framework
            'weighted_contribution': float(smoothed_score * 0.30),
            
            'sub_indicators': {
                'funding_analysis': {
                    'funding_rate_8h': round(funding, 6),
                    'funding_rate_annual_pct': round(funding * 365 * 3 * 100, 2),
                    'f_funding_contrarian': round(f_funding, 3),
                    'funding_interpretation': self._interpret_funding(funding)
                },
                'basis_analysis': {
                    'spot_price': round(spot, 2),
                    'mark_price': round(mark, 2),
                    'basis_bps': round(basis * 10000, 2),
                    'f_basis_stretch': round(f_basis, 3),
                    'basis_interpretation': self._interpret_basis(basis)
                },
                'oi_analysis': {
                    'current_oi': round(oi, 2),
                    'oi_change_5m': round(oi_change_5m, 2),
                    'oi_z_score': round(f_oi * 3, 2),
                    'f_oi_momentum': round(f_oi, 3)
                }
            },
            
            'market_context': {
                'data_source': 'Kraken_Spot_Futures',
                'data_quality': 'high',
                'oi_history_length': len(self.oi_history),
                'funding_frequency': '8_hours',
                'price_currency': 'USD'
            },
            
            'kpi_name': 'Bitcoin_Funding_Basis',
            'framework_weight': 0.30,
            'analysis_timestamp': datetime.now().isoformat(),
            'raw_score': round(raw_score, 3),
            'smoothed_score': round(smoothed_score, 3)
        }
    
    def _interpret_funding(self, funding: float) -> str:
        """Interpret funding rate level"""
        if funding > 0.002:
            return "Very high positive - extreme long bias"
        elif funding > 0.001:
            return "High positive - strong long bias"
        elif funding > 0.0005:
            return "Moderate positive - slight long bias"
        elif funding < -0.001:
            return "High negative - strong short bias"
        elif funding < -0.0005:
            return "Moderate negative - slight short bias"
        else:
            return "Neutral - balanced positioning"
    
    def _interpret_basis(self, basis: float) -> str:
        """Interpret basis level"""
        basis_bps = basis * 10000
        if basis_bps > 20:
            return "Premium - futures expensive vs spot"
        elif basis_bps > 5:
            return "Slight premium - mild futures premium"
        elif basis_bps < -20:
            return "Discount - futures cheap vs spot"
        elif basis_bps < -5:
            return "Slight discount - mild futures discount"
        else:
            return "Fair value - futures aligned with spot"
    
    def _calculate_confidence(self, funding: float, basis: float, oi_history_length: int) -> float:
        """Calculate confidence score based on data quality and market conditions"""
        # Data history confidence
        data_confidence = min(0.9, 0.3 + (oi_history_length / 90) * 0.6)
        
        # Market condition confidence - prefer clear signals
        funding_clarity = min(0.9, abs(funding) / 0.001 * 0.5 + 0.4)
        basis_clarity = min(0.9, abs(basis) / 0.002 * 0.3 + 0.6)
        
        # Combined confidence
        combined_confidence = (data_confidence + funding_clarity + basis_clarity) / 3
        
        return max(0.3, min(0.95, combined_confidence))
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when data is insufficient"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.30 * 0.5,  # Reduced weight
            'sub_indicators': {
                'fallback': {
                    'reason': 'invalid_price_data',
                    'message': 'Using neutral fallback due to price data issues'
                }
            },
            'market_context': {
                'data_source': 'Kraken_Futures_Fallback',
                'data_quality': 'low'
            },
            'kpi_name': 'Bitcoin_Funding_Basis',
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
            'kpi_name': 'Bitcoin_Funding_Basis',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test the component
if __name__ == "__main__":
    import asyncio
    
    async def test_funding_basis():
        print("🧪 Testing Bitcoin Funding & Basis Analyzer")
        print("=" * 50)
        
        analyzer = BitcoinFundingBasisAnalyzer()
        result = await analyzer.analyze_funding_basis()
        
        print(f"\n📊 Test Results:")
        print(f"Sentiment: {result.get('component_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('component_confidence', 'N/A')}")
        print(f"Status: {'✅ Success' if 'error' not in result.get('sub_indicators', {}) else '❌ Error'}")
        
        # Display detailed metrics
        sub_indicators = result.get('sub_indicators', {})
        if 'funding_analysis' in sub_indicators:
            funding = sub_indicators['funding_analysis']
            print(f"\n💰 Funding Analysis:")
            print(f"  8h Rate: {funding.get('funding_rate_8h', 'N/A')}")
            print(f"  Annual %: {funding.get('funding_rate_annual_pct', 'N/A')}")
            print(f"  Interpretation: {funding.get('funding_interpretation', 'N/A')}")
        
        if 'basis_analysis' in sub_indicators:
            basis = sub_indicators['basis_analysis']
            print(f"\n📏 Basis Analysis:")
            print(f"  Spot: ${basis.get('spot_price', 'N/A')}")
            print(f"  Mark: ${basis.get('mark_price', 'N/A')}")
            print(f"  Basis (bps): {basis.get('basis_bps', 'N/A')}")
        
        return result
    
    # Run test
    asyncio.run(test_funding_basis())
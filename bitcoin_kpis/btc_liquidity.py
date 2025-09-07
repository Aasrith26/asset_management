# BTC LIQUIDITY ANALYZER - Modular Component
# ==========================================
# Split from friend's btc_sentiment_composite_kraken_allinone.py
# KPI 3: Liquidity & Slippage (order book analysis)

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import math

class BitcoinLiquidityAnalyzer:
    """
    Bitcoin Liquidity & Slippage Analysis (Order Book)
    Features: Spread improvement, slippage improvement, depth imbalance
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.prev_score = None
        self.spread_history = deque(maxlen=60)  # 60 minutes of spread data
        self.impact_history = deque(maxlen=60)  # 60 minutes of impact data
        self.session = self._make_session()
        
    def _default_config(self) -> Dict:
        return {
            'pair': 'XBTUSD',
            'kraken_depth_url': 'https://api.kraken.com/0/public/Depth',
            'depth_levels': 200,
            'standard_notional': 250000,  # $250k standard trade size
            'band_bps': 25,  # 25 bps depth band
            'alpha_smoothing': 0.20,
            'verify_ssl': True,
            'timeout': 15
        }
    
    def _make_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        session.headers.update({"User-Agent": "btc-liquidity/1.0"})
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
    
    def _bps(self, x: float) -> float:
        """Convert to basis points"""
        return 10000.0 * x
    
    async def fetch_order_book(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Fetch order book from Kraken"""
        try:
            params = {
                "pair": self.config['pair'],
                "count": self.config['depth_levels']
            }
            
            response = self.session.get(
                self.config['kraken_depth_url'],
                params=params,
                timeout=self.config['timeout'],
                verify=self.config['verify_ssl']
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("error"):
                raise RuntimeError(f"Kraken depth API error: {data['error']}")
            
            result = data["result"]
            pair_key = next(iter(result.keys()))
            
            # Parse bids and asks
            bids = [(float(price), float(volume)) for price, volume, _ in result[pair_key]["bids"]]
            asks = [(float(price), float(volume)) for price, volume, _ in result[pair_key]["asks"]]
            
            # Sort order book
            bids.sort(key=lambda x: -x[0])  # Highest bid first
            asks.sort(key=lambda x: x[0])   # Lowest ask first
            
            return bids, asks
            
        except Exception as e:
            print(f"❌ Error fetching order book: {e}")
            return [], []
    
    def _calculate_mid_spread(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate mid price and spread in bps"""
        if not bids or not asks:
            return 0.0, 0.0
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = 0.5 * (best_bid + best_ask)
        spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 0.0
        
        return mid_price, self._bps(spread)
    
    def _calculate_slippage(self, side: str, book: List[Tuple[float, float]], 
                           mid: float, notional: float) -> float:
        """Calculate slippage for given trade size"""
        if not book or mid <= 0:
            return 0.0
        
        remaining_notional = notional
        total_cost = 0.0
        total_quantity = 0.0
        
        for price, size in book:
            if remaining_notional <= 0:
                break
            
            line_value = price * size
            take_value = min(remaining_notional, line_value)
            quantity = take_value / price
            
            total_cost += price * quantity
            total_quantity += quantity
            remaining_notional -= take_value
        
        if total_quantity <= 0:
            return 0.0
        
        vwap = total_cost / total_quantity
        
        if side == "buy":
            return self._bps((vwap - mid) / mid)
        else:
            return self._bps((mid - vwap) / mid)
    
    def _calculate_depth_in_band(self, bids: List[Tuple[float, float]], 
                                asks: List[Tuple[float, float]], 
                                mid: float, band_bps: float) -> Tuple[float, float]:
        """Calculate notional depth within price band"""
        if mid <= 0:
            return 0.0, 0.0
        
        upper_bound = mid * (1 + band_bps / 10000.0)
        lower_bound = mid * (1 - band_bps / 10000.0)
        
        bid_notional = sum(price * size for price, size in bids if price >= lower_bound)
        ask_notional = sum(price * size for price, size in asks if price <= upper_bound)
        
        return bid_notional, ask_notional
    
    async def analyze_liquidity(self) -> Dict[str, Any]:
        """
        Main analysis function for liquidity & slippage
        Returns standardized result compatible with orchestrator
        """
        start_time = datetime.now()
        
        try:
            print("₿ Analyzing Bitcoin Liquidity...")
            
            # Fetch order book
            bids, asks = await self.fetch_order_book()
            
            if not bids or not asks:
                print("⚠️ No order book data available")
                return self._create_fallback_result()
            
            # Calculate liquidity metrics
            result = self._compute_liquidity_score(bids, asks)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time_seconds'] = execution_time
            
            print(f"✅ Liquidity analysis completed in {execution_time:.1f}s")
            print(f"📊 Score: {result['component_sentiment']:.3f}")
            print(f"🎯 Confidence: {result['component_confidence']:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Liquidity analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _compute_liquidity_score(self, bids: List[Tuple[float, float]], 
                                asks: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Compute liquidity sentiment score"""
        
        # 1. Calculate basic metrics
        mid_price, spread_bps = self._calculate_mid_spread(bids, asks)
        
        # 2. Calculate slippage for standard trade
        buy_slippage = self._calculate_slippage("buy", asks, mid_price, self.config['standard_notional'])
        sell_slippage = self._calculate_slippage("sell", bids, mid_price, self.config['standard_notional'])
        impact_bps = 0.5 * (buy_slippage + sell_slippage)
        
        # 3. Calculate depth imbalance
        bid_notional, ask_notional = self._calculate_depth_in_band(
            bids, asks, mid_price, self.config['band_bps']
        )
        imbalance = (bid_notional - ask_notional) / ((bid_notional + ask_notional) + 1e-12)
        
        # 4. Track historical medians for relative analysis
        if spread_bps > 0:
            self.spread_history.append(spread_bps)
        if impact_bps > 0:
            self.impact_history.append(impact_bps)
        
        # Calculate historical benchmarks
        median_spread = float(np.median(self.spread_history)) if len(self.spread_history) >= 10 else spread_bps or 1.0
        median_impact = float(np.median(self.impact_history)) if len(self.impact_history) >= 10 else impact_bps or 1.0
        
        # 5. Relative improvement features
        # Spread improvement (negative = tighter spreads = positive signal)
        rel_spread = (spread_bps / (median_spread + 1e-9)) - 1.0
        f_spread = self._tanh_clip(-rel_spread, scale=0.25)
        
        # Impact improvement (negative = lower impact = positive signal)
        rel_impact = (impact_bps / (median_impact + 1e-9)) - 1.0
        f_impact = self._tanh_clip(-rel_impact, scale=0.30)
        
        # 6. Depth imbalance feature
        f_imbalance = self._tanh_clip(imbalance, scale=0.5)
        
        # 7. Combine features
        raw_score = 0.35 * f_spread + 0.45 * f_impact + 0.20 * f_imbalance
        raw_score = max(-1.0, min(1.0, raw_score))
        
        # Apply EMA smoothing
        smoothed_score = self._ema_smoothing(
            self.prev_score,
            raw_score,
            self.config['alpha_smoothing']
        )
        self.prev_score = smoothed_score
        
        # Calculate confidence
        confidence = self._calculate_confidence(spread_bps, impact_bps, len(bids), len(asks))
        
        return {
            'component_sentiment': float(smoothed_score),
            'component_confidence': float(confidence),
            'component_weight': 0.20,  # 20% weight in Bitcoin framework
            'weighted_contribution': float(smoothed_score * 0.20),
            
            'sub_indicators': {
                'spread_analysis': {
                    'current_spread_bps': round(spread_bps, 2),
                    'median_spread_bps': round(median_spread, 2),
                    'relative_spread': round(rel_spread, 3),
                    'f_spread_improvement': round(f_spread, 3),
                    'spread_interpretation': self._interpret_spread(spread_bps)
                },
                'slippage_analysis': {
                    'buy_slippage_bps': round(buy_slippage, 2),
                    'sell_slippage_bps': round(sell_slippage, 2),
                    'average_impact_bps': round(impact_bps, 2),
                    'median_impact_bps': round(median_impact, 2),
                    'f_impact_improvement': round(f_impact, 3),
                    'standard_trade_size': self.config['standard_notional']
                },
                'depth_analysis': {
                    'bid_notional': round(bid_notional, 2),
                    'ask_notional': round(ask_notional, 2),
                    'depth_imbalance': round(imbalance, 3),
                    'f_imbalance': round(f_imbalance, 3),
                    'band_bps': self.config['band_bps']
                }
            },
            
            'market_context': {
                'data_source': 'Kraken_OrderBook',
                'data_quality': 'high',
                'mid_price': round(mid_price, 2),
                'depth_levels_analyzed': len(bids) + len(asks),
                'spread_history_length': len(self.spread_history),
                'impact_history_length': len(self.impact_history),
                'price_currency': 'USD'
            },
            
            'kpi_name': 'Bitcoin_Liquidity',
            'framework_weight': 0.20,
            'analysis_timestamp': datetime.now().isoformat(),
            'raw_score': round(raw_score, 3),
            'smoothed_score': round(smoothed_score, 3)
        }
    
    def _interpret_spread(self, spread_bps: float) -> str:
        """Interpret spread level"""
        if spread_bps < 1.0:
            return "Very tight - excellent liquidity"
        elif spread_bps < 2.5:
            return "Tight - good liquidity"
        elif spread_bps < 5.0:
            return "Moderate - average liquidity"
        elif spread_bps < 10.0:
            return "Wide - poor liquidity"
        else:
            return "Very wide - very poor liquidity"
    
    def _calculate_confidence(self, spread: float, impact: float, bid_levels: int, ask_levels: int) -> float:
        """Calculate confidence score based on data quality"""
        # Order book depth confidence
        depth_confidence = min(0.9, 0.4 + (min(bid_levels, ask_levels) / 100) * 0.5)
        
        # Spread quality confidence (prefer reasonable spreads)
        if 0.5 <= spread <= 20:  # 0.5 to 20 bps is reasonable
            spread_confidence = 0.9
        elif spread > 50:  # Very wide spreads
            spread_confidence = 0.5
        else:  # Very tight or zero spreads (suspicious)
            spread_confidence = 0.7
        
        # Impact reasonableness confidence
        if 1 <= impact <= 50:  # 1 to 50 bps impact is reasonable
            impact_confidence = 0.9
        elif impact > 100:  # Very high impact
            impact_confidence = 0.6
        else:
            impact_confidence = 0.7
        
        # History length confidence
        history_confidence = min(0.9, 0.5 + (len(self.spread_history) / 60) * 0.4)
        
        return (depth_confidence + spread_confidence + impact_confidence + history_confidence) / 4
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when data is insufficient"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.20 * 0.5,  # Reduced weight
            'sub_indicators': {
                'fallback': {
                    'reason': 'no_order_book_data',
                    'message': 'Using neutral fallback due to order book data issues'
                }
            },
            'market_context': {
                'data_source': 'Kraken_OrderBook_Fallback',
                'data_quality': 'low'
            },
            'kpi_name': 'Bitcoin_Liquidity',
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
            'kpi_name': 'Bitcoin_Liquidity',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test the component
if __name__ == "__main__":
    import asyncio
    
    async def test_liquidity():
        print("🧪 Testing Bitcoin Liquidity Analyzer")
        print("=" * 50)
        
        analyzer = BitcoinLiquidityAnalyzer()
        result = await analyzer.analyze_liquidity()
        
        print(f"\n📊 Test Results:")
        print(f"Sentiment: {result.get('component_sentiment', 'N/A')}")
        print(f"Confidence: {result.get('component_confidence', 'N/A')}")
        print(f"Status: {'✅ Success' if 'error' not in result.get('sub_indicators', {}) else '❌ Error'}")
        
        # Display detailed metrics
        sub_indicators = result.get('sub_indicators', {})
        if 'spread_analysis' in sub_indicators:
            spread = sub_indicators['spread_analysis']
            print(f"\n📏 Spread Analysis:")
            print(f"  Current Spread: {spread.get('current_spread_bps', 'N/A')} bps")
            print(f"  Median Spread: {spread.get('median_spread_bps', 'N/A')} bps")
            print(f"  Interpretation: {spread.get('spread_interpretation', 'N/A')}")
        
        if 'slippage_analysis' in sub_indicators:
            slippage = sub_indicators['slippage_analysis']
            print(f"\n💧 Slippage Analysis:")
            print(f"  Buy Slippage: {slippage.get('buy_slippage_bps', 'N/A')} bps")
            print(f"  Sell Slippage: {slippage.get('sell_slippage_bps', 'N/A')} bps")
            print(f"  Average Impact: {slippage.get('average_impact_bps', 'N/A')} bps")
        
        return result
    
    # Run test
    asyncio.run(test_liquidity())
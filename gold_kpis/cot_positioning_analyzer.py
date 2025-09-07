# PROFESSIONAL POSITIONING (COT) SENTIMENT ANALYZER v1.0
# ========================================================
# Integrates with existing enterprise gold sentiment system
# CFTC Commitment of Traders data analysis for smart money positioning

import pandas as pd
import numpy as np
import requests
import json
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import hashlib
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class CachedCOTData:
    data: any
    timestamp: datetime
    ttl_minutes: int = 480  # 8 hours for COT data (updated weekly)
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(minutes=self.ttl_minutes)

class ProfessionalPositioningAnalyzer:
    """
    Professional Positioning (COT) Sentiment Analyzer
    - CFTC Commitment of Traders data analysis
    - Commercial vs Speculative positioning
    - Smart money sentiment detection
    - 15% weight in composite gold sentiment
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.cache = {}
        self.cot_contract = "GC"  # Gold futures contract code
        print("Professional Positioning (COT) Analyzer v1.0 initialized")
        print("🏦 Smart money sentiment tracking enabled")
    
    def _default_config(self) -> Dict:
        return {
            'request_timeout': 10,
            'cache_ttl_minutes': 480,  # 8 hours (COT updates weekly)
            'confidence_threshold': 0.6,
            'lookback_weeks': 12,  # 3 months of COT data
            'smoothing_window': 3,  # 3-week moving average
        }
    
    def _get_cache_key(self, source: str, params: str = "") -> str:
        """Generate cache key for COT data storage"""
        return hashlib.md5(f"cot_{source}_{params}".encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[any]:
        """Retrieve cached COT data if valid"""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if not cached.is_expired():
                return cached.data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: any, ttl_minutes: int = None):
        """Cache COT data with TTL"""
        ttl = ttl_minutes or self.config['cache_ttl_minutes']
        self.cache[cache_key] = CachedCOTData(data, datetime.now(), ttl)

    async def fetch_cftc_cot_data(self) -> Dict:
        """Fetch CFTC Commitment of Traders data for gold futures"""
        cache_key = self._get_cache_key("cftc_cot", "gold")
        cached_result = self._get_cached_data(cache_key)
        
        if cached_result:
            print("📊 COT Data: Using cached result")
            return cached_result
        
        print("📊 Fetching CFTC Commitment of Traders data...")
        
        try:
            # Primary method: Direct CFTC API
            result = await self._fetch_cftc_direct()
            if result and result.get('confidence', 0) > 0.5:
                self._cache_data(cache_key, result)
                return result
            
            # Fallback method: Scraped COT data
            result = await self._fetch_cot_alternative()
            if result:
                self._cache_data(cache_key, result, 240)  # Shorter cache for fallback
                return result
            
            # Final fallback: Estimated positioning based on price action
            return await self._estimate_positioning_proxy()
            
        except Exception as e:
            print(f"COT data fetch error: {e}")
            return await self._estimate_positioning_proxy()
    
    async def _fetch_cftc_direct(self) -> Dict:
        """Fetch data directly from CFTC"""
        try:
            # CFTC publishes COT data weekly on Tuesdays for positions as of previous Tuesday
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config['request_timeout']))
            
            # COT Legacy report for Gold futures (contract market code: 001602)
            url = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
            params = {
                "$where": "contract_market_code='001602'",
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": str(self.config['lookback_weeks'])
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    await session.close()
                    
                    if data:
                        return await self._process_cftc_data(data)
            
            await session.close()
            return None
            
        except Exception as e:
            print(f"Direct CFTC fetch error: {e}")
            return None
    
    async def _process_cftc_data(self, raw_data: List[Dict]) -> Dict:
        """Process raw CFTC COT data into sentiment signals"""
        try:
            if not raw_data:
                return None
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(raw_data)
            
            # Key COT fields for gold futures
            required_fields = [
                'report_date_as_yyyy_mm_dd',
                'comm_positions_long_all',      # Commercial long positions
                'comm_positions_short_all',     # Commercial short positions  
                'noncomm_positions_long_all',   # Non-commercial long positions
                'noncomm_positions_short_all',  # Non-commercial short positions
                'open_interest_all'             # Total open interest
            ]
            
            # Check if we have required data
            if not all(field in df.columns for field in required_fields):
                return None
            
            # Calculate positioning metrics
            df['comm_long'] = pd.to_numeric(df['comm_positions_long_all'], errors='coerce')
            df['comm_short'] = pd.to_numeric(df['comm_positions_short_all'], errors='coerce')
            df['noncomm_long'] = pd.to_numeric(df['noncomm_positions_long_all'], errors='coerce')
            df['noncomm_short'] = pd.to_numeric(df['noncomm_positions_short_all'], errors='coerce')
            df['open_interest'] = pd.to_numeric(df['open_interest_all'], errors='coerce')
            
            # Calculate net positions
            df['comm_net'] = df['comm_long'] - df['comm_short']
            df['noncomm_net'] = df['noncomm_long'] - df['noncomm_short']
            
            # Calculate as percentage of open interest
            df['comm_net_pct'] = (df['comm_net'] / df['open_interest']) * 100
            df['noncomm_net_pct'] = (df['noncomm_net'] / df['open_interest']) * 100
            
            # Sort by date (most recent first)
            df['report_date'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
            df = df.sort_values('report_date', ascending=False)
            
            # Current positioning (most recent data)
            latest = df.iloc[0]
            current_comm_net_pct = latest['comm_net_pct']
            
            # Historical context (3-month average and trend)
            if len(df) >= 4:
                avg_comm_net_pct = df['comm_net_pct'].head(12).mean()  # 3-month average
                trend_comm_net = df['comm_net_pct'].head(4).mean() - df['comm_net_pct'].tail(4).mean()
            else:
                avg_comm_net_pct = current_comm_net_pct
                trend_comm_net = 0
            
            # Calculate sentiment based on commercial positioning
            sentiment = self._calculate_cot_sentiment(current_comm_net_pct, avg_comm_net_pct, trend_comm_net)
            confidence = self._calculate_cot_confidence(df, current_comm_net_pct)
            
            return {
                'source': 'CFTC_COT_Direct',
                'sentiment': float(sentiment),
                'confidence': float(confidence),
                'weight': 0.15,  # 15% weight in composite
                'description': f'COT: Commercials {current_comm_net_pct:+.1f}% net (vs {avg_comm_net_pct:+.1f}% avg)',
                'metadata': {
                    'current_comm_net_pct': float(current_comm_net_pct),
                    'avg_comm_net_pct': float(avg_comm_net_pct),
                    'trend_comm_net': float(trend_comm_net),
                    'data_points': len(df),
                    'report_date': latest['report_date'].isoformat(),
                    'method': 'cftc_direct'
                }
            }
            
        except Exception as e:
            print(f"COT data processing error: {e}")
            return None
    
    def _calculate_cot_sentiment(self, current_net_pct: float, avg_net_pct: float, trend: float) -> float:
        """Calculate sentiment based on COT positioning logic"""
        # Base sentiment from current positioning
        if current_net_pct > 60:
            base_sentiment = 1.0  # Very bullish - commercials heavily net long
        elif current_net_pct > 20:
            base_sentiment = 0.5 + (current_net_pct - 20) / 40 * 0.5  # Bullish
        elif current_net_pct > -20:
            base_sentiment = current_net_pct / 20 * 0.2  # Neutral zone  
        elif current_net_pct > -60:
            base_sentiment = -0.2 + (current_net_pct + 20) / 40 * -0.3  # Bearish
        else:
            base_sentiment = -1.0  # Very bearish - commercials heavily net short
        
        # Trend adjustment (is positioning improving or deteriorating?)
        trend_factor = np.tanh(trend / 10) * 0.2  # Max ±0.2 adjustment
        
        # Relative positioning (vs historical average)
        relative_factor = np.tanh((current_net_pct - avg_net_pct) / 20) * 0.1  # Max ±0.1 adjustment
        
        final_sentiment = base_sentiment + trend_factor + relative_factor
        return max(-1.0, min(1.0, final_sentiment))
    
    def _calculate_cot_confidence(self, df: pd.DataFrame, current_net_pct: float) -> float:
        """Calculate confidence based on data quality and signal strength"""
        # Base confidence from data availability
        if len(df) >= 12:
            data_confidence = 0.9
        elif len(df) >= 6:
            data_confidence = 0.8
        elif len(df) >= 3:
            data_confidence = 0.7
        else:
            data_confidence = 0.6
        
        # Signal strength (extreme positions are more confident)
        if abs(current_net_pct) > 50:
            signal_confidence = 0.95
        elif abs(current_net_pct) > 30:
            signal_confidence = 0.85
        elif abs(current_net_pct) > 15:
            signal_confidence = 0.75
        else:
            signal_confidence = 0.65
        
        # Consistency (lower volatility = higher confidence)
        if len(df) > 4:
            volatility = df['comm_net_pct'].head(12).std()
            consistency_confidence = max(0.5, 1.0 - volatility / 50)
        else:
            consistency_confidence = 0.7
        
        return (data_confidence + signal_confidence + consistency_confidence) / 3
    
    async def _fetch_cot_alternative(self) -> Dict:
        """Alternative COT data source (web scraping)"""
        try:
            print("📊 Attempting alternative COT data source...")
            # This would implement alternative data sources like:
            # - COT reports from financial websites
            # - Third-party COT data providers
            # - Cached historical patterns
            
            # For now, return None to trigger proxy estimation
            return None
            
        except Exception as e:
            print(f"Alternative COT fetch error: {e}")
            return None
    
    async def _estimate_positioning_proxy(self) -> Dict:
        """Estimate positioning based on gold price action and volume"""
        try:
            print("📊 Using COT proxy estimation based on price action...")
            
            # Get recent gold price data
            gold = yf.Ticker("GC=F")  # Gold futures
            hist = gold.history(period="3mo")  # 3 months
            
            if hist.empty:
                return self._create_error_signal("COT_Proxy_Error", "No price data for proxy")
            
            # Calculate price momentum and volume indicators
            returns = hist['Close'].pct_change().dropna()
            volume_trend = hist['Volume'].rolling(10).mean() / hist['Volume'].rolling(30).mean()
            
            # Estimate commercial positioning based on contrarian logic
            # Commercials typically position opposite to price trends (they're hedgers)
            recent_return = returns.tail(10).mean()
            volume_factor = volume_trend.iloc[-1] if not pd.isna(volume_trend.iloc[-1]) else 1.0
            
            # Contrarian logic: if price rising strongly, commercials likely reducing longs/adding shorts
            estimated_comm_position = -recent_return * 100 * volume_factor  # Convert to percentage-like
            
            # Apply positioning logic
            sentiment = self._calculate_cot_sentiment(estimated_comm_position, 0, 0)
            confidence = 0.5  # Lower confidence for proxy method
            
            return {
                'source': 'COT_Price_Proxy',
                'sentiment': float(sentiment),
                'confidence': float(confidence),
                'weight': 0.12,  # Reduced weight for proxy method
                'description': f'COT proxy: Est. commercial position {estimated_comm_position:+.1f}%',
                'metadata': {
                    'estimated_position': float(estimated_comm_position),
                    'recent_return': float(recent_return),
                    'volume_factor': float(volume_factor),
                    'method': 'price_proxy',
                    'note': 'Estimated from price action - lower confidence'
                }
            }
            
        except Exception as e:
            print(f"COT proxy estimation error: {e}")
            return self._create_error_signal("COT_Estimation_Error", f"Proxy estimation failed: {e}")
    
    def _create_error_signal(self, source_name: str, error_message: str) -> Dict:
        """Create error signal with neutral default"""
        return {
            'source': source_name,
            'sentiment': 0.0,  # Neutral when no data
            'confidence': 0.4,
            'weight': 0.10,  # Reduced weight for error case
            'description': error_message,
            'metadata': {'error': True, 'method': 'fallback'}
        }
    
    async def analyze_professional_positioning(self) -> Dict:
        """Main analysis function for COT sentiment"""
        print("🏦 Analyzing Professional Positioning (COT)...")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # Fetch COT data
            cot_result = await self.fetch_cftc_cot_data()
            
            if not cot_result:
                return self._create_error_signal("COT_Analysis_Failed", "All COT data methods failed")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Add execution metadata
            cot_result['execution_time_seconds'] = execution_time
            cot_result['analysis_timestamp'] = datetime.now().isoformat()
            cot_result['kpi_name'] = 'Professional_Positioning_COT'
            cot_result['framework_weight'] = 0.15  # 15% of total gold sentiment
            
            # Interpretation
            sentiment_score = cot_result['sentiment']
            if sentiment_score > 0.6:
                interpretation = "🟢 STRONG BULLISH - Commercials heavily net long"
            elif sentiment_score > 0.2:
                interpretation = "🟢 BULLISH - Commercials net long positioning" 
            elif sentiment_score > -0.2:
                interpretation = "🟡 NEUTRAL - Balanced commercial positioning"
            elif sentiment_score > -0.6:
                interpretation = "🔴 BEARISH - Commercials net short positioning"
            else:
                interpretation = "🔴 STRONG BEARISH - Commercials heavily net short"
            
            cot_result['interpretation'] = interpretation
            
            print(f"✅ COT Analysis completed in {execution_time:.1f}s")
            print(f"📊 Sentiment: {sentiment_score:.3f} ({interpretation})")
            print(f"🎯 Confidence: {cot_result['confidence']:.1%}")
            
            return cot_result
            
        except Exception as e:
            print(f"❌ COT analysis failed: {e}")
            return self._create_error_signal("COT_Analysis_Exception", f"Analysis exception: {e}")

# Integration test function
async def test_cot_analyzer():
    """Test the COT analyzer"""
    analyzer = ProfessionalPositioningAnalyzer()
    result = await analyzer.analyze_professional_positioning()
    
    print("\n" + "="*60)
    print("COT ANALYZER TEST RESULTS")
    print("="*60)
    
    if result:
        print(f"Source: {result.get('source', 'Unknown')}")
        print(f"Sentiment: {result.get('sentiment', 0):.3f}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"Weight: {result.get('weight', 0):.1%}")
        print(f"Description: {result.get('description', 'N/A')}")
        
        if 'metadata' in result:
            print("\nMetadata:")
            for key, value in result['metadata'].items():
                print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_cot_analyzer())
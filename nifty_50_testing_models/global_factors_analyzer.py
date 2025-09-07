
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import asyncio
import aiohttp
from typing import Dict, List, Optional
import json
import hashlib
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class CachedGlobalData:
    data: any
    timestamp: datetime
    ttl_minutes: int = 30  # 30 minutes for global data
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(minutes=self.ttl_minutes)

class GlobalFactorsAnalyzer:
    """
    Global Factors Sentiment Analyzer for Nifty 50
    - US markets overnight performance
    - INR/USD exchange rate trends
    - Asian markets correlation
    - Crude oil price impact
    - Dollar Index (DXY) influence
    - 5% weight in composite sentiment
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.cache = {}
        
        # Global market tickers for real-time data
        self.global_tickers = {
            'us_markets': {
                'sp500': '^GSPC',
                'nasdaq': '^IXIC', 
                'dow': '^DJI'
            },
            'asian_markets': {
                'nikkei': '^N225',
                'hang_seng': '^HSI',
                'shanghai': '000001.SS'
            },
            'currencies': {
                'inr_usd': 'INRUSD=X',
                'dxy': 'DX-Y.NYB'
            },
            'commodities': {
                'crude_oil': 'CL=F',
                'gold': 'GC=F'
            }
        }
        
        print("Global Factors Analyzer v1.0 initialized")
        print("🌍 Real-time global market correlation tracking enabled")
    
    def _default_config(self) -> Dict:
        return {
            'cache_ttl_minutes': 30,  # 30 minutes for global data
            'lookback_days': 5,  # 5 days for global correlation
            'confidence_threshold': 0.75,
            'component_weights': {
                'us_overnight': 0.35,      # US market overnight performance
                'inr_usd_trend': 0.25,     # INR/USD exchange rate
                'asian_correlation': 0.20, # Asian markets performance
                'crude_oil_impact': 0.15,  # Oil price impact on India
                'dxy_influence': 0.05      # Dollar index
            }
        }
    
    def _get_cache_key(self, source: str) -> str:
        return hashlib.md5(f"global_{source}".encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[any]:
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if not cached.is_expired():
                return cached.data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: any):
        self.cache[cache_key] = CachedGlobalData(data, datetime.now(), self.config['cache_ttl_minutes'])
    
    async def fetch_us_overnight_performance(self) -> Dict:
        """Fetch US market overnight performance"""
        cache_key = self._get_cache_key("us_overnight")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("🇺🇸 Fetching US market overnight data...")
            us_data = {}
            
            for market, ticker in self.global_tickers['us_markets'].items():
                try:
                    market_ticker = yf.Ticker(ticker)
                    hist = market_ticker.history(period="2d")
                    
                    if not hist.empty and len(hist) >= 2:
                        prev_close = hist['Close'].iloc[-2]
                        current_close = hist['Close'].iloc[-1]
                        overnight_change = (current_close - prev_close) / prev_close * 100
                        
                        us_data[market] = {
                            'ticker': ticker,
                            'previous_close': float(prev_close),
                            'current_close': float(current_close),
                            'overnight_change_pct': float(overnight_change)
                        }
                except Exception as e:
                    print(f"Failed to fetch {market}: {e}")
                    continue
            
            if us_data:
                # Calculate composite US sentiment
                changes = [data['overnight_change_pct'] for data in us_data.values()]
                avg_change = np.mean(changes)
                
                # Convert to sentiment (-1 to +1)
                us_sentiment = np.tanh(avg_change / 2)  # Scale 2% move to ~0.76 sentiment
                
                result = {
                    'sentiment': float(us_sentiment),
                    'confidence': 0.85,
                    'individual_markets': us_data,
                    'average_change_pct': float(avg_change),
                    'interpretation': f"US markets {'up' if avg_change > 0 else 'down'} {abs(avg_change):.2f}% overnight"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            return self._create_fallback_result("us_overnight", "No US market data available")
            
        except Exception as e:
            print(f"US overnight fetch error: {e}")
            return self._create_fallback_result("us_overnight", f"US market error: {e}")
    
    async def fetch_inr_usd_trend(self) -> Dict:
        """Fetch INR/USD exchange rate trend"""
        cache_key = self._get_cache_key("inr_usd")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("💱 Fetching INR/USD exchange rate data...")
            
            inr_ticker = yf.Ticker(self.global_tickers['currencies']['inr_usd'])
            hist = inr_ticker.history(period="5d")
            
            if not hist.empty and len(hist) >= 3:
                current_rate = hist['Close'].iloc[-1]
                prev_rate = hist['Close'].iloc[-2]
                week_avg = hist['Close'].mean()
                
                daily_change = (current_rate - prev_rate) / prev_rate * 100
                trend_change = (current_rate - week_avg) / week_avg * 100
                
                # INR weakening is negative for Nifty (higher USD/INR)
                # But our ticker might be INR/USD, so check direction
                if current_rate < 0.02:  # If it's INR/USD (small number)
                    inr_sentiment = np.tanh(daily_change * 2)  # INR strength is positive
                else:  # If it's USD/INR (large number like 83)
                    inr_sentiment = -np.tanh(daily_change / 2)  # INR weakness is negative
                
                result = {
                    'sentiment': float(inr_sentiment),
                    'confidence': 0.80,
                    'current_rate': float(current_rate),
                    'daily_change_pct': float(daily_change),
                    'trend_change_pct': float(trend_change),
                    'interpretation': f"INR {'strengthening' if inr_sentiment > 0 else 'weakening'} vs USD"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            return self._create_fallback_result("inr_usd", "No INR/USD data available")
            
        except Exception as e:
            print(f"INR/USD fetch error: {e}")
            return self._create_fallback_result("inr_usd", f"Currency error: {e}")
    
    async def fetch_asian_markets_correlation(self) -> Dict:
        """Fetch Asian markets performance for correlation"""
        cache_key = self._get_cache_key("asian_markets")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("🌏 Fetching Asian markets data...")
            asian_data = {}
            
            for market, ticker in self.global_tickers['asian_markets'].items():
                try:
                    market_ticker = yf.Ticker(ticker)
                    hist = market_ticker.history(period="2d")
                    
                    if not hist.empty and len(hist) >= 2:
                        prev_close = hist['Close'].iloc[-2]
                        current_close = hist['Close'].iloc[-1]
                        change_pct = (current_close - prev_close) / prev_close * 100
                        
                        asian_data[market] = {
                            'ticker': ticker,
                            'change_pct': float(change_pct)
                        }
                except Exception as e:
                    print(f"Failed to fetch {market}: {e}")
                    continue
            
            if asian_data:
                changes = [data['change_pct'] for data in asian_data.values()]
                avg_asian_change = np.mean(changes)
                
                # Asian markets correlation with Nifty
                asian_sentiment = np.tanh(avg_asian_change / 3)
                
                result = {
                    'sentiment': float(asian_sentiment),
                    'confidence': 0.75,
                    'individual_markets': asian_data,
                    'average_change_pct': float(avg_asian_change),
                    'markets_analyzed': len(asian_data),
                    'interpretation': f"Asian markets {'positive' if avg_asian_change > 0 else 'negative'} {abs(avg_asian_change):.2f}%"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            return self._create_fallback_result("asian_markets", "No Asian market data available")
            
        except Exception as e:
            print(f"Asian markets fetch error: {e}")
            return self._create_fallback_result("asian_markets", f"Asian markets error: {e}")
    
    async def fetch_crude_oil_impact(self) -> Dict:
        """Fetch crude oil price impact on India"""
        cache_key = self._get_cache_key("crude_oil")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("🛢️ Fetching crude oil price data...")
            
            oil_ticker = yf.Ticker(self.global_tickers['commodities']['crude_oil'])
            hist = oil_ticker.history(period="5d")
            
            if not hist.empty and len(hist) >= 3:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                week_avg = hist['Close'].mean()
                
                daily_change = (current_price - prev_price) / prev_price * 100
                trend_change = (current_price - week_avg) / week_avg * 100
                
                # Higher oil prices are negative for India (import dependent)
                oil_sentiment = -np.tanh(daily_change / 5)  # Oil up 5% = -0.76 sentiment
                
                result = {
                    'sentiment': float(oil_sentiment),
                    'confidence': 0.85,
                    'current_price': float(current_price),
                    'daily_change_pct': float(daily_change),
                    'trend_change_pct': float(trend_change),
                    'interpretation': f"Oil {'up' if daily_change > 0 else 'down'} {abs(daily_change):.2f}% ({'negative' if daily_change > 0 else 'positive'} for India)"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            return self._create_fallback_result("crude_oil", "No crude oil data available")
            
        except Exception as e:
            print(f"Crude oil fetch error: {e}")
            return self._create_fallback_result("crude_oil", f"Oil price error: {e}")
    
    async def fetch_dxy_influence(self) -> Dict:
        """Fetch Dollar Index influence"""
        cache_key = self._get_cache_key("dxy")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("💵 Fetching Dollar Index (DXY) data...")
            
            dxy_ticker = yf.Ticker(self.global_tickers['currencies']['dxy'])
            hist = dxy_ticker.history(period="5d")
            
            if not hist.empty and len(hist) >= 3:
                current_dxy = hist['Close'].iloc[-1]
                prev_dxy = hist['Close'].iloc[-2]
                week_avg = hist['Close'].mean()
                
                daily_change = (current_dxy - prev_dxy) / prev_dxy * 100
                trend_change = (current_dxy - week_avg) / week_avg * 100
                
                # Stronger dollar is generally negative for emerging markets like India
                dxy_sentiment = -np.tanh(daily_change / 3)
                
                result = {
                    'sentiment': float(dxy_sentiment),
                    'confidence': 0.75,
                    'current_dxy': float(current_dxy),
                    'daily_change_pct': float(daily_change),
                    'trend_change_pct': float(trend_change),
                    'interpretation': f"DXY {'stronger' if daily_change > 0 else 'weaker'} {abs(daily_change):.2f}%"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            return self._create_fallback_result("dxy", "No DXY data available")
            
        except Exception as e:
            print(f"DXY fetch error: {e}")
            return self._create_fallback_result("dxy", f"DXY error: {e}")
    
    def _create_fallback_result(self, component: str, error_msg: str) -> Dict:
        """Create fallback result for failed components"""
        return {
            'sentiment': 0.0,
            'confidence': 0.3,
            'error': True,
            'error_message': error_msg,
            'interpretation': f"{component} data unavailable"
        }
    
    async def analyze_global_factors_sentiment(self) -> Dict:
        """Main analysis function for global factors sentiment"""
        print("🌍 Analyzing Global Factors Impact...")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # Fetch all global components concurrently
            tasks = [
                self.fetch_us_overnight_performance(),
                self.fetch_inr_usd_trend(),
                self.fetch_asian_markets_correlation(),
                self.fetch_crude_oil_impact(),
                self.fetch_dxy_influence()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            component_results = {
                'us_overnight': results[0] if not isinstance(results[0], Exception) else self._create_fallback_result("us_overnight", str(results[0])),
                'inr_usd_trend': results[1] if not isinstance(results[1], Exception) else self._create_fallback_result("inr_usd_trend", str(results[1])),
                'asian_correlation': results[2] if not isinstance(results[2], Exception) else self._create_fallback_result("asian_correlation", str(results[2])),
                'crude_oil_impact': results[3] if not isinstance(results[3], Exception) else self._create_fallback_result("crude_oil_impact", str(results[3])),
                'dxy_influence': results[4] if not isinstance(results[4], Exception) else self._create_fallback_result("dxy_influence", str(results[4]))
            }
            
            # Calculate composite global sentiment
            total_weighted_sentiment = 0
            total_weight = 0
            confidence_scores = []
            
            weights = self.config['component_weights']
            for comp_name, comp_data in component_results.items():
                if isinstance(comp_data, dict) and 'sentiment' in comp_data:
                    weight = weights.get(comp_name, 0.2)  # Default weight
                    sentiment = comp_data.get('sentiment', 0)
                    confidence = comp_data.get('confidence', 0.5)
                    
                    adjusted_weight = weight * confidence
                    total_weighted_sentiment += sentiment * adjusted_weight
                    total_weight += adjusted_weight
                    confidence_scores.append(confidence)
            
            composite_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Build result structure
            result = {
                'component_sentiment': float(composite_sentiment),
                'component_confidence': float(avg_confidence),
                'component_weight': 0.05,  # 5% weight in overall system
                'weighted_contribution': float(composite_sentiment * 0.05),
                
                'sub_indicators': component_results,
                
                'market_context': {
                    'components_analyzed': len([c for c in component_results.values() if not c.get('error', False)]),
                    'components_failed': len([c for c in component_results.values() if c.get('error', False)]),
                    'data_sources': ['Yahoo_Finance_Global', 'Currency_Markets', 'Commodity_Markets'],
                    'update_frequency': '30_minutes',
                    'global_session': self._get_global_session()
                },
                
                'execution_time_seconds': execution_time,
                'kpi_name': 'Global_Factors',
                'framework_weight': 0.05,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': 'high' if avg_confidence > 0.7 else 'moderate'
            }
            
            # Overall interpretation
            if composite_sentiment > 0.3:
                interpretation = "🟢 POSITIVE - Global factors supportive"
            elif composite_sentiment > 0.1:
                interpretation = "🟡 MILDLY POSITIVE - Slight global tailwinds"
            elif composite_sentiment > -0.1:
                interpretation = "⚪ NEUTRAL - Mixed global signals"
            elif composite_sentiment > -0.3:
                interpretation = "🟡 MILDLY NEGATIVE - Slight global headwinds"
            else:
                interpretation = "🔴 NEGATIVE - Global factors challenging"
            
            result['interpretation'] = interpretation
            
            print(f"✅ Global Factors Analysis completed in {execution_time:.1f}s")
            print(f"📊 Sentiment: {composite_sentiment:.3f} ({interpretation})")
            print(f"🎯 Confidence: {avg_confidence:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Global factors analysis failed: {e}")
            return self._create_error_result(f"Analysis exception: {e}")
    
    def _get_global_session(self) -> str:
        """Determine current global trading session"""
        now = datetime.now()
        hour = now.hour
        
        # Rough global session times (IST)
        if 21 <= hour <= 23 or 0 <= hour <= 4:
            return "us_session"
        elif 5 <= hour <= 11:
            return "asian_session"
        elif 12 <= hour <= 20:
            return "european_asian_overlap"
        else:
            return "off_hours"
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result with neutral default"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.05 * 0.5,  # Reduced weight for error
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
async def test_global_factors_analyzer():
    """Test the global factors analyzer"""
    analyzer = GlobalFactorsAnalyzer()
    result = await analyzer.analyze_global_factors_sentiment()
    
    print("\n" + "="*60)
    print("GLOBAL FACTORS ANALYZER TEST RESULTS")
    print("="*60)
    
    if result:
        print(f"Component Sentiment: {result.get('component_sentiment', 0):.3f}")
        print(f"Component Confidence: {result.get('component_confidence', 0):.1%}")
        print(f"Component Weight: {result.get('component_weight', 0):.0%}")
        print(f"Interpretation: {result.get('interpretation', 'N/A')}")
        
        # Save detailed JSON
        with open('global_factors_analysis.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print("\n✅ Detailed results saved to: global_factors_analysis.json")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_global_factors_analyzer())
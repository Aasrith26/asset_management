# FII/DII FLOWS SENTIMENT ANALYZER v1.0
# ====================================
# Real-time Foreign & Domestic Institutional Investment flow analysis for Nifty 50
# 35% weight in composite sentiment - highest priority KPI

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
import re
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

@dataclass
class CachedFlowData:
    data: any
    timestamp: datetime
    ttl_minutes: int = 60  # 1 hour for flow data
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(minutes=self.ttl_minutes)

class FIIDIIFlowsAnalyzer:
    """
    FII/DII Flows Sentiment Analyzer for Nifty 50
    - Real-time institutional flow tracking
    - Multi-source data with fallbacks
    - Flow momentum and trend analysis
    - 35% weight in composite sentiment
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.cache = {}
        print("FII/DII Flows Analyzer v1.0 initialized")
        print("🏦 Real-time institutional flow tracking enabled")
    
    def _default_config(self) -> Dict:
        return {
            'request_timeout': 10,
            'cache_ttl_minutes': 60,  # 1 hour for flow data
            'lookback_days': 30,  # 1 month of flow data
            'confidence_threshold': 0.7,
            'flow_thresholds': {
                'strong_positive': 2000,  # Crores
                'positive': 500,
                'neutral_high': 100,
                'neutral_low': -100,
                'negative': -500,
                'strong_negative': -2000
            }
        }
    
    def _get_cache_key(self, source: str, params: str = "") -> str:
        """Generate cache key for flow data"""
        return hashlib.md5(f"flow_{source}_{params}".encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[any]:
        """Retrieve cached flow data if valid"""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if not cached.is_expired():
                return cached.data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: any, ttl_minutes: int = None):
        """Cache flow data with TTL"""
        ttl = ttl_minutes or self.config['cache_ttl_minutes']
        self.cache[cache_key] = CachedFlowData(data, datetime.now(), ttl)
    
    async def fetch_fii_dii_flows(self) -> Dict:
        cache_key = self._get_cache_key("fii_dii_flows")
        cached_result = self._get_cached_data(cache_key)
        
        if cached_result:
            print("FII/DII Data: Using cached result")
            return cached_result
        
        print("Fetching real-time FII/DII flow data...")
        
        try:
            result = await self._fetch_nse_flow_data()
            if result and result.get('confidence', 0) > 0.6:
                self._cache_data(cache_key, result)
                return result

            result = await self._fetch_moneycontrol_flows()
            if result:
                self._cache_data(cache_key, result, 30)  # Shorter cache for fallback
                return result
            
            # Fallback 2: Economic Times data
            result = await self._fetch_et_flows()
            if result:
                self._cache_data(cache_key, result, 30)
                return result
            
            # Final fallback: Estimated flows from market movement
            return await self._estimate_flows_from_market()
            
        except Exception as e:
            print(f"FII/DII flow fetch error: {e}")
            return await self._estimate_flows_from_market()
    
    async def _fetch_nse_flow_data(self) -> Dict:
        """Fetch FII/DII data from NSE official sources"""
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config['request_timeout']),
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            # NSE FII/DII data endpoint
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    await session.close()
                    
                    if data and len(data) > 0:
                        return await self._process_nse_flow_data(data)
            
            await session.close()
            return None
            
        except Exception as e:
            print(f"NSE flow data fetch error: {e}")
            return None
    
    async def _process_nse_flow_data(self, raw_data: List[Dict]) -> Dict:
        """Process NSE FII/DII flow data into sentiment signals"""
        try:
            if not raw_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            
            # Extract latest flows
            latest = df.iloc[0] if len(df) > 0 else None
            if not latest.empty:
                return None
            
            # Extract flow values (in crores)
            fii_flow = float(latest.get('fiiNetFlow', 0))
            dii_flow = float(latest.get('diiNetFlow', 0))
            net_flow = fii_flow + dii_flow
            
            # Historical context (if available)
            if len(df) >= 5:
                avg_fii = df['fiiNetFlow'].astype(float).tail(5).mean()
                avg_dii = df['diiNetFlow'].astype(float).tail(5).mean()
                avg_net = avg_fii + avg_dii
            else:
                avg_fii = fii_flow
                avg_dii = dii_flow
                avg_net = net_flow
            
            # Calculate flow momentum
            flow_momentum = self._calculate_flow_momentum(df)
            
            # Calculate sentiment scores
            fii_sentiment = self._calculate_flow_sentiment(fii_flow, avg_fii, 'FII')
            dii_sentiment = self._calculate_flow_sentiment(dii_flow, avg_dii, 'DII')
            net_sentiment = self._calculate_flow_sentiment(net_flow, avg_net, 'NET')
            
            # Composite sentiment (DII gets slightly higher weight as domestic)
            composite_sentiment = (fii_sentiment * 0.45 + dii_sentiment * 0.55)
            
            # Confidence based on flow magnitude and consistency
            confidence = self._calculate_flow_confidence(fii_flow, dii_flow, df)
            
            result = {
                'component_sentiment': float(composite_sentiment),
                'component_confidence': float(confidence),
                'component_weight': 0.35,
                'weighted_contribution': float(composite_sentiment * 0.35),
                
                'sub_indicators': {
                    'fii_flows': {
                        'net_flow_crores': float(fii_flow),
                        'sentiment': float(fii_sentiment),
                        'avg_5d_flow': float(avg_fii),
                        'interpretation': self._interpret_flow(fii_flow, 'FII')
                    },
                    'dii_flows': {
                        'net_flow_crores': float(dii_flow),
                        'sentiment': float(dii_sentiment),
                        'avg_5d_flow': float(avg_dii),
                        'interpretation': self._interpret_flow(dii_flow, 'DII')
                    },
                    'net_flows': {
                        'total_net_flow_crores': float(net_flow),
                        'sentiment': float(net_sentiment),
                        'avg_5d_flow': float(avg_net),
                        'interpretation': self._interpret_flow(net_flow, 'NET')
                    },
                    'flow_momentum': {
                        'momentum_score': float(flow_momentum),
                        'trend': 'increasing' if flow_momentum > 0.1 else 'decreasing' if flow_momentum < -0.1 else 'stable',
                        'interpretation': f'Flow momentum: {flow_momentum:+.3f}'
                    },
                    'flow_balance': {
                        'fii_dii_ratio': float(fii_flow / (dii_flow + 1)) if dii_flow != 0 else 0,
                        'dominance': 'FII' if abs(fii_flow) > abs(dii_flow) else 'DII',
                        'interpretation': 'FII dominance' if abs(fii_flow) > abs(dii_flow) else 'DII dominance'
                    }
                },
                
                'market_context': {
                    'data_date': latest.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'data_points': len(df),
                    'data_source': 'NSE_Official',
                    'data_freshness': 'real_time',
                    'total_market_impact': abs(net_flow),
                    'flow_category': self._categorize_flow_strength(net_flow)
                },
                
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': 'high'
            }
            
            return result
            
        except Exception as e:
            print(f"NSE flow data processing error: {e}")
            return None
    
    def _calculate_flow_sentiment(self, current_flow: float, avg_flow: float, flow_type: str) -> float:
        """Calculate sentiment based on flow thresholds and trends"""
        thresholds = self.config['flow_thresholds']
        
        # Base sentiment from absolute flow
        if current_flow >= thresholds['strong_positive']:
            base_sentiment = 1.0
        elif current_flow >= thresholds['positive']:
            base_sentiment = 0.5 + (current_flow - thresholds['positive']) / (thresholds['strong_positive'] - thresholds['positive']) * 0.5
        elif current_flow >= thresholds['neutral_high']:
            base_sentiment = current_flow / thresholds['positive'] * 0.5
        elif current_flow >= thresholds['neutral_low']:
            base_sentiment = 0.0
        elif current_flow >= thresholds['negative']:
            base_sentiment = current_flow / abs(thresholds['negative']) * -0.5
        elif current_flow >= thresholds['strong_negative']:
            base_sentiment = -0.5 + (current_flow - thresholds['negative']) / (thresholds['strong_negative'] - thresholds['negative']) * -0.5
        else:
            base_sentiment = -1.0
        
        # Trend adjustment (vs recent average)
        if avg_flow != 0:
            trend_factor = (current_flow - avg_flow) / (abs(avg_flow) + 100) * 0.2
            base_sentiment += trend_factor
        
        return max(-1.0, min(1.0, base_sentiment))

    # QUICK FIX for FII/DII Flow Sentiment Calculation
    # Add this improved version to fii_dii_flows_analyzer.py

    def _calculate_flow_sentiment(self, current_flow: float, avg_flow: float, flow_type: str) -> float:
        """Calculate sentiment based on flow thresholds and trends - IMPROVED VERSION"""
        thresholds = self.config['flow_thresholds']

        # ENHANCED: Add micro-sentiment for small flows (weekend/after-hours)
        if -50 < current_flow < 50:
            # Micro-sentiment: scale small flows to -0.5 to +0.5 range
            micro_sentiment = current_flow / 100.0  # -7 → -0.07, +15 → +0.15

            # Apply trend adjustment if we have historical data
            if avg_flow != 0:
                trend_factor = (current_flow - avg_flow) / (abs(avg_flow) + 50) * 0.1
                micro_sentiment += trend_factor

            return max(-0.5, min(0.5, micro_sentiment))

        # ORIGINAL: Base sentiment from absolute flow (for large flows)
        if current_flow >= thresholds['strong_positive']:
            base_sentiment = 1.0
        elif current_flow >= thresholds['positive']:
            base_sentiment = 0.5 + (current_flow - thresholds['positive']) / (
                        thresholds['strong_positive'] - thresholds['positive']) * 0.5
        elif current_flow >= thresholds['neutral_high']:
            base_sentiment = current_flow / thresholds['positive'] * 0.5
        elif current_flow >= thresholds['neutral_low']:
            base_sentiment = 0.0
        elif current_flow >= thresholds['negative']:
            base_sentiment = current_flow / abs(thresholds['negative']) * -0.5
        elif current_flow >= thresholds['strong_negative']:
            base_sentiment = -0.5 + (current_flow - thresholds['negative']) / (
                        thresholds['strong_negative'] - thresholds['negative']) * -0.5
        else:
            base_sentiment = -1.0

        # Trend adjustment (vs recent average)
        if avg_flow != 0:
            trend_factor = (current_flow - avg_flow) / (abs(avg_flow) + 100) * 0.2
            base_sentiment += trend_factor

        return max(-1.0, min(1.0, base_sentiment))

    # Also update the flow thresholds to be more sensitive:
    def _default_config(self) -> Dict:
        return {
            'request_timeout': 10,
            'cache_ttl_minutes': 60,
            'lookback_days': 30,
            'confidence_threshold': 0.7,
            'flow_thresholds': {
                'strong_positive': 2000,  # Crores
                'positive': 500,
                'neutral_high': 50,  # REDUCED from 100
                'neutral_low': -50,  # REDUCED from -100
                'negative': -500,
                'strong_negative': -2000
            }
        }
    def _calculate_flow_confidence(self, fii_flow: float, dii_flow: float, df: pd.DataFrame) -> float:
        """Calculate confidence based on flow data quality and magnitude"""
        # Base confidence from data availability
        data_confidence = min(0.9, 0.6 + len(df) * 0.05)
        
        # Flow magnitude confidence (larger flows = higher confidence)
        net_flow = abs(fii_flow + dii_flow)
        if net_flow > 2000:
            magnitude_confidence = 0.95
        elif net_flow > 1000:
            magnitude_confidence = 0.90
        elif net_flow > 500:
            magnitude_confidence = 0.85
        elif net_flow > 100:
            magnitude_confidence = 0.75
        else:
            magnitude_confidence = 0.65
        
        # Consistency confidence (aligned FII/DII = higher confidence)
        if (fii_flow > 0 and dii_flow > 0) or (fii_flow < 0 and dii_flow < 0):
            consistency_confidence = 0.9  # Both buying or both selling
        else:
            consistency_confidence = 0.7  # Mixed signals
        
        return (data_confidence + magnitude_confidence + consistency_confidence) / 3
    
    def _interpret_flow(self, flow: float, flow_type: str) -> str:
        """Interpret flow value into human-readable description"""
        if flow >= 2000:
            return f"Strong {flow_type} buying (₹{flow:.0f} cr)"
        elif flow >= 500:
            return f"Moderate {flow_type} buying (₹{flow:.0f} cr)"
        elif flow >= 100:
            return f"Light {flow_type} buying (₹{flow:.0f} cr)"
        elif flow >= -100:
            return f"Neutral {flow_type} activity (₹{flow:.0f} cr)"
        elif flow >= -500:
            return f"Light {flow_type} selling (₹{flow:.0f} cr)"
        elif flow >= -2000:
            return f"Moderate {flow_type} selling (₹{flow:.0f} cr)"
        else:
            return f"Strong {flow_type} selling (₹{flow:.0f} cr)"
    
    def _categorize_flow_strength(self, net_flow: float) -> str:
        """Categorize overall flow strength"""
        abs_flow = abs(net_flow)
        if abs_flow >= 2000:
            return "very_strong"
        elif abs_flow >= 1000:
            return "strong"
        elif abs_flow >= 500:
            return "moderate"
        elif abs_flow >= 100:
            return "light"
        else:
            return "minimal"
    
    async def _fetch_moneycontrol_flows(self) -> Dict:
        """Fallback: Scrape FII/DII data from Moneycontrol"""
        try:
            print("📊 Attempting Moneycontrol FII/DII data...")
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8),
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            
            url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/"
            
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    await session.close()
                    
                    # Parse HTML to extract FII/DII flows
                    return self._parse_moneycontrol_data(html)
            
            await session.close()
            return None
            
        except Exception as e:
            print(f"Moneycontrol flow fetch error: {e}")
            return None
    
    def _parse_moneycontrol_data(self, html: str) -> Dict:
        """Parse Moneycontrol HTML to extract flow data"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for FII/DII flow values in the HTML
            # This is a simplified parser - would need to be more robust for production
            flow_pattern = r'[\-]?\d+[\d,]*\.?\d*'
            numbers = re.findall(flow_pattern, html)
            
            if len(numbers) >= 4:
                # Estimate FII/DII flows from parsed numbers
                # This is approximate - actual parsing would be more specific
                fii_flow = float(numbers[0].replace(',', ''))
                dii_flow = float(numbers[1].replace(',', ''))
                
                # Create simplified result
                net_flow = fii_flow + dii_flow
                sentiment = self._calculate_flow_sentiment(net_flow, 0, 'NET')
                
                return {
                    'component_sentiment': float(sentiment),
                    'component_confidence': 0.7,
                    'component_weight': 0.35,
                    'sub_indicators': {
                        'estimated_flows': {
                            'fii_flow': fii_flow,
                            'dii_flow': dii_flow,
                            'net_flow': net_flow,
                            'source': 'moneycontrol_estimate'
                        }
                    },
                    'market_context': {
                        'data_source': 'Moneycontrol_Scraped',
                        'confidence_note': 'Estimated from web scraping'
                    }
                }
            
            return None
            
        except Exception as e:
            print(f"Moneycontrol parsing error: {e}")
            return None
    
    async def _fetch_et_flows(self) -> Dict:
        """Fallback: Economic Times FII/DII data"""
        try:
            print("📊 Attempting Economic Times FII/DII data...")
            # Similar implementation to Moneycontrol
            # Would scrape ET's FII/DII section
            return None
            
        except Exception as e:
            print(f"ET flow fetch error: {e}")
            return None
    
    async def _estimate_flows_from_market(self) -> Dict:
        """Final fallback: Estimate flows from Nifty price movement and volume"""
        try:
            print("📊 Estimating flows from market movement...")
            
            # Get Nifty data
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")
            
            if hist.empty:
                return self._create_error_result("No market data available")
            
            # Estimate institutional activity from price and volume
            price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
            volume_ratio = hist['Volume'].iloc[-1] / hist['Volume'].tail(5).mean()
            
            # Simple estimation: strong price moves with high volume suggest institutional activity
            estimated_net_flow = price_change * volume_ratio * 1000  # Rough estimation
            
            sentiment = self._calculate_flow_sentiment(estimated_net_flow, 0, 'ESTIMATED')
            
            return {
                'component_sentiment': float(sentiment),
                'component_confidence': 0.5,  # Lower confidence for estimation
                'component_weight': 0.35,
                'sub_indicators': {
                    'estimated_flows': {
                        'estimated_net_flow': float(estimated_net_flow),
                        'price_change': float(price_change),
                        'volume_ratio': float(volume_ratio),
                        'method': 'price_volume_proxy'
                    }
                },
                'market_context': {
                    'data_source': 'Market_Movement_Estimate',
                    'confidence_note': 'Estimated from price/volume - low confidence'
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Flow estimation error: {e}")
            return self._create_error_result(f"Flow estimation failed: {e}")
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result with neutral default"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.35 * 0.5,  # Reduced weight for error
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
    
    async def analyze_fii_dii_sentiment(self) -> Dict:
        """Main analysis function for FII/DII sentiment"""
        print("🏦 Analyzing FII/DII Institutional Flows...")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # Fetch flow data
            flow_result = await self.fetch_fii_dii_flows()
            
            if not flow_result:
                return self._create_error_result("All FII/DII data methods failed")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Add execution metadata
            flow_result['execution_time_seconds'] = execution_time
            flow_result['kpi_name'] = 'FII_DII_Flows'
            flow_result['framework_weight'] = 0.35
            
            # Overall interpretation
            sentiment_score = flow_result['component_sentiment']
            if sentiment_score > 0.6:
                interpretation = "🟢 STRONG BULLISH - Heavy institutional buying"
            elif sentiment_score > 0.3:
                interpretation = "🟢 BULLISH - Net institutional inflows"
            elif sentiment_score > 0.1:
                interpretation = "🟡 MILDLY BULLISH - Light buying pressure"
            elif sentiment_score > -0.1:
                interpretation = "⚪ NEUTRAL - Balanced institutional activity"
            elif sentiment_score > -0.3:
                interpretation = "🟡 MILDLY BEARISH - Light selling pressure"
            elif sentiment_score > -0.6:
                interpretation = "🔴 BEARISH - Net institutional outflows"
            else:
                interpretation = "🔴 STRONG BEARISH - Heavy institutional selling"
            
            flow_result['interpretation'] = interpretation
            
            print(f"FII/DII Analysis completed in {execution_time:.1f}s")
            print(f"Sentiment: {sentiment_score:.3f} ({interpretation})")
            print(f"Confidence: {flow_result['component_confidence']:.1%}")
            
            return flow_result
            
        except Exception as e:
            print(f"FII/DII analysis failed: {e}")
            return self._create_error_result(f"Analysis exception: {e}")

# Test function
async def test_fii_dii_analyzer():
    """Test the FII/DII analyzer"""
    analyzer = FIIDIIFlowsAnalyzer()
    result = await analyzer.analyze_fii_dii_sentiment()
    
    print("\n" + "="*60)
    print("FII/DII FLOWS ANALYZER TEST RESULTS")
    print("="*60)
    
    if result:
        print(f"Component Sentiment: {result.get('component_sentiment', 0):.3f}")
        print(f"Component Confidence: {result.get('component_confidence', 0):.1%}")
        print(f"Component Weight: {result.get('component_weight', 0):.0%}")
        print(f"Interpretation: {result.get('interpretation', 'N/A')}")
        
        # Save detailed JSON
        with open('fii_dii_flows_analysis.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print("\n✅ Detailed results saved to: fii_dii_flows_analysis.json")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fii_dii_analyzer())
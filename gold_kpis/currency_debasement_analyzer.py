# CURRENCY DEBASEMENT RISK SENTIMENT ANALYZER v1.0
# ====================================================
# Integrates with existing enterprise gold sentiment system
# Multi-component currency debasement analysis for gold sentiment

import pandas as pd
import numpy as np
import requests
import json
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import hashlib
from dataclasses import dataclass
import os
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

@dataclass
class CachedCurrencyData:
    data: any
    timestamp: datetime
    ttl_minutes: int = 60  # 1 hour for currency data (high frequency)
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(minutes=self.ttl_minutes)

class CurrencyDebasementAnalyzer:
    """
    Currency Debasement Risk Sentiment Analyzer
    - Multi-component analysis beyond just DXY
    - USD Dollar Index (40% weight)
    - Emerging Market Currency Stress (30% weight)  
    - Central Bank Balance Sheet Expansion (30% weight)
    - 10% weight in composite gold sentiment
    """
    
    def __init__(self, config: Dict = None, fred_api_key: str = None):
        self.config = config or self._default_config()
        self.cache = {}
        self.fred_api_key = fred_api_key or os.environ.get('FRED_API_KEY')
        
        # Initialize FRED client if API key available
        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            self.fred = None
            print("⚠️ Warning: FRED API key not available, using fallback methods")
        
        print("Currency Debasement Risk Analyzer v1.0 initialized")
        print("💱 Multi-component currency debasement analysis enabled")
    
    def _default_config(self) -> Dict:
        return {
            'request_timeout': 8,
            'cache_ttl_minutes': 60,  # 1 hour for currency data
            'confidence_threshold': 0.5,
            'lookback_days': 90,  # 3 months of currency data
            'component_weights': {
                'dxy': 0.40,           # USD Dollar Index - 40%
                'em_stress': 0.30,     # EM Currency Stress - 30%
                'cb_expansion': 0.30   # CB Balance Sheet - 30%
            },
            'em_currencies': ['USDCNY=X', 'USDINR=X', 'USDBRL=X', 'USDZAR=X', 'USDTRY=X'],
            'major_currencies': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCAD=X']
        }
    
    def _get_cache_key(self, source: str, params: str = "") -> str:
        """Generate cache key for currency data storage"""
        return hashlib.md5(f"currency_{source}_{params}".encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[any]:
        """Retrieve cached currency data if valid"""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if not cached.is_expired():
                return cached.data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: any, ttl_minutes: int = None):
        """Cache currency data with TTL"""
        ttl = ttl_minutes or self.config['cache_ttl_minutes']
        self.cache[cache_key] = CachedCurrencyData(data, datetime.now(), ttl)

    async def fetch_dxy_sentiment(self) -> Dict:
        """Fetch and analyze USD Dollar Index (DXY) sentiment"""
        cache_key = self._get_cache_key("dxy_sentiment")
        cached_result = self._get_cached_data(cache_key)
        
        if cached_result:
            return cached_result
        
        print("📊 Analyzing USD Dollar Index (DXY)...")
        
        try:
            # Primary: FRED API for DXY
            if self.fred:
                dxy_result = await self._fetch_dxy_fred()
                if dxy_result:
                    self._cache_data(cache_key, dxy_result)
                    return dxy_result
            
            # Fallback: Yahoo Finance for DX-Y.NYB
            dxy_result = await self._fetch_dxy_yahoo()
            if dxy_result:
                self._cache_data(cache_key, dxy_result, 30)  # Shorter cache for fallback
                return dxy_result
            
            # Final fallback: Estimated DXY sentiment
            return await self._estimate_dxy_sentiment()
            
        except Exception as e:
            print(f"DXY analysis error: {e}")
            return await self._estimate_dxy_sentiment()
    
    async def _fetch_dxy_fred(self) -> Dict:
        """Fetch DXY data from FRED"""
        try:
            # DXY series from FRED
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['lookback_days'])
            
            dxy_data = self.fred.get_series('DTWEXBGS', 
                                          observation_start=start_date.strftime('%Y-%m-%d'),
                                          observation_end=end_date.strftime('%Y-%m-%d'))
            
            if dxy_data.empty:
                return None
            
            # Calculate DXY sentiment (inverse relationship with gold)
            current_dxy = dxy_data.iloc[-1]
            ma_30 = dxy_data.tail(30).mean()
            ma_90 = dxy_data.tail(90).mean() if len(dxy_data) >= 90 else ma_30
            
            # DXY strength is bearish for gold (inverse relationship)
            short_term_change = (current_dxy - ma_30) / ma_30
            long_term_change = (current_dxy - ma_90) / ma_90
            
            # Inverse sentiment (DXY up = gold down)
            dxy_sentiment = -np.tanh(short_term_change * 20 + long_term_change * 10)
            
            return {
                'component': 'DXY_USD_Index',
                'sentiment': float(dxy_sentiment),
                'confidence': 0.85,
                'weight': self.config['component_weights']['dxy'],
                'value': float(current_dxy),
                'description': f'DXY: {current_dxy:.2f} (vs 30d avg: {ma_30:.2f})',
                'metadata': {
                    'current_dxy': float(current_dxy),
                    'ma_30': float(ma_30),
                    'ma_90': float(ma_90),
                    'short_term_change': float(short_term_change),
                    'method': 'fred_api'
                }
            }
            
        except Exception as e:
            print(f"FRED DXY fetch error: {e}")
            return None
    
    async def _fetch_dxy_yahoo(self) -> Dict:
        """Fetch DXY data from Yahoo Finance"""
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="3mo")
            
            if hist.empty:
                return None
            
            current_dxy = hist['Close'].iloc[-1]
            ma_30 = hist['Close'].tail(30).mean()
            ma_90 = hist['Close'].mean()
            
            short_term_change = (current_dxy - ma_30) / ma_30
            long_term_change = (current_dxy - ma_90) / ma_90
            
            # Inverse sentiment
            dxy_sentiment = -np.tanh(short_term_change * 20 + long_term_change * 10)
            
            return {
                'component': 'DXY_USD_Index_Yahoo',
                'sentiment': float(dxy_sentiment),
                'confidence': 0.80,
                'weight': self.config['component_weights']['dxy'],
                'value': float(current_dxy),
                'description': f'DXY: {current_dxy:.2f} (Yahoo)',
                'metadata': {
                    'current_dxy': float(current_dxy),
                    'ma_30': float(ma_30),
                    'short_term_change': float(short_term_change),
                    'method': 'yahoo_finance'
                }
            }
            
        except Exception as e:
            print(f"Yahoo DXY fetch error: {e}")
            return None
    
    async def _estimate_dxy_sentiment(self) -> Dict:
        """Estimate DXY sentiment from major currency pairs"""
        try:
            print("📊 Estimating DXY from major currency pairs...")
            
            # Fetch major currency pairs
            major_pairs = self.config['major_currencies']
            pair_data = {}
            
            for pair in major_pairs:
                try:
                    ticker = yf.Ticker(pair)
                    hist = ticker.history(period="1mo")
                    if not hist.empty:
                        pair_data[pair] = hist['Close']
                except:
                    continue
            
            if not pair_data:
                return self._create_currency_error("DXY_Estimate_Failed", "No currency data available")
            
            # Calculate average USD strength
            usd_strength_signals = []
            for pair, data in pair_data.items():
                if 'USD' in pair:
                    recent_change = (data.iloc[-1] - data.mean()) / data.mean()
                    if pair.startswith('USD'):
                        usd_strength_signals.append(recent_change)  # USD strengthening
                    else:
                        usd_strength_signals.append(-recent_change)  # USD weakening
            
            avg_usd_strength = np.mean(usd_strength_signals) if usd_strength_signals else 0
            estimated_dxy_sentiment = -np.tanh(avg_usd_strength * 15)  # Inverse for gold
            
            return {
                'component': 'DXY_Estimated',
                'sentiment': float(estimated_dxy_sentiment),
                'confidence': 0.65,
                'weight': self.config['component_weights']['dxy'] * 0.8,  # Reduced weight for estimate
                'description': f'DXY estimate from {len(pair_data)} currency pairs',
                'metadata': {
                    'avg_usd_strength': float(avg_usd_strength),
                    'pairs_analyzed': len(pair_data),
                    'method': 'currency_pair_estimate'
                }
            }
            
        except Exception as e:
            print(f"DXY estimation error: {e}")
            return self._create_currency_error("DXY_Estimation_Error", f"DXY estimation failed: {e}")

    async def fetch_em_currency_stress(self) -> Dict:
        """Analyze Emerging Market currency stress"""
        cache_key = self._get_cache_key("em_stress")
        cached_result = self._get_cached_data(cache_key)
        
        if cached_result:
            return cached_result
        
        print("📊 Analyzing Emerging Market currency stress...")
        
        try:
            em_currencies = self.config['em_currencies']
            em_data = {}
            
            # Fetch EM currency data
            for currency in em_currencies:
                try:
                    ticker = yf.Ticker(currency)
                    hist = ticker.history(period="3mo")
                    if not hist.empty:
                        em_data[currency] = hist
                except Exception as e:
                    print(f"Failed to fetch {currency}: {e}")
                    continue
            
            if not em_data:
                return self._create_currency_error("EM_Stress_No_Data", "No EM currency data available")
            
            # Calculate stress indicators
            stress_signals = []
            currency_details = []
            
            for currency, hist in em_data.items():
                try:
                    # Calculate volatility and depreciation
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    
                    # Recent depreciation vs USD (higher values = more stress)
                    recent_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30]
                    depreciation_rate = recent_change  # Positive = depreciation vs USD
                    
                    # Stress score: combines high volatility + depreciation
                    stress_score = volatility * 2 + max(0, depreciation_rate) * 3
                    stress_signals.append(stress_score)
                    
                    currency_details.append({
                        'currency': currency,
                        'volatility': volatility,
                        'recent_change': depreciation_rate,
                        'stress_score': stress_score
                    })
                    
                except Exception as e:
                    print(f"Error analyzing {currency}: {e}")
                    continue
            
            if not stress_signals:
                return self._create_currency_error("EM_Stress_Calculation_Failed", "Could not calculate EM stress")
            
            # Average stress level
            avg_stress = np.mean(stress_signals)
            max_stress = max(stress_signals)
            
            # High EM stress is bullish for gold (flight to safety)
            stress_sentiment = np.tanh(avg_stress * 2)  # Normalize to -1 to 1
            
            result = {
                'component': 'EM_Currency_Stress',
                'sentiment': float(stress_sentiment),
                'confidence': min(0.9, 0.6 + len(em_data) * 0.05),  # Higher confidence with more currencies
                'weight': self.config['component_weights']['em_stress'],
                'description': f'EM stress: {avg_stress:.3f} avg across {len(em_data)} currencies',
                'metadata': {
                    'avg_stress': float(avg_stress),
                    'max_stress': float(max_stress),
                    'currencies_analyzed': len(em_data),
                    'currency_details': currency_details,
                    'method': 'volatility_depreciation'
                }
            }
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            print(f"EM stress analysis error: {e}")
            return self._create_currency_error("EM_Stress_Error", f"EM stress analysis failed: {e}")

    async def fetch_cb_balance_sheet_expansion(self) -> Dict:
        """Analyze Central Bank balance sheet expansion"""
        cache_key = self._get_cache_key("cb_expansion")
        cached_result = self._get_cached_data(cache_key)
        
        if cached_result:
            return cached_result
        
        print("📊 Analyzing Central Bank balance sheet expansion...")
        
        try:
            cb_expansion_signals = []
            
            # Federal Reserve balance sheet
            if self.fred:
                fed_result = await self._fetch_fed_balance_sheet()
                if fed_result:
                    cb_expansion_signals.append(fed_result)
            
            # ECB, BOJ proxies (using currency strength as proxy)
            other_cb_result = await self._estimate_other_cb_expansion()
            if other_cb_result:
                cb_expansion_signals.append(other_cb_result)
            
            if not cb_expansion_signals:
                return self._create_currency_error("CB_Expansion_No_Data", "No CB balance sheet data available")
            
            # Aggregate CB expansion sentiment
            weighted_sentiment = sum(signal['sentiment'] * signal['weight'] for signal in cb_expansion_signals)
            total_weight = sum(signal['weight'] for signal in cb_expansion_signals)
            avg_confidence = np.mean([signal['confidence'] for signal in cb_expansion_signals])
            
            final_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
            
            result = {
                'component': 'CB_Balance_Sheet_Expansion',
                'sentiment': float(final_sentiment),
                'confidence': float(avg_confidence),
                'weight': self.config['component_weights']['cb_expansion'],
                'description': f'CB expansion from {len(cb_expansion_signals)} central banks',
                'metadata': {
                    'signals_analyzed': len(cb_expansion_signals),
                    'signal_details': cb_expansion_signals,
                    'method': 'multi_cb_analysis'
                }
            }
            
            self._cache_data(cache_key, result, 120)  # 2 hour cache for CB data
            return result
            
        except Exception as e:
            print(f"CB expansion analysis error: {e}")
            return self._create_currency_error("CB_Expansion_Error", f"CB expansion analysis failed: {e}")
    
    async def _fetch_fed_balance_sheet(self) -> Dict:
        """Fetch Federal Reserve balance sheet data"""
        try:
            # FRED series for Fed total assets
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year lookback
            
            fed_assets = self.fred.get_series('WALCL', 
                                            observation_start=start_date.strftime('%Y-%m-%d'),
                                            observation_end=end_date.strftime('%Y-%m-%d'))
            
            if fed_assets.empty:
                return None
            
            # Calculate expansion rate
            current_assets = fed_assets.iloc[-1]
            assets_6m_ago = fed_assets.iloc[-26] if len(fed_assets) >= 26 else fed_assets.iloc[0]
            
            expansion_rate = (current_assets - assets_6m_ago) / assets_6m_ago
            
            # Balance sheet expansion is bullish for gold
            expansion_sentiment = np.tanh(expansion_rate * 20)  # Scale to reasonable range
            
            return {
                'central_bank': 'Federal_Reserve',
                'sentiment': float(expansion_sentiment),
                'confidence': 0.85,
                'weight': 0.6,  # Fed gets 60% weight in CB component
                'expansion_rate': float(expansion_rate),
                'current_assets_trillions': float(current_assets / 1000000),  # Convert to trillions
                'method': 'fed_balance_sheet'
            }
            
        except Exception as e:
            print(f"Fed balance sheet fetch error: {e}")
            return None
    
    async def _estimate_other_cb_expansion(self) -> Dict:
        """Estimate other central bank expansion from currency weakness"""
        try:
            # Use currency weakness as proxy for CB expansion
            major_currencies = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X']
            expansion_signals = []
            
            for currency in major_currencies:
                try:
                    ticker = yf.Ticker(currency)
                    hist = ticker.history(period="6mo")
                    
                    if not hist.empty:
                        # Currency weakness suggests potential CB expansion
                        recent_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60]
                        
                        if currency == 'USDJPY=X':
                            # For USDJPY, higher values suggest JPY weakness (BOJ expansion)
                            expansion_proxy = recent_change
                        else:
                            # For EUR, GBP vs USD, lower values suggest weakness (expansion)
                            expansion_proxy = -recent_change
                        
                        expansion_signals.append(expansion_proxy)
                        
                except:
                    continue
            
            if not expansion_signals:
                return None
            
            avg_expansion_proxy = np.mean(expansion_signals)
            expansion_sentiment = np.tanh(avg_expansion_proxy * 10)
            
            return {
                'central_bank': 'Other_CB_Proxy',
                'sentiment': float(expansion_sentiment),
                'confidence': 0.65,
                'weight': 0.4,  # Other CBs get 40% weight
                'expansion_proxy': float(avg_expansion_proxy),
                'currencies_analyzed': len(expansion_signals),
                'method': 'currency_weakness_proxy'
            }
            
        except Exception as e:
            print(f"Other CB estimation error: {e}")
            return None
    
    def _create_currency_error(self, source_name: str, error_message: str) -> Dict:
        """Create error signal for currency components"""
        return {
            'component': source_name,
            'sentiment': 0.0,  # Neutral when no data
            'confidence': 0.3,
            'weight': 0.05,  # Minimal weight for error case
            'description': error_message,
            'metadata': {'error': True, 'method': 'fallback'}
        }
    
    async def analyze_currency_debasement_risk(self) -> Dict:
        """Main analysis function for currency debasement sentiment"""
        print("💱 Analyzing Currency Debasement Risk...")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # Fetch all components concurrently
            components_tasks = [
                self.fetch_dxy_sentiment(),
                self.fetch_em_currency_stress(),
                self.fetch_cb_balance_sheet_expansion()
            ]
            
            component_results = await asyncio.gather(*components_tasks, return_exceptions=True)
            
            # Filter valid components
            valid_components = []
            for result in component_results:
                if isinstance(result, dict) and 'sentiment' in result:
                    valid_components.append(result)
            
            if not valid_components:
                return self._create_currency_error("Currency_Analysis_Failed", "All currency components failed")
            
            # Calculate composite currency debasement sentiment
            total_weighted_sentiment = 0
            total_weight = 0
            component_details = []
            
            for component in valid_components:
                weight = component.get('weight', 0.33)  # Default equal weight
                sentiment = component.get('sentiment', 0)
                confidence = component.get('confidence', 0.5)
                
                # Adjust weight by confidence
                adjusted_weight = weight * confidence
                total_weighted_sentiment += sentiment * adjusted_weight
                total_weight += adjusted_weight
                
                component_details.append({
                    'name': component.get('component', 'Unknown'),
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'weight': weight,
                    'contribution': sentiment * adjusted_weight
                })
            
            final_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
            avg_confidence = np.mean([comp['confidence'] for comp in valid_components])
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Interpretation
            if final_sentiment > 0.5:
                interpretation = "🟢 HIGH DEBASEMENT RISK - Very bullish for gold"
            elif final_sentiment > 0.2:
                interpretation = "🟢 MODERATE DEBASEMENT RISK - Bullish for gold"
            elif final_sentiment > -0.2:
                interpretation = "🟡 NEUTRAL - Balanced currency environment"
            elif final_sentiment > -0.5:
                interpretation = "🔴 LOW DEBASEMENT RISK - USD strength pressure"
            else:
                interpretation = "🔴 STRONG CURRENCIES - Bearish for gold"
            
            result = {
                'source': 'Currency_Debasement_Risk_Multi',
                'sentiment': float(final_sentiment),
                'confidence': float(avg_confidence),
                'weight': 0.10,  # 10% weight in composite gold sentiment
                'description': f'Currency debasement risk from {len(valid_components)} components',
                'interpretation': interpretation,
                'execution_time_seconds': execution_time,
                'analysis_timestamp': datetime.now().isoformat(),
                'kpi_name': 'Currency_Debasement_Risk',
                'framework_weight': 0.10,
                'metadata': {
                    'components_analyzed': len(valid_components),
                    'component_details': component_details,
                    'method': 'multi_component_analysis',
                    'component_weights': self.config['component_weights']
                }
            }
            
            print(f"✅ Currency Analysis completed in {execution_time:.1f}s")
            print(f"📊 Sentiment: {final_sentiment:.3f} ({interpretation})")
            print(f"🎯 Confidence: {avg_confidence:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Currency analysis failed: {e}")
            return self._create_currency_error("Currency_Analysis_Exception", f"Analysis exception: {e}")

# Integration test function
async def test_currency_analyzer():
    """Test the Currency Debasement analyzer"""
    analyzer = CurrencyDebasementAnalyzer()
    result = await analyzer.analyze_currency_debasement_risk()
    
    print("\n" + "="*60)
    print("CURRENCY DEBASEMENT ANALYZER TEST RESULTS")
    print("="*60)
    
    if result:
        print(f"Source: {result.get('source', 'Unknown')}")
        print(f"Sentiment: {result.get('sentiment', 0):.3f}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"Weight: {result.get('weight', 0):.1%}")
        print(f"Description: {result.get('description', 'N/A')}")
        print(f"Interpretation: {result.get('interpretation', 'N/A')}")
        
        if 'metadata' in result and 'component_details' in result['metadata']:
            print("\nComponent Analysis:")
            for comp in result['metadata']['component_details']:
                print(f"  {comp['name']}: {comp['sentiment']:+.3f} (conf: {comp['confidence']:.2f}, weight: {comp['weight']:.2f})")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_currency_analyzer())
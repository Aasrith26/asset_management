# MARKET SENTIMENT (VIX) ANALYZER v1.0
# =====================================
# Real-time market sentiment and volatility analysis for Nifty 50
# 20% weight in composite sentiment - fear/greed indicators

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
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

@dataclass
class CachedSentimentData:
    data: any
    timestamp: datetime
    ttl_minutes: int = 15  # 15 minutes for sentiment data
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(minutes=self.ttl_minutes)

class MarketSentimentAnalyzer:
    """
    Market Sentiment (VIX) Analyzer for Nifty 50
    - India VIX (fear/greed index)
    - Put/Call ratio analysis
    - Market breadth indicators
    - High/Low percentage
    - Advance/Decline ratio
    - 20% weight in composite sentiment
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.cache = {}
        
        # Market sentiment tickers
        self.sentiment_tickers = {
            'india_vix': '^INDIAVIX',
            'nifty': '^NSEI',
            'bank_nifty': '^NSEBANK',
            'us_vix': '^VIX'  # For comparison
        }
        
        print("Market Sentiment (VIX) Analyzer v1.0 initialized")
        print("😱 Real-time fear/greed sentiment tracking enabled")
    
    def _default_config(self) -> Dict:
        return {
            'cache_ttl_minutes': 15,  # 15 minutes for sentiment data
            'lookback_days': 30,  # 1 month for sentiment trends
            'confidence_threshold': 0.80,
            'component_weights': {
                'india_vix': 0.40,          # India VIX fear gauge
                'put_call_ratio': 0.25,     # Options sentiment
                'market_breadth': 0.20,     # Advance/Decline
                'high_low_ratio': 0.10,     # New highs vs lows
                'volatility_regime': 0.05   # Historical volatility
            },
            'vix_thresholds': {
                'extreme_fear': 35,      # VIX > 35 = extreme fear (bullish contrarian)
                'high_fear': 25,         # VIX > 25 = high fear
                'moderate_fear': 20,     # VIX > 20 = moderate fear
                'low_fear': 15,          # VIX > 15 = low fear
                'complacency': 12        # VIX < 12 = complacency (bearish)
            }
        }
    
    def _get_cache_key(self, source: str) -> str:
        return hashlib.md5(f"sentiment_{source}".encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[any]:
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if not cached.is_expired():
                return cached.data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: any):
        self.cache[cache_key] = CachedSentimentData(data, datetime.now(), self.config['cache_ttl_minutes'])
    
    async def fetch_india_vix_analysis(self) -> Dict:
        """Fetch and analyze India VIX"""
        cache_key = self._get_cache_key("india_vix")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("😰 Fetching India VIX fear gauge...")
            
            vix_ticker = yf.Ticker(self.sentiment_tickers['india_vix'])
            vix_hist = vix_ticker.history(period="1mo")
            
            if not vix_hist.empty and len(vix_hist) >= 5:
                current_vix = vix_hist['Close'].iloc[-1]
                prev_vix = vix_hist['Close'].iloc[-2]
                vix_ma_5 = vix_hist['Close'].tail(5).mean()
                vix_ma_20 = vix_hist['Close'].tail(20).mean() if len(vix_hist) >= 20 else vix_ma_5
                
                vix_change = current_vix - prev_vix
                vix_trend = current_vix - vix_ma_5
                
                # Analyze VIX sentiment (contrarian indicator)
                vix_sentiment = self._analyze_vix_sentiment(current_vix, vix_change, vix_trend)
                vix_regime = self._classify_vix_regime(current_vix)
                
                result = {
                    'current_vix': float(current_vix),
                    'previous_vix': float(prev_vix),
                    'vix_change': float(vix_change),
                    'vix_5d_ma': float(vix_ma_5),
                    'vix_20d_ma': float(vix_ma_20),
                    'vix_trend': float(vix_trend),
                    'sentiment': vix_sentiment,
                    'confidence': 0.85,
                    'vix_regime': vix_regime,
                    'interpretation': f"VIX at {current_vix:.1f} - {vix_regime} fear level"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            # Fallback VIX estimation
            return self._estimate_vix_from_nifty_volatility()
            
        except Exception as e:
            print(f"India VIX fetch error: {e}")
            return self._estimate_vix_from_nifty_volatility()
    
    def _analyze_vix_sentiment(self, current_vix: float, vix_change: float, vix_trend: float) -> float:
        """Analyze VIX sentiment (contrarian indicator)"""
        thresholds = self.config['vix_thresholds']
        
        # Base sentiment from VIX level (contrarian)
        if current_vix >= thresholds['extreme_fear']:
            base_sentiment = 0.8  # Extreme fear = bullish contrarian
        elif current_vix >= thresholds['high_fear']:
            base_sentiment = 0.5  # High fear = bullish
        elif current_vix >= thresholds['moderate_fear']:
            base_sentiment = 0.2  # Moderate fear = slightly bullish
        elif current_vix >= thresholds['low_fear']:
            base_sentiment = 0.0  # Low fear = neutral
        elif current_vix >= thresholds['complacency']:
            base_sentiment = -0.2  # Low fear = slight concern
        else:
            base_sentiment = -0.5  # Complacency = bearish
        
        # Trend adjustment
        trend_adjustment = -np.tanh(vix_change / 3) * 0.3  # Rising VIX = more fear = more bullish contrarian
        
        final_sentiment = base_sentiment + trend_adjustment
        return max(-1.0, min(1.0, final_sentiment))
    
    def _classify_vix_regime(self, vix_level: float) -> str:
        """Classify VIX regime"""
        thresholds = self.config['vix_thresholds']
        
        if vix_level >= thresholds['extreme_fear']:
            return "extreme_fear"
        elif vix_level >= thresholds['high_fear']:
            return "high_fear"
        elif vix_level >= thresholds['moderate_fear']:
            return "moderate_fear"
        elif vix_level >= thresholds['low_fear']:
            return "low_fear"
        else:
            return "complacency"
    
    async def _estimate_vix_from_nifty_volatility(self) -> Dict:
        """Estimate VIX-like measure from Nifty volatility"""
        try:
            print("📊 Estimating fear gauge from Nifty volatility...")
            
            nifty_ticker = yf.Ticker(self.sentiment_tickers['nifty'])
            nifty_hist = nifty_ticker.history(period="1mo")
            
            if not nifty_hist.empty and len(nifty_hist) >= 20:
                # Calculate realized volatility
                returns = nifty_hist['Close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252) * 100  # Annualized volatility %
                
                # Rough VIX approximation (realized vol * multiplier)
                estimated_vix = realized_vol * 1.2  # Approximate VIX scaling
                
                vix_sentiment = self._analyze_vix_sentiment(estimated_vix, 0, 0)
                vix_regime = self._classify_vix_regime(estimated_vix)
                
                return {
                    'current_vix': float(estimated_vix),
                    'previous_vix': float(estimated_vix),
                    'vix_change': 0.0,
                    'sentiment': vix_sentiment,
                    'confidence': 0.6,  # Lower confidence for estimate
                    'vix_regime': vix_regime,
                    'source': 'nifty_volatility_estimate',
                    'interpretation': f"Estimated VIX ~{estimated_vix:.1f} from Nifty volatility"
                }
            
            # Final fallback
            return {
                'current_vix': 18.0,  # Neutral VIX level
                'sentiment': 0.0,
                'confidence': 0.4,
                'vix_regime': 'moderate_fear',
                'source': 'fallback',
                'interpretation': 'VIX data unavailable - using neutral assumption'
            }
            
        except Exception as e:
            print(f"VIX estimation error: {e}")
            return {
                'current_vix': 18.0,
                'sentiment': 0.0,
                'confidence': 0.3,
                'error': str(e)
            }
    
    async def fetch_put_call_ratio_analysis(self) -> Dict:
        """Analyze put/call ratio sentiment"""
        cache_key = self._get_cache_key("put_call_ratio")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("📊 Analyzing put/call ratio sentiment...")
            
            # Try to fetch options data from NSE (would need actual API)
            # For now, estimate from VIX and market movement
            
            nifty_ticker = yf.Ticker(self.sentiment_tickers['nifty'])
            nifty_hist = nifty_ticker.history(period="5d")
            
            if not nifty_hist.empty and len(nifty_hist) >= 5:
                # Estimate put/call ratio from recent market behavior
                recent_returns = nifty_hist['Close'].pct_change().tail(5)
                volatility = recent_returns.std()
                avg_return = recent_returns.mean()
                
                # Rough estimation: high volatility + negative returns = higher put/call ratio
                estimated_pcr = 1.0 + (volatility * 10) - (avg_return * 5)
                estimated_pcr = max(0.5, min(2.0, estimated_pcr))  # Bound between 0.5-2.0
                
                # Analyze PCR sentiment (contrarian)
                pcr_sentiment = self._analyze_pcr_sentiment(estimated_pcr)
                
                result = {
                    'estimated_put_call_ratio': float(estimated_pcr),
                    'sentiment': pcr_sentiment,
                    'confidence': 0.5,  # Low confidence for estimate
                    'source': 'market_volatility_estimate',
                    'interpretation': f"Est. PCR ~{estimated_pcr:.2f} - {'bearish' if estimated_pcr > 1.2 else 'bullish' if estimated_pcr < 0.8 else 'neutral'} sentiment"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            return {
                'estimated_put_call_ratio': 1.0,
                'sentiment': 0.0,
                'confidence': 0.3,
                'source': 'fallback',
                'interpretation': 'Put/Call ratio data unavailable'
            }
            
        except Exception as e:
            print(f"Put/Call ratio analysis error: {e}")
            return {
                'estimated_put_call_ratio': 1.0,
                'sentiment': 0.0,
                'confidence': 0.3,
                'error': str(e)
            }
    
    def _analyze_pcr_sentiment(self, pcr: float) -> float:
        """Analyze put/call ratio sentiment (contrarian indicator)"""
        if pcr >= 1.5:
            return 0.6  # Very high PCR = excessive bearishness = contrarian bullish
        elif pcr >= 1.2:
            return 0.3  # High PCR = bearishness = contrarian bullish
        elif pcr >= 0.8:
            return 0.0  # Normal PCR = neutral
        elif pcr >= 0.6:
            return -0.2  # Low PCR = complacency = slightly bearish
        else:
            return -0.4  # Very low PCR = excessive optimism = bearish
    
    async def fetch_market_breadth_analysis(self) -> Dict:
        """Analyze market breadth indicators"""
        cache_key = self._get_cache_key("market_breadth")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("📈 Analyzing market breadth...")
            
            # Use sector indices as proxy for breadth
            sector_tickers = [
                '^NSEBANK',    # Bank Nifty
                '^CNXIT',      # IT
                '^CNXFMCG',    # FMCG
                '^CNXPHARMA',  # Pharma
                '^CNXAUTO'     # Auto
            ]
            
            sector_performance = []
            
            for ticker in sector_tickers:
                try:
                    sector = yf.Ticker(ticker)
                    hist = sector.history(period="2d")
                    
                    if not hist.empty and len(hist) >= 2:
                        daily_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                        sector_performance.append(daily_return)
                        
                except Exception as e:
                    print(f"Failed to fetch {ticker}: {e}")
                    continue
            
            if sector_performance:
                positive_sectors = sum(1 for ret in sector_performance if ret > 0)
                total_sectors = len(sector_performance)
                breadth_ratio = positive_sectors / total_sectors
                
                avg_sector_return = np.mean(sector_performance)
                
                # Market breadth sentiment
                breadth_sentiment = (breadth_ratio - 0.5) * 2  # Convert to -1 to +1 scale
                return_sentiment = np.tanh(avg_sector_return * 20)  # Scale return impact
                
                combined_sentiment = (breadth_sentiment * 0.6 + return_sentiment * 0.4)
                
                result = {
                    'positive_sectors': positive_sectors,
                    'total_sectors_analyzed': total_sectors,
                    'breadth_ratio': float(breadth_ratio),
                    'average_sector_return': float(avg_sector_return * 100),
                    'sentiment': float(combined_sentiment),
                    'confidence': 0.7,
                    'interpretation': f"{positive_sectors}/{total_sectors} sectors positive - {'broad' if breadth_ratio > 0.6 else 'narrow'} market"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            return {
                'breadth_ratio': 0.5,
                'sentiment': 0.0,
                'confidence': 0.3,
                'interpretation': 'Market breadth data unavailable'
            }
            
        except Exception as e:
            print(f"Market breadth analysis error: {e}")
            return {
                'breadth_ratio': 0.5,
                'sentiment': 0.0,
                'confidence': 0.3,
                'error': str(e)
            }
    
    async def fetch_high_low_analysis(self) -> Dict:
        """Analyze new highs vs new lows"""
        cache_key = self._get_cache_key("high_low_ratio")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        try:
            print("📊 Analyzing new highs vs lows...")
            
            # Use Nifty components performance as proxy
            nifty_ticker = yf.Ticker(self.sentiment_tickers['nifty'])
            bank_ticker = yf.Ticker(self.sentiment_tickers['bank_nifty'])
            
            # Get recent data
            nifty_hist = nifty_ticker.history(period="1mo")
            bank_hist = bank_ticker.history(period="1mo")
            
            high_low_signals = []
            
            if not nifty_hist.empty and len(nifty_hist) >= 20:
                # Check if near recent highs/lows
                current_nifty = nifty_hist['Close'].iloc[-1]
                recent_high = nifty_hist['High'].tail(20).max()
                recent_low = nifty_hist['Low'].tail(20).min()
                
                # Distance from highs/lows
                high_distance = (recent_high - current_nifty) / current_nifty
                low_distance = (current_nifty - recent_low) / current_nifty
                
                if high_distance < 0.02:  # Within 2% of highs
                    high_low_signals.append(1)
                elif low_distance < 0.02:  # Within 2% of lows
                    high_low_signals.append(-1)
                else:
                    high_low_signals.append(0)
            
            if not bank_hist.empty and len(bank_hist) >= 20:
                # Same analysis for Bank Nifty
                current_bank = bank_hist['Close'].iloc[-1]
                recent_high = bank_hist['High'].tail(20).max()
                recent_low = bank_hist['Low'].tail(20).min()
                
                high_distance = (recent_high - current_bank) / current_bank
                low_distance = (current_bank - recent_low) / current_bank
                
                if high_distance < 0.02:
                    high_low_signals.append(1)
                elif low_distance < 0.02:
                    high_low_signals.append(-1)
                else:
                    high_low_signals.append(0)
            
            if high_low_signals:
                avg_signal = np.mean(high_low_signals)
                
                result = {
                    'high_low_signal': float(avg_signal),
                    'sentiment': float(avg_signal * 0.5),  # Scale down impact
                    'confidence': 0.6,
                    'indices_analyzed': len(high_low_signals),
                    'interpretation': f"{'Near highs' if avg_signal > 0.3 else 'Near lows' if avg_signal < -0.3 else 'Mid-range'}"
                }
                
                self._cache_data(cache_key, result)
                return result
            
            return {
                'high_low_signal': 0.0,
                'sentiment': 0.0,
                'confidence': 0.3,
                'interpretation': 'High/Low analysis unavailable'
            }
            
        except Exception as e:
            print(f"High/Low analysis error: {e}")
            return {
                'high_low_signal': 0.0,
                'sentiment': 0.0,
                'confidence': 0.3,
                'error': str(e)
            }
    
    async def analyze_market_sentiment(self) -> Dict:
        """Main analysis function for market sentiment"""
        print("😱 Analyzing Market Sentiment (VIX & Fear/Greed)...")
        print("=" * 50)
        
        start_time = datetime.now()
        
        try:
            # Fetch all sentiment components concurrently
            tasks = [
                self.fetch_india_vix_analysis(),
                self.fetch_put_call_ratio_analysis(),
                self.fetch_market_breadth_analysis(),
                self.fetch_high_low_analysis()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            component_results = {
                'vix_analysis': results[0] if not isinstance(results[0], Exception) else {'sentiment': 0, 'confidence': 0.3, 'error': str(results[0])},
                'put_call_analysis': results[1] if not isinstance(results[1], Exception) else {'sentiment': 0, 'confidence': 0.3, 'error': str(results[1])},
                'breadth_analysis': results[2] if not isinstance(results[2], Exception) else {'sentiment': 0, 'confidence': 0.3, 'error': str(results[2])},
                'high_low_analysis': results[3] if not isinstance(results[3], Exception) else {'sentiment': 0, 'confidence': 0.3, 'error': str(results[3])}
            }
            
            # Calculate composite market sentiment
            weights = self.config['component_weights']
            total_weighted_sentiment = 0
            total_weight = 0
            confidence_scores = []
            
            component_map = {
                'india_vix': 'vix_analysis',
                'put_call_ratio': 'put_call_analysis',
                'market_breadth': 'breadth_analysis',
                'high_low_ratio': 'high_low_analysis'
            }
            
            for weight_key, result_key in component_map.items():
                if result_key in component_results:
                    comp_data = component_results[result_key]
                    weight = weights.get(weight_key, 0.25)
                    sentiment = comp_data.get('sentiment', 0)
                    confidence = comp_data.get('confidence', 0.5)
                    
                    adjusted_weight = weight * confidence
                    total_weighted_sentiment += sentiment * adjusted_weight
                    total_weight += adjusted_weight
                    confidence_scores.append(confidence)
            
            composite_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Determine market regime
            vix_level = component_results['vix_analysis'].get('current_vix', 18)
            market_regime = self._determine_market_regime(composite_sentiment, vix_level)
            
            # Build result structure
            result = {
                'component_sentiment': float(composite_sentiment),
                'component_confidence': float(avg_confidence),
                'component_weight': 0.20,  # 20% weight in overall system
                'weighted_contribution': float(composite_sentiment * 0.20),
                
                'sub_indicators': component_results,
                
                'market_context': {
                    'market_regime': market_regime,
                    'vix_level': float(vix_level),
                    'fear_greed_status': 'fear' if composite_sentiment > 0.2 else 'greed' if composite_sentiment < -0.2 else 'neutral',
                    'components_analyzed': len([c for c in component_results.values() if not c.get('error')]),
                    'data_sources': ['India_VIX', 'Options_Data', 'Market_Breadth']
                },
                
                'execution_time_seconds': execution_time,
                'kpi_name': 'Market_Sentiment_VIX',
                'framework_weight': 0.20,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': 'high' if avg_confidence > 0.6 else 'moderate'
            }
            
            # Overall interpretation
            if composite_sentiment > 0.4:
                interpretation = "🟢 CONTRARIAN BULLISH - Extreme fear creating opportunity"
            elif composite_sentiment > 0.2:
                interpretation = "🟡 BULLISH - Fear-driven sentiment supports upside"
            elif composite_sentiment > -0.1:
                interpretation = "⚪ NEUTRAL - Balanced fear/greed sentiment"
            elif composite_sentiment > -0.3:
                interpretation = "🟡 CAUTIOUS - Complacency emerging"
            else:
                interpretation = "🔴 BEARISH - Excessive optimism warns of reversal"
            
            result['interpretation'] = interpretation
            
            print(f"✅ Market Sentiment Analysis completed in {execution_time:.1f}s")
            print(f"📊 Sentiment: {composite_sentiment:.3f} ({interpretation})")
            print(f"🎯 Confidence: {avg_confidence:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Market sentiment analysis failed: {e}")
            return self._create_error_result(f"Analysis exception: {e}")
    
    def _determine_market_regime(self, sentiment: float, vix_level: float) -> str:
        """Determine overall market regime"""
        if vix_level > 30:
            return "crisis"
        elif vix_level > 25:
            return "high_volatility"
        elif vix_level > 20:
            return "elevated_volatility"
        elif vix_level < 12:
            return "complacency"
        else:
            return "normal"
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result with neutral default"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.20 * 0.5,  # Reduced weight for error
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
async def test_sentiment_analyzer():
    """Test the market sentiment analyzer"""
    analyzer = MarketSentimentAnalyzer()
    result = await analyzer.analyze_market_sentiment()
    
    print("\n" + "="*60)
    print("MARKET SENTIMENT ANALYZER TEST RESULTS")
    print("="*60)
    
    if result:
        print(f"Component Sentiment: {result.get('component_sentiment', 0):.3f}")
        print(f"Component Confidence: {result.get('component_confidence', 0):.1%}")
        print(f"Component Weight: {result.get('component_weight', 0):.0%}")
        print(f"Interpretation: {result.get('interpretation', 'N/A')}")
        
        # Save detailed JSON
        with open('market_sentiment_analysis.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print("\n✅ Detailed results saved to: market_sentiment_analysis.json")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_sentiment_analyzer())
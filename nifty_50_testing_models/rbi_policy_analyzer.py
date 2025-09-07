# RBI INTEREST RATES & MONETARY POLICY ANALYZER v1.1 - FIXED
# ===========================================================
# Real-time RBI policy rates and monetary policy analysis for Nifty 50
# 10% weight in composite sentiment - policy impact
# FIXED: Bond ticker errors, improved data sources

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional
import json
import hashlib
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re

warnings.filterwarnings('ignore')


@dataclass
class CachedRBIData:
    data: any
    timestamp: datetime
    ttl_minutes: int = 240  # 4 hours for policy data

    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(minutes=self.ttl_minutes)


class RBIInterestRatesAnalyzer:
    """
    RBI Interest Rates & Monetary Policy Sentiment Analyzer v1.1
    - RBI repo rate tracking with improved sources
    - Banking sector yield proxy (FIXED - no more 404 errors)
    - Policy meeting proximity detection
    - Credit growth monitoring via banking performance
    - 10% weight in composite sentiment
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.cache = {}

        print("RBI Interest Rates & Policy Analyzer v1.1 initialized")
        print("🏛️ Real-time monetary policy impact tracking enabled")
        print("🔧 FIXED: Using banking sector proxy for bond yields")

    def _default_config(self) -> Dict:
        return {
            'cache_ttl_minutes': 240,  # 4 hours for policy data
            'lookback_days': 30,  # 1 month for rate trends
            'confidence_threshold': 0.75,
            'component_weights': {
                'repo_rate_trend': 0.35,  # RBI repo rate changes
                'banking_sector_proxy': 0.30,  # Banking sector as yield proxy (IMPROVED)
                'policy_proximity': 0.20,  # Distance to next RBI meeting
                'credit_growth': 0.15  # Credit growth trends
            },
            'rate_sensitivity': {
                'repo_cut_positive': 0.8,  # Rate cut = bullish
                'repo_hike_negative': -0.6,  # Rate hike = bearish
                'yield_threshold': 0.25  # 25 bps yield change significance
            },
            # FIXED: Use working tickers instead of failing NSE bond tickers
            'working_tickers': {
                'bank_nifty': '^NSEBANK',
                'sbi': 'SBIN.NS',
                'hdfc_bank': 'HDFCBANK.NS',
                'icici_bank': 'ICICIBANK.NS',
                'kotak_bank': 'KOTAKBANK.NS',
                'axis_bank': 'AXISBANK.NS',
                'us_10y_reference': '^TNX'
            },
            'current_known_rates': {
                'repo_rate': 6.50,  # Current repo rate (Sept 2025)
                'reverse_repo': 3.35,  # Current reverse repo rate
                'bank_rate': 6.75,  # Current bank rate
                'crr': 4.5,  # Current Cash Reserve Ratio
                'slr': 18.0  # Current Statutory Liquidity Ratio
            }
        }

    def _get_cache_key(self, source: str) -> str:
        return hashlib.md5(f"rbi_{source}".encode()).hexdigest()

    def _get_cached_data(self, cache_key: str) -> Optional[any]:
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if not cached.is_expired():
                return cached.data
            else:
                del self.cache[cache_key]
        return None

    def _cache_data(self, cache_key: str, data: any):
        self.cache[cache_key] = CachedRBIData(data, datetime.now(), self.config['cache_ttl_minutes'])

    async def fetch_current_repo_rate_improved(self) -> Dict:
        """Fetch current RBI repo rate with multiple improved methods"""
        cache_key = self._get_cache_key("repo_rate_improved")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached

        try:
            print("Fetching RBI repo rate data...")

            # Method 1: Try direct RBI website scraping
            try:
                repo_rate, source = await self._scrape_rbi_repo_rate_improved()
                if repo_rate:
                    result = self._build_repo_rate_result(repo_rate, source, 0.85)
                    self._cache_data(cache_key, result)
                    return result
            except Exception as e:
                print(f"RBI website scraping failed: {e}")

            # Method 2: Try financial news sites
            try:
                repo_rate, source = await self._scrape_financial_news_repo_rate()
                if repo_rate:
                    result = self._build_repo_rate_result(repo_rate, source, 0.75)
                    self._cache_data(cache_key, result)
                    return result
            except Exception as e:
                print(f"Financial news scraping failed: {e}")

            # Method 3: Use current known rate
            current_rate = self.config['current_known_rates']['repo_rate']
            result = self._build_repo_rate_result(current_rate, 'current_known_rate', 0.80)
            self._cache_data(cache_key, result)
            return result

        except Exception as e:
            print(f"All repo rate methods failed: {e}")
            return self._build_repo_rate_result(6.5, 'final_fallback', 0.60, str(e))

    async def _scrape_rbi_repo_rate_improved(self) -> tuple:
        """Improved RBI website scraping"""
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )

        try:
            # Try RBI's current rates page
            rbi_urls = [
                "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
                "https://www.rbi.org.in/Scripts/NotificationUser.aspx",
                "https://www.rbi.org.in/home.aspx"
            ]

            for url in rbi_urls:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()

                            # Enhanced repo rate patterns
                            repo_patterns = [
                                r'repo\s+rate.*?(\d+\.?\d*)\s*%',
                                r'policy\s+repo\s+rate.*?(\d+\.?\d*)\s*%',
                                r'repo.*?(\d+\.?\d*)\s*per\s+cent',
                                r'(\d+\.?\d*)\s*%.*?repo\s+rate',
                                r'current.*?repo.*?(\d+\.?\d*)',
                                r'repo.*?rate.*?(\d+\.?\d*)'
                            ]

                            for pattern in repo_patterns:
                                matches = re.findall(pattern, html, re.IGNORECASE | re.MULTILINE)
                                for match in matches:
                                    try:
                                        rate = float(match)
                                        if 3.0 <= rate <= 10.0:  # Reasonable repo rate range
                                            await session.close()
                                            return rate, "rbi_website_scraped"
                                    except:
                                        continue

                except Exception as e:
                    print(f"Failed to scrape {url}: {e}")
                    continue

            await session.close()
            return None, None

        except Exception as e:
            await session.close()
            raise e

    async def _scrape_financial_news_repo_rate(self) -> tuple:
        """Try to get repo rate from financial news sites"""
        # This is a placeholder - could be enhanced with actual news scraping
        # For now, return None to trigger fallback
        return None, None

    def _build_repo_rate_result(self, repo_rate: float, source: str, confidence: float, error: str = None) -> Dict:
        """Build standardized repo rate result"""
        repo_sentiment = self._analyze_repo_rate_sentiment(repo_rate)

        result = {
            'current_repo_rate': float(repo_rate),
            'sentiment': repo_sentiment,
            'confidence': confidence,
            'source': source,
            'interpretation': f"Repo rate at {repo_rate}% - {'accommodative' if repo_rate < 6.0 else 'neutral' if repo_rate < 7.0 else 'restrictive'}",
            'policy_stance': 'accommodative' if repo_rate < 6.0 else 'neutral' if repo_rate < 7.0 else 'restrictive',
            'last_updated': datetime.now().isoformat()
        }

        if error:
            result['error'] = error

        return result

    def _analyze_repo_rate_sentiment(self, repo_rate: float) -> float:
        """Analyze repo rate sentiment impact on equity markets"""
        # Lower rates are generally positive for equities
        if repo_rate <= 4.5:
            return 0.8  # Very accommodative
        elif repo_rate <= 5.5:
            return 0.5  # Accommodative
        elif repo_rate <= 6.5:
            return 0.1  # Mildly accommodative
        elif repo_rate <= 7.5:
            return -0.1  # Mildly restrictive
        elif repo_rate <= 8.5:
            return -0.4  # Restrictive
        else:
            return -0.7  # Very restrictive

    async def fetch_banking_sector_yield_proxy(self) -> Dict:
        """FIXED: Use banking sector performance as bond yield proxy"""
        cache_key = self._get_cache_key("banking_yield_proxy")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached

        try:
            print("📊 Fetching banking sector yield proxy (IMPROVED METHOD)...")

            working_tickers = self.config['working_tickers']
            bank_performance = {}

            # Fetch major bank stocks and Bank Nifty
            for bank_name, ticker in working_tickers.items():
                if bank_name == 'us_10y_reference':
                    continue

                try:
                    bank_ticker = yf.Ticker(ticker)
                    hist = bank_ticker.history(period="1mo")

                    if not hist.empty and len(hist) >= 10:
                        current_price = hist['Close'].iloc[-1]
                        price_1w_ago = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]
                        price_2w_ago = hist['Close'].iloc[-10] if len(hist) >= 10 else hist['Close'].iloc[0]
                        price_1m_ago = hist['Close'].iloc[0]

                        # Calculate multiple timeframe returns
                        weekly_return = (current_price - price_1w_ago) / price_1w_ago if price_1w_ago > 0 else 0
                        biweekly_return = (current_price - price_2w_ago) / price_2w_ago if price_2w_ago > 0 else 0
                        monthly_return = (current_price - price_1m_ago) / price_1m_ago if price_1m_ago > 0 else 0

                        # Calculate volatility as yield proxy
                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() if len(returns) > 5 else 0

                        bank_performance[bank_name] = {
                            'weekly_return': float(weekly_return),
                            'biweekly_return': float(biweekly_return),
                            'monthly_return': float(monthly_return),
                            'volatility': float(volatility),
                            'current_price': float(current_price),
                            'ticker': ticker
                        }

                        print(f"{bank_name}: {weekly_return:+.2%} (1W), {monthly_return:+.2%} (1M)")

                except Exception as e:
                    print(f"Failed to fetch {bank_name} ({ticker}): {e}")
                    continue

            if len(bank_performance) >= 2:  # Need at least 2 banks for reliability
                # Calculate aggregate metrics
                weekly_returns = [perf['weekly_return'] for perf in bank_performance.values()]
                monthly_returns = [perf['monthly_return'] for perf in bank_performance.values()]
                volatilities = [perf['volatility'] for perf in bank_performance.values()]

                avg_weekly_return = np.mean(weekly_returns)
                avg_monthly_return = np.mean(monthly_returns)
                avg_volatility = np.mean(volatilities)

                # Banking sector performance correlation with interest rate expectations
                # Strong banks = lower rate expectations = positive for equity markets
                # Weak banks = higher rate expectations = negative for equity markets

                # Combine returns and volatility for yield proxy
                performance_score = avg_weekly_return * 0.6 + avg_monthly_return * 0.4
                volatility_adjustment = -avg_volatility * 0.3  # High volatility = uncertainty = slightly negative

                combined_score = performance_score + volatility_adjustment

                # Convert to bond yield equivalent sentiment
                yield_proxy_sentiment = np.tanh(combined_score * 8)  # Scale to -1 to +1

                # Get US 10Y for reference
                us_yield_ref = None
                try:
                    us_ticker = yf.Ticker(working_tickers['us_10y_reference'])
                    us_hist = us_ticker.history(period="2d")
                    if not us_hist.empty and len(us_hist) >= 2:
                        us_yield_ref = float(us_hist['Close'].iloc[-1])
                        us_yield_change = float(us_hist['Close'].iloc[-1] - us_hist['Close'].iloc[-2])
                except Exception as e:
                    print(f"   US 10Y reference failed: {e}")
                    us_yield_ref = 4.5  # Fallback
                    us_yield_change = 0

                result = {
                    'banking_sector_performance': bank_performance,
                    'avg_weekly_return': float(avg_weekly_return),
                    'avg_monthly_return': float(avg_monthly_return),
                    'avg_volatility': float(avg_volatility),
                    'performance_score': float(performance_score),
                    'combined_score': float(combined_score),
                    'sentiment': float(yield_proxy_sentiment),
                    'confidence': 0.78,  # Good confidence for banking sector proxy
                    'us_10y_reference': us_yield_ref,
                    'us_yield_change': us_yield_change if 'us_yield_change' in locals() else 0,
                    'banks_analyzed': len(bank_performance),
                    'method': 'banking_sector_proxy',
                    'interpretation': f"Banking sector {'outperforming' if avg_weekly_return > 0.02 else 'underperforming' if avg_weekly_return < -0.02 else 'neutral'} - {'positive' if yield_proxy_sentiment > 0.1 else 'negative' if yield_proxy_sentiment < -0.1 else 'neutral'} yield proxy"
                }

                self._cache_data(cache_key, result)
                return result

            # Insufficient banking data
            return {
                'sentiment': 0.0,
                'confidence': 0.5,
                'banks_analyzed': len(bank_performance),
                'interpretation': 'Insufficient banking sector data for yield proxy',
                'method': 'insufficient_data_fallback'
            }

        except Exception as e:
            print(f"Banking sector yield proxy error: {e}")
            return {
                'sentiment': 0.0,
                'confidence': 0.4,
                'error': str(e),
                'method': 'error_fallback'
            }

    async def fetch_policy_meeting_proximity_improved(self) -> Dict:
        """Improved RBI policy meeting proximity analysis"""
        cache_key = self._get_cache_key("policy_meeting_improved")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached

        try:
            print("📅 Analyzing RBI policy meeting proximity...")

            now = datetime.now()

            # RBI MPC meeting schedule 2025-2026 (approximate)
            # Typically meets 6 times a year: Feb, Apr, Jun, Aug, Oct, Dec
            upcoming_meetings_2025 = [
                datetime(2025, 10, 9),  # October 2025
                datetime(2025, 12, 6),  # December 2025
            ]

            upcoming_meetings_2026 = [
                datetime(2026, 2, 7),  # February 2026
                datetime(2026, 4, 8),  # April 2026
                datetime(2026, 6, 5),  # June 2026
                datetime(2026, 8, 6),  # August 2026
                datetime(2026, 10, 7),  # October 2026
                datetime(2026, 12, 4),  # December 2026
            ]

            all_meetings = upcoming_meetings_2025 + upcoming_meetings_2026

            # Find next meeting
            future_meetings = [meeting for meeting in all_meetings if meeting > now]
            next_meeting = min(future_meetings) if future_meetings else datetime(2026, 2, 7)

            days_to_meeting = (next_meeting - now).days
            weeks_to_meeting = days_to_meeting / 7

            # Policy proximity sentiment impact
            # Markets typically get nervous before policy meetings
            if days_to_meeting <= 3:
                proximity_sentiment = -0.3  # High uncertainty just before meeting
            elif days_to_meeting <= 7:
                proximity_sentiment = -0.2  # Moderate uncertainty
            elif days_to_meeting <= 14:
                proximity_sentiment = -0.1  # Slight uncertainty
            elif days_to_meeting <= 30:
                proximity_sentiment = 0.05  # Slight positive (clarity period)
            elif days_to_meeting <= 45:
                proximity_sentiment = 0.1  # Policy clarity period
            else:
                proximity_sentiment = 0.0  # Neutral

            # Adjust based on recent policy pattern
            # If last meeting was a change, market expects more stability
            # If last meeting was status quo, market watches for changes
            last_meeting_effect = self._estimate_last_meeting_effect(now)

            final_sentiment = proximity_sentiment + last_meeting_effect
            final_sentiment = max(-0.5, min(0.5, final_sentiment))  # Cap at ±0.5

            result = {
                'next_meeting_date': next_meeting.strftime('%Y-%m-%d'),
                'days_to_meeting': days_to_meeting,
                'weeks_to_meeting': float(weeks_to_meeting),
                'proximity_phase': self._classify_proximity_phase(days_to_meeting),
                'base_proximity_sentiment': proximity_sentiment,
                'last_meeting_effect': last_meeting_effect,
                'sentiment': float(final_sentiment),
                'confidence': 0.70,  # Moderate confidence for estimated dates
                'interpretation': f"Next RBI MPC meeting in {days_to_meeting} days ({next_meeting.strftime('%b %d')})"
            }

            self._cache_data(cache_key, result)
            return result

        except Exception as e:
            print(f"Policy meeting analysis error: {e}")
            return {
                'next_meeting_date': 'unknown',
                'days_to_meeting': 30,
                'sentiment': 0.0,
                'confidence': 0.4,
                'interpretation': 'Policy meeting timing analysis failed',
                'error': str(e)
            }

    def _classify_proximity_phase(self, days_to_meeting: int) -> str:
        """Classify the phase relative to policy meeting"""
        if days_to_meeting <= 3:
            return "pre_meeting_uncertainty"
        elif days_to_meeting <= 7:
            return "meeting_week"
        elif days_to_meeting <= 14:
            return "pre_meeting_watch"
        elif days_to_meeting <= 30:
            return "policy_digestion"
        elif days_to_meeting <= 45:
            return "policy_clarity"
        else:
            return "inter_meeting_period"

    def _estimate_last_meeting_effect(self, current_date: datetime) -> float:
        """Estimate effect of last policy meeting"""
        # This is a simplified estimation
        # In practice, you'd track actual policy changes

        # Assume last meeting was in August 2025 (status quo)
        # If repo rate has been stable, market expects continued stability
        months_since_last_change = 6  # Approximate

        if months_since_last_change >= 6:
            return 0.05  # Stability expectation = slight positive
        elif months_since_last_change >= 3:
            return 0.0  # Neutral
        else:
            return -0.05  # Recent change = more uncertainty

    async def fetch_credit_growth_enhanced(self) -> Dict:
        """Enhanced credit growth analysis using multiple banking metrics"""
        cache_key = self._get_cache_key("credit_growth_enhanced")
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached

        try:
            print("🏦 Analyzing enhanced credit growth indicators...")

            # Use Bank Nifty and major banks for credit growth proxy
            bank_tickers = ['SBIN.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', '^NSEBANK']
            nifty_ticker = '^NSEI'

            bank_performances = []

            # Fetch banking sector data
            for ticker in bank_tickers:
                try:
                    bank = yf.Ticker(ticker)
                    hist = bank.history(period="3mo")  # 3 months for better trend

                    if not hist.empty and len(hist) >= 20:
                        current_price = hist['Close'].iloc[-1]
                        price_1m_ago = hist['Close'].iloc[-20] if len(hist) >= 20 else hist['Close'].iloc[0]
                        price_3m_ago = hist['Close'].iloc[0]

                        monthly_return = (current_price - price_1m_ago) / price_1m_ago if price_1m_ago > 0 else 0
                        quarterly_return = (current_price - price_3m_ago) / price_3m_ago if price_3m_ago > 0 else 0

                        bank_performances.append({
                            'ticker': ticker,
                            'monthly_return': monthly_return,
                            'quarterly_return': quarterly_return
                        })

                except Exception as e:
                    print(f"Failed to fetch {ticker}: {e}")
                    continue

            # Fetch Nifty for relative comparison
            try:
                nifty = yf.Ticker(nifty_ticker)
                nifty_hist = nifty.history(period="3mo")

                if not nifty_hist.empty and len(nifty_hist) >= 20:
                    nifty_current = nifty_hist['Close'].iloc[-1]
                    nifty_1m_ago = nifty_hist['Close'].iloc[-20] if len(nifty_hist) >= 20 else nifty_hist['Close'].iloc[
                        0]
                    nifty_3m_ago = nifty_hist['Close'].iloc[0]

                    nifty_monthly = (nifty_current - nifty_1m_ago) / nifty_1m_ago if nifty_1m_ago > 0 else 0
                    nifty_quarterly = (nifty_current - nifty_3m_ago) / nifty_3m_ago if nifty_3m_ago > 0 else 0
                else:
                    nifty_monthly = 0
                    nifty_quarterly = 0
            except:
                nifty_monthly = 0
                nifty_quarterly = 0

            if len(bank_performances) >= 2:
                # Calculate average banking performance
                avg_monthly_return = np.mean([b['monthly_return'] for b in bank_performances])
                avg_quarterly_return = np.mean([b['quarterly_return'] for b in bank_performances])

                # Banking sector relative performance vs Nifty
                relative_monthly = avg_monthly_return - nifty_monthly
                relative_quarterly = avg_quarterly_return - nifty_quarterly

                # Credit growth sentiment
                # Banking sector outperformance suggests healthy credit growth
                credit_growth_score = relative_monthly * 0.6 + relative_quarterly * 0.4
                credit_sentiment = np.tanh(credit_growth_score * 5)  # Scale to sentiment

                result = {
                    'banking_sector_monthly_return': float(avg_monthly_return * 100),
                    'banking_sector_quarterly_return': float(avg_quarterly_return * 100),
                    'nifty_monthly_return': float(nifty_monthly * 100),
                    'nifty_quarterly_return': float(nifty_quarterly * 100),
                    'relative_monthly_performance': float(relative_monthly * 100),
                    'relative_quarterly_performance': float(relative_quarterly * 100),
                    'credit_growth_score': float(credit_growth_score),
                    'sentiment': float(credit_sentiment),
                    'confidence': 0.72,
                    'banks_analyzed': len(bank_performances),
                    'interpretation': f"Banking sector {'outperforming' if relative_monthly > 0.01 else 'underperforming' if relative_monthly < -0.01 else 'matching'} Nifty - {'positive' if credit_sentiment > 0.1 else 'negative' if credit_sentiment < -0.1 else 'neutral'} credit growth proxy"
                }

                self._cache_data(cache_key, result)
                return result

            return {
                'sentiment': 0.0,
                'confidence': 0.4,
                'interpretation': 'Insufficient banking data for credit growth analysis',
                'banks_analyzed': len(bank_performances)
            }

        except Exception as e:
            print(f"Enhanced credit growth analysis error: {e}")
            return {
                'sentiment': 0.0,
                'confidence': 0.35,
                'interpretation': 'Credit growth analysis failed',
                'error': str(e)
            }

    async def analyze_rbi_policy_sentiment(self) -> Dict:
        """Main analysis function for RBI policy sentiment - IMPROVED"""
        print("🏛️ Analyzing RBI Interest Rates & Policy Impact...")
        print("=" * 50)

        start_time = datetime.now()

        try:
            # Fetch all RBI policy components with improved methods
            tasks = [
                self.fetch_current_repo_rate_improved(),
                self.fetch_banking_sector_yield_proxy(),  # FIXED method
                self.fetch_policy_meeting_proximity_improved(),
                self.fetch_credit_growth_enhanced()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results with better error handling
            component_results = {
                'repo_rate': self._safe_result(results[0], 'repo_rate'),
                'banking_yield_proxy': self._safe_result(results[1], 'banking_yield_proxy'),
                'policy_proximity': self._safe_result(results[2], 'policy_proximity'),
                'credit_growth': self._safe_result(results[3], 'credit_growth')
            }

            # Calculate composite RBI policy sentiment
            weights = self.config['component_weights']
            total_weighted_sentiment = 0
            total_weight = 0
            confidence_scores = []

            component_map = {
                'repo_rate_trend': 'repo_rate',
                'banking_sector_proxy': 'banking_yield_proxy',
                'policy_proximity': 'policy_proximity',
                'credit_growth': 'credit_growth'
            }

            contribution_breakdown = {}

            for weight_key, result_key in component_map.items():
                if result_key in component_results:
                    comp_data = component_results[result_key]
                    weight = weights.get(weight_key, 0.25)
                    sentiment = comp_data.get('sentiment', 0)
                    confidence = comp_data.get('confidence', 0.5)

                    adjusted_weight = weight * confidence
                    contribution = sentiment * adjusted_weight

                    total_weighted_sentiment += contribution
                    total_weight += adjusted_weight
                    confidence_scores.append(confidence)
                    contribution_breakdown[result_key] = {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'weight': weight,
                        'contribution': contribution
                    }

            composite_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5

            execution_time = (datetime.now() - start_time).total_seconds()

            # Build comprehensive result structure
            result = {
                'component_sentiment': float(composite_sentiment),
                'component_confidence': float(avg_confidence),
                'component_weight': 0.10,  # 10% weight in overall system
                'weighted_contribution': float(composite_sentiment * 0.10),

                'sub_indicators': {
                    'repo_rate_analysis': component_results['repo_rate'],
                    'banking_yield_proxy': component_results['banking_yield_proxy'],
                    'policy_meeting_proximity': component_results['policy_proximity'],
                    'credit_growth_enhanced': component_results['credit_growth']
                },

                'contribution_breakdown': contribution_breakdown,

                'market_context': {
                    'policy_stance': self._determine_policy_stance(composite_sentiment, component_results),
                    'components_analyzed': len([c for c in component_results.values() if not c.get('error')]),
                    'successful_components': len(
                        [c for c in component_results.values() if c.get('confidence', 0) > 0.5]),
                    'data_sources': ['RBI_Policy', 'Banking_Sector', 'Bond_Proxy', 'Credit_Markets'],
                    'update_frequency': '4_hours',
                    'improvements_applied': ['fixed_bond_tickers', 'banking_sector_proxy', 'enhanced_credit_analysis']
                },

                'execution_time_seconds': execution_time,
                'kpi_name': 'RBI_Interest_Rates_Policy',
                'framework_weight': 0.10,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': 'high' if avg_confidence > 0.7 else 'moderate' if avg_confidence > 0.5 else 'basic',
                'version': '1.1_improved'
            }

            # Generate interpretation
            interpretation = self._generate_comprehensive_interpretation(composite_sentiment, avg_confidence,
                                                                         component_results)
            result['interpretation'] = interpretation

            print(f"RBI Policy Analysis completed in {execution_time:.1f}s")
            print(f"Sentiment: {composite_sentiment:.3f} ({interpretation})")
            print(f"Confidence: {avg_confidence:.1%}")

            return result

        except Exception as e:
            print(f"RBI policy analysis failed: {e}")
            return self._create_error_result(f"Analysis exception: {e}")

    def _safe_result(self, result: any, component_name: str) -> Dict:
        """Safely extract result or create fallback"""
        if isinstance(result, Exception):
            return {
                'sentiment': 0.0,
                'confidence': 0.3,
                'error': str(result),
                'component': component_name,
                'fallback_applied': True
            }
        elif isinstance(result, dict) and 'sentiment' in result:
            return result
        else:
            return {
                'sentiment': 0.0,
                'confidence': 0.3,
                'error': 'Invalid result format',
                'component': component_name,
                'fallback_applied': True
            }

    def _determine_policy_stance(self, composite_sentiment: float, component_results: Dict) -> str:
        """Determine overall policy stance"""
        repo_data = component_results.get('repo_rate', {})
        repo_rate = repo_data.get('current_repo_rate', 6.5)

        if composite_sentiment > 0.3:
            return 'accommodative'
        elif composite_sentiment < -0.3:
            return 'restrictive'
        elif repo_rate < 6.0:
            return 'accommodative'
        elif repo_rate > 7.0:
            return 'restrictive'
        else:
            return 'neutral'

    def _generate_comprehensive_interpretation(self, sentiment: float, confidence: float, components: Dict) -> str:
        """Generate comprehensive interpretation"""
        confidence_level = "HIGH_CONFIDENCE" if confidence > 0.75 else "MODERATE_CONFIDENCE" if confidence > 0.6 else "LOW_CONFIDENCE"

        if sentiment > 0.3:
            base_interpretation = "ACCOMMODATIVE"
        elif sentiment > 0.1:
            base_interpretation = "MILDLY_POSITIVE"
        elif sentiment > -0.1:
            base_interpretation = "NEUTRAL"
        elif sentiment > -0.3:
            base_interpretation = "MILDLY_RESTRICTIVE"
        else:
            base_interpretation = "RESTRICTIVE"

        return f"🏛️ {base_interpretation} - RBI policy {confidence_level}"

    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result with neutral default"""
        return {
            'component_sentiment': 0.0,
            'component_confidence': 0.4,
            'component_weight': 0.10 * 0.5,  # Reduced weight for error
            'sub_indicators': {
                'error': {
                    'message': error_message,
                    'fallback_applied': True,
                    'version': '1.1_error_fallback'
                }
            },
            'market_context': {
                'data_source': 'Error_Fallback',
                'error': True,
                'policy_stance': 'unknown'
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'version': '1.1_improved'
        }


# Test function
async def test_rbi_analyzer_improved():
    """Test the improved RBI policy analyzer"""
    analyzer = RBIInterestRatesAnalyzer()
    result = await analyzer.analyze_rbi_policy_sentiment()

    print("\n" + "=" * 70)
    print("RBI POLICY ANALYZER v1.1 - IMPROVED TEST RESULTS")
    print("=" * 70)

    if result:
        print(f"Component Sentiment: {result.get('component_sentiment', 0):.3f}")
        print(f"Component Confidence: {result.get('component_confidence', 0):.1%}")
        print(f"Component Weight: {result.get('component_weight', 0):.0%}")
        print(f"Interpretation: {result.get('interpretation', 'N/A')}")
        print(f"Policy Stance: {result.get('market_context', {}).get('policy_stance', 'N/A')}")

        # Show improvements
        improvements = result.get('market_context', {}).get('improvements_applied', [])
        if improvements:
            print(f"\n🔧 Improvements Applied:")
            for improvement in improvements:
                print(f"{improvement.replace('_', ' ').title()}")

        # Show component breakdown
        breakdown = result.get('contribution_breakdown', {})
        if breakdown:
            print(f"\nComponent Contributions:")
            for comp_name, comp_data in breakdown.items():
                comp_display = comp_name.replace('_', ' ').title()
                sentiment = comp_data.get('sentiment', 0)
                confidence = comp_data.get('confidence', 0)
                contribution = comp_data.get('contribution', 0)
                print(f"   {comp_display}: {sentiment:+.3f} ({confidence:.0%}) → {contribution:+.4f}")

        # Save detailed JSON
        with open('rbi_policy_analysis_improved.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✅ Detailed results saved to: rbi_policy_analysis_improved.json")

    return result


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_rbi_analyzer_improved())
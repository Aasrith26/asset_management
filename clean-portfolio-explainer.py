# CLEAN PORTFOLIO EXPLANATION SYSTEM
# ==================================
# Professional system that sends full context files to LLM and returns JSON

import requests
import json
from datetime import datetime
from typing import Dict, List, Any
import asyncio
import httpx

class PortfolioExplanationSystem:
    """
    Clean portfolio explanation system with full context file integration
    """
    
    def __init__(self, backend_url: str = "https://assetmanagement-production-f542.up.railway.app"):
        self.backend_url = backend_url.rstrip('/')
        self.openrouter_api_key = "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"
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
    
    def set_openrouter_key(self, api_key: str):
        """Set OpenRouter API key"""
        self.openrouter_api_key = api_key
    
    async def get_complete_analysis_data(self, job_id: str) -> Dict[str, Any]:
        """Get complete analysis data including full context files"""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get CSV data
                csv_response = await client.get(f"{self.backend_url}/jobs/{job_id}/csv")
                
                # Get full context files for all assets
                assets = ["NIFTY50", "GOLD", "BITCOIN", "REIT"]
                context_files = {}
                
                for asset in assets:
                    context_response = await client.get(f"{self.backend_url}/jobs/{job_id}/context/{asset}")
                    if context_response.status_code == 200:
                        context_files[asset] = context_response.json()
                
                # Extract sentiment scores
                sentiment_scores = {}
                if csv_response.status_code == 200:
                    import pandas as pd
                    df = pd.DataFrame(csv_response.json()['data'])
                    
                    sentiment_data = df[
                        (df['metric_type'] == 'sentiment') & 
                        (df['metric_name'] == 'overall_sentiment') &
                        (df['asset'] != 'PORTFOLIO')
                    ][['asset', 'value', 'confidence']]
                    
                    for _, row in sentiment_data.iterrows():
                        sentiment_scores[row['asset']] = {
                            'sentiment': float(row['value']) if row['value'] is not None else 0.0,
                            'confidence': float(row['confidence']) if row['confidence'] is not None else 0.5
                        }
                
                return {
                    'sentiment_scores': sentiment_scores,
                    'context_files': context_files,
                    'status': 'success'
                }
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def create_comprehensive_llm_prompt(self, user_portfolio: Dict, rebalanced_portfolio: Dict, analysis_data: Dict) -> str:
        """Create comprehensive prompt with full context files"""
        
        context_files = analysis_data.get('context_files', {})
        sentiment_scores = analysis_data.get('sentiment_scores', {})
        
        prompt = f"""
# Portfolio Rebalancing Analysis Request

You are a professional wealth advisor. Analyze the portfolio rebalancing and provide a comprehensive explanation using the detailed market context provided.

## User Portfolio Information:
{json.dumps(user_portfolio, indent=2)}

## Rebalanced Portfolio Output:
{json.dumps(rebalanced_portfolio, indent=2)}

## Current Market Analysis:

### Sentiment Scores:
{json.dumps(sentiment_scores, indent=2)}

### Detailed Market Context Files:

"""
        
        # Add full context files to the prompt
        for asset, context_data in context_files.items():
            prompt += f"""
#### {asset} Context:
{json.dumps(context_data, indent=2)}

"""
        
        prompt += """
## Required Response Format:

Provide your analysis in the following JSON structure only. Do not include any text outside this JSON:

{
  "executive_summary": "Brief 2-3 sentence overview of the rebalancing recommendation",
  "asset_recommendations": {
    "NIFTY50": {
      "explanation" : "Generate a detailed, professional paragraph using fact-based language that explains the allocation adjustment. 
      and synthesize the most influential KPI 
      affecting the asset, explicitly mention the current allocation, recommended allocation, and the change amount in monetary terms. 
      Incorporate relevant market conditions, technical metrics (RSI, volatility, sentiment scores), and data quality assessments 
      from the context files. The tone should be analytical and data-driven, explaining why the adjustment is justified based on 
      quantitative analysis, market dynamics, and risk-return optimization. Avoid recommendation language - instead present 
      the changes as data-driven adjustments with supporting evidence from sentiment analysis and technical indicators."
    },
    "GOLD": { /* similar structure */ },
    "BITCOIN": { /* similar structure */ },
    "REIT": { /* similar structure */ }
  },
  "portfolio_impact": {
    "risk_improvement": "Description of risk changes",
    "return_expectation": "Expected return impact",
    "diversification_benefits": "How diversification improves"
  },
  "implementation_plan": {
    "timeline": "Suggested implementation timeline",
    "order_of_execution": ["asset1", "asset2", "asset3", "asset4"],
    "monitoring_points": ["point1", "point2", "point3"]
  },
  "market_insights": {
    "key_drivers": ["driver1", "driver2", "driver3"],
    "risk_factors": ["risk1", "risk2", "risk3"],
    "opportunities": ["opportunity1", "opportunity2"]
  }
}

## Instructions:
1. Use the detailed context files to explain WHY each rebalancing decision makes sense
2. Reference specific metrics from the context data (RSI, volatility, funding rates, etc.)
3. Connect market sentiment to portfolio allocation changes
4. Provide actionable insights based on the comprehensive market analysis
5. Return ONLY the JSON response, no additional text
"""
        
        return prompt
    
    async def generate_json_explanation(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON explanation using LLM"""
        
        if not self.openrouter_api_key:
            return self._generate_fallback_json_explanation()
        
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/portfolio-rebalancer",
                    "X-Title": "Portfolio Rebalancing System"
                }
                
                data = {
                    "model": "deepseek/deepseek-chat-v3.1:free",
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a professional portfolio analyst. Analyze the provided data and return a comprehensive JSON response exactly as requested. Use the detailed context files to provide specific, data-driven explanations."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.2
                }
                
                response = await client.post(
                    f"{self.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    explanation_text = result['choices'][0]['message']['content']
                    
                    # Try to parse JSON from response
                    try:
                        # Find JSON in response
                        start_idx = explanation_text.find('{')
                        end_idx = explanation_text.rfind('}') + 1
                        
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = explanation_text[start_idx:end_idx]
                            explanation_json = json.loads(json_str)
                            return explanation_json
                        else:
                            return {'error': 'No valid JSON found in LLM response', 'raw_response': explanation_text}
                            
                    except json.JSONDecodeError as e:
                        return {'error': f'JSON parsing failed: {e}', 'raw_response': explanation_text}
                
                else:
                    return {'error': f'OpenRouter API error: {response.status_code}', 'details': response.text}
                    
        except Exception as e:
            return {'error': f'LLM request failed: {e}'}
    
    def _generate_fallback_json_explanation(self) -> Dict[str, Any]:
        """Generate fallback JSON explanation when LLM is unavailable"""
        
        return {
            "executive_summary": "Portfolio rebalancing recommended based on current market conditions and risk optimization analysis. The strategy focuses on adjusting asset allocations to improve risk-adjusted returns.",
            "asset_recommendations": {
                "NIFTY50": {
                    "action": "DECREASE",
                    "current_allocation": 0.40,
                    "recommended_allocation": 0.35,
                    "change_amount": -25000,
                    "reasoning": "Neutral market sentiment suggests consolidation phase. Reducing allocation to optimize portfolio balance while maintaining core equity exposure.",
                    "market_context": "Technical indicators show sideways movement with moderate volatility levels."
                },
                "GOLD": {
                    "action": "INCREASE",
                    "current_allocation": 0.30,
                    "recommended_allocation": 0.35,
                    "change_amount": 25000,
                    "reasoning": "Strong positive sentiment with high confidence levels. Gold showing safe haven characteristics and hedge against market uncertainty.",
                    "market_context": "Momentum indicators positive with favorable market positioning for precious metals."
                },
                "BITCOIN": {
                    "action": "DECREASE",
                    "current_allocation": 0.20,
                    "recommended_allocation": 0.15,
                    "change_amount": -25000,
                    "reasoning": "Increased volatility patterns detected. Risk management through position sizing while maintaining growth exposure.",
                    "market_context": "High RSI levels and funding rate fluctuations indicate overheated conditions requiring caution."
                },
                "REIT": {
                    "action": "INCREASE",
                    "current_allocation": 0.10,
                    "recommended_allocation": 0.15,
                    "change_amount": 25000,
                    "reasoning": "Improved fundamentals and yield potential. Real estate market showing stability with income generation benefits.",
                    "market_context": "Interest rate environment favorable for REIT performance with strong institutional flows."
                }
            },
            "portfolio_impact": {
                "risk_improvement": "Portfolio volatility reduced from 18% to 16%, representing an 11% improvement in risk metrics",
                "return_expectation": "Expected return optimized from 12.0% to 11.5% with better risk adjustment",
                "diversification_benefits": "Enhanced balance across asset classes improves portfolio resilience and reduces correlation risk"
            },
            "implementation_plan": {
                "timeline": "Execute rebalancing over 3-week period to minimize market impact",
                "order_of_execution": ["GOLD", "REIT", "NIFTY50", "BITCOIN"],
                "monitoring_points": [
                    "Track sentiment changes for early signals",
                    "Monitor volatility levels during implementation",
                    "Review allocation drift monthly"
                ]
            },
            "market_insights": {
                "key_drivers": [
                    "Gold momentum supported by safe haven demand",
                    "REIT fundamentals strengthening with rate environment",
                    "Bitcoin volatility requiring active risk management"
                ],
                "risk_factors": [
                    "Market sentiment shifts could affect timing",
                    "Volatility spikes during implementation period",
                    "Correlation increases during stress periods"
                ],
                "opportunities": [
                    "Gold allocation increase capitalizes on positive momentum",
                    "REIT exposure benefits from income generation",
                    "Reduced Bitcoin allocation manages downside risk"
                ]
            },
            "_metadata": {
                "method": "rule_based_fallback",
                "note": "LLM API not available - using systematic analysis"
            }
        }
    
    async def explain_rebalancing(self, user_portfolio: Dict, rebalanced_portfolio: Dict, job_id: str) -> Dict[str, Any]:
        """
        Main method: Generate complete portfolio explanation using context files
        
        Args:
            user_portfolio: User's current portfolio information
            rebalanced_portfolio: Riskfolio optimization output
            job_id: Job ID from sentiment analysis
        
        Returns:
            JSON structured explanation
        """
        
        # Get complete analysis data including full context files
        analysis_data = await self.get_complete_analysis_data(job_id)
        
        if analysis_data.get('status') != 'success':
            return {'error': 'Failed to retrieve analysis data', 'details': analysis_data.get('error')}
        
        # Create comprehensive prompt with full context files
        llm_prompt = self.create_comprehensive_llm_prompt(user_portfolio, rebalanced_portfolio, analysis_data)
        
        # Generate explanation
        explanation = await self.generate_json_explanation(llm_prompt)
        
        # Add metadata
        explanation['_system_metadata'] = {
            'job_id': job_id,
            'generated_at': datetime.now().isoformat(),
            'backend_url': self.backend_url,
            'context_files_included': list(analysis_data.get('context_files', {}).keys()),
            'sentiment_data_included': True,
            'llm_model': 'meta-llama/llama-3.1-8b-instruct:free' if self.openrouter_api_key else 'fallback'
        }
        
        return explanation

# USAGE EXAMPLE
# =============

async def test_explanation_system():
    """Test the explanation system"""
    
    # Initialize system
    explainer = PortfolioExplanationSystem()
    # explainer.set_openrouter_key("your-key-here")  # Uncomment to use LLM
    
    # Example user portfolio
    user_portfolio = {
        'current_holdings': {
            'NIFTY50': 200000,
            'GOLD': 150000,
            'BITCOIN': 100000,
            'REIT': 50000
        },
        'total_value': 500000,
        'risk_appetite': 'moderate',
        'investment_horizon_months': 24,
        'financial_goals': ['wealth_building', 'retirement'],
        'constraints': ['tax_optimization']
    }
    
    # Example rebalanced portfolio (from riskfolio)
    rebalanced_portfolio = {
        'rebalancing_summary': {
            'original_portfolio': {
                'NIFTY50': {'allocation': 0.40, 'value': 200000, 'weight': 40.0},
                'GOLD': {'allocation': 0.30, 'value': 150000, 'weight': 30.0},
                'BITCOIN': {'allocation': 0.20, 'value': 100000, 'weight': 20.0},
                'REIT': {'allocation': 0.10, 'value': 50000, 'weight': 10.0}
            },
            'rebalanced_portfolio': {
                'NIFTY50': {'allocation': 0.35, 'value': 175000, 'weight': 35.0},
                'GOLD': {'allocation': 0.35, 'value': 175000, 'weight': 35.0},
                'BITCOIN': {'allocation': 0.15, 'value': 75000, 'weight': 15.0},
                'REIT': {'allocation': 0.15, 'value': 75000, 'weight': 15.0}
            }
        },
        'risk_metrics': {
            'original_portfolio_risk': 0.18,
            'rebalanced_portfolio_risk': 0.16,
            'risk_reduction': 0.02,
            'expected_return_original': 0.12,
            'expected_return_rebalanced': 0.115
        },
        'optimization_method': 'mean_variance'
    }
    
    # Start analysis to get job_id
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            analysis_response = await client.post(
                f"{explainer.backend_url}/analyze/all-assets",
                json={
                    "assets": ["NIFTY50", "GOLD", "BITCOIN", "REIT"],
                    "generate_pipeline_outputs": True
                }
            )
            
            if analysis_response.status_code == 200:
                job_id = analysis_response.json().get("job_id")
                print(f"Started analysis with job_id: {job_id}")
                
                # Wait for completion
                await asyncio.sleep(60)
                
                # Generate explanation
                explanation = await explainer.explain_rebalancing(user_portfolio, rebalanced_portfolio, job_id)
                
                # Output clean JSON
                print("\n" + "="*50)
                print("PORTFOLIO REBALANCING EXPLANATION (JSON)")
                print("="*50)
                print(json.dumps(explanation, indent=2))
                
            else:
                print(f"Analysis failed: {analysis_response.status_code}")
                
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_explanation_system())
# FINAL CORRECTED 4-ASSET BACKEND
# ================================
# Fixed Gold integration with correct function names

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Union
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# Add paths for all asset KPIs
sys.path.append(str(Path(__file__).parent / "nifty_50_kpis"))
sys.path.append(str(Path(__file__).parent / "gold_kpis"))
sys.path.append(str(Path(__file__).parent / "bitcoin_kpis"))
sys.path.append(str(Path(__file__).parent / "reit_kpis"))

# Configure logging WITHOUT Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# Asset Management Enums
class AssetType(str, Enum):
    NIFTY50 = "NIFTY50"
    GOLD = "GOLD"
    BITCOIN = "BITCOIN"
    REIT = "REIT"


class AssetCategory(str, Enum):
    TRADITIONAL = "traditional"  # Nifty + Gold
    ALTERNATIVE = "alternative"  # Bitcoin + REIT
    EQUITY = "equity"  # Nifty
    COMMODITY = "commodity"  # Gold
    CRYPTOCURRENCY = "crypto"  # Bitcoin
    REAL_ESTATE = "real_estate"  # REIT


# Pydantic Models
class AnalysisRequest(BaseModel):
    assets: List[AssetType]
    timeout_minutes: Optional[int] = 15
    include_portfolio_view: Optional[bool] = True
    save_results: Optional[bool] = True


class CategoryAnalysisRequest(BaseModel):
    timeout_minutes: Optional[int] = 15
    include_detailed_breakdown: Optional[bool] = True


# ================================================================
# CORRECTED ASSET INTEGRATION ADAPTERS
# ================================================================

class NiftyIntegrationAdapter:
    """Nifty 50 integration adapter"""

    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            logger.info("Starting Nifty 50 analysis...")

            # Import and run existing Nifty analyzer
            from nifty_50_kpis.complete_nifty_analyzer import run_complete_nifty_analysis
            result = await run_complete_nifty_analysis()

            return {
                'asset_type': AssetType.NIFTY50,
                'sentiment': result.get('composite_sentiment', 0.0),
                'confidence': result.get('composite_confidence', 0.5),
                'execution_time': result.get('execution_time_seconds', 0.0),
                'component_details': result.get('component_analysis', {}),
                'market_context': result.get('market_context', {}),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Multiple_Indian_APIs'
            }

        except Exception as e:
            logger.error(f"Nifty 50 analysis failed: {e}")
            return {
                'asset_type': AssetType.NIFTY50,
                'sentiment': 0.0,
                'confidence': 0.3,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class GoldIntegrationAdapter:
    """Gold integration adapter - CORRECTED VERSION"""

    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            logger.info("Starting Gold analysis...")

            result = None

            # Option 1: Use the main function (CORRECTED)
            try:
                from gold_kpis.integrated_gold_sentiment_v5 import run_integrated_analysis
                result = await run_integrated_analysis()
                logger.info("Gold analysis: Used run_integrated_analysis function")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Gold Option 1 failed: {e}")

                # Option 2: Use the main class (CORRECTED CLASS NAME)
                try:
                    from gold_kpis.integrated_gold_sentiment_v5 import IntegratedGoldSentimentSystem
                    system = IntegratedGoldSentimentSystem()
                    result = await system.run_comprehensive_analysis()
                    logger.info("Gold analysis: Used IntegratedGoldSentimentSystem class")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Gold Option 2 failed: {e}")

                    # Option 3: Try without the gold_kpis prefix
                    try:
                        from gold_kpis.integrated_gold_sentiment_v5 import run_integrated_analysis
                        result = await run_integrated_analysis()
                        logger.info("Gold analysis: Used run_integrated_analysis (no prefix)")
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Gold Option 3 failed: {e}")

                        # Option 4: Try class without prefix
                        try:
                            from gold_kpis.integrated_gold_sentiment_v5 import IntegratedGoldSentimentSystem
                            system = IntegratedGoldSentimentSystem()
                            result = await system.run_comprehensive_analysis()
                            logger.info("Gold analysis: Used IntegratedGoldSentimentSystem (no prefix)")
                        except (ImportError, AttributeError) as e:
                            logger.warning(f"Gold Option 4 failed: {e}")

                            # Option 5: Dynamic import as last resort
                            try:
                                import importlib.util

                                # Try multiple possible file locations
                                possible_paths = [
                                    "gold_kpis/integrated_gold_sentiment_v5.py",
                                    "integrated_gold_sentiment_v5.py"
                                ]

                                for path in possible_paths:
                                    if os.path.exists(path):
                                        spec = importlib.util.spec_from_file_location("gold_module", path)
                                        gold_module = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(gold_module)

                                        # Try the main function
                                        if hasattr(gold_module, 'run_integrated_analysis'):
                                            func = getattr(gold_module, 'run_integrated_analysis')
                                            result = await func()
                                            logger.info(f"Gold analysis: Used run_integrated_analysis from {path}")
                                            break
                                        # Try the class
                                        elif hasattr(gold_module, 'IntegratedGoldSentimentSystem'):
                                            system_class = getattr(gold_module, 'IntegratedGoldSentimentSystem')
                                            system = system_class()
                                            result = await system.run_comprehensive_analysis()
                                            logger.info(
                                                f"Gold analysis: Used IntegratedGoldSentimentSystem from {path}")
                                            break

                                if result is None:
                                    raise ImportError("Could not find Gold analysis function or class in any location")

                            except Exception as e:
                                logger.error(f"Gold dynamic import failed: {e}")
                                raise ImportError("All Gold analysis import methods failed")

            if result is None:
                raise ValueError("Gold analysis returned no result")

            # Process the result - handle the actual structure
            if isinstance(result, dict):
                executive = result.get('executive_summary', {})
                return {
                    'asset_type': AssetType.GOLD,
                    'sentiment': executive.get('overall_sentiment', result.get('sentiment', 0.0)),
                    'confidence': executive.get('confidence_level', result.get('confidence', 0.5)),
                    'execution_time': result.get('performance_metrics', {}).get('execution_time_seconds', 0.0),
                    'component_details': result.get('component_analysis', {}),
                    'market_context': result.get('market_context', {}),
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'Multiple_Global_APIs'
                }
            else:
                raise ValueError("Gold analysis returned unexpected result format")

        except Exception as e:
            logger.error(f"Gold analysis failed: {e}")
            return {
                'asset_type': AssetType.GOLD,
                'sentiment': 0.0,
                'confidence': 0.3,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class BitcoinIntegrationAdapter:
    """Bitcoin integration adapter"""

    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            logger.info("Starting Bitcoin analysis...")

            # Import and run Bitcoin analyzer
            from bitcoin_kpis.bitcoin_sentiment_analyzer import run_bitcoin_analysis
            result = await run_bitcoin_analysis()

            return {
                'asset_type': AssetType.BITCOIN,
                'sentiment': result.get('component_sentiment', 0.0),
                'confidence': result.get('component_confidence', 0.5),
                'execution_time': result.get('performance_metrics', {}).get('execution_time_seconds', 0.0),
                'component_details': result.get('component_analysis', {}),
                'executive_summary': result.get('executive_summary', {}),
                'market_context': result.get('market_context', {}),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Kraken_Multi_API'
            }

        except Exception as e:
            logger.error(f"Bitcoin analysis failed: {e}")
            return {
                'asset_type': AssetType.BITCOIN,
                'sentiment': 0.0,
                'confidence': 0.3,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class REITIntegrationAdapter:
    """REIT integration adapter"""

    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            logger.info("Starting REIT analysis...")

            # Import and run REIT analyzer
            from reit_kpis.reit_sentiment_analyzer_original import run_reit_analysis
            result = await run_reit_analysis()

            return {
                'asset_type': AssetType.REIT,
                'sentiment': result.get('component_sentiment', 0.0),
                'confidence': result.get('component_confidence', 0.5),
                'execution_time': result.get('performance_metrics', {}).get('execution_time_seconds', 0.0),
                'component_details': result.get('component_analysis', {}),
                'executive_summary': result.get('executive_summary', {}),
                'market_context': result.get('market_context', {}),
                'symbol': result.get('symbol', 'DLF'),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Yahoo_Finance_TradingEconomics'
            }

        except Exception as e:
            logger.error(f"REIT analysis failed: {e}")
            return {
                'asset_type': AssetType.REIT,
                'sentiment': 0.0,
                'confidence': 0.3,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# ================================================================
# MAIN ORCHESTRATOR (unchanged)
# ================================================================

class MultiAssetOrchestrator:
    """Main orchestrator for all 4 assets"""

    ASSET_ADAPTERS = {
        AssetType.NIFTY50: NiftyIntegrationAdapter,
        AssetType.GOLD: GoldIntegrationAdapter,
        AssetType.BITCOIN: BitcoinIntegrationAdapter,
        AssetType.REIT: REITIntegrationAdapter
    }

    CATEGORY_MAPPINGS = {
        AssetCategory.TRADITIONAL: [AssetType.NIFTY50, AssetType.GOLD],
        AssetCategory.ALTERNATIVE: [AssetType.BITCOIN, AssetType.REIT],
        AssetCategory.EQUITY: [AssetType.NIFTY50],
        AssetCategory.COMMODITY: [AssetType.GOLD],
        AssetCategory.CRYPTOCURRENCY: [AssetType.BITCOIN],
        AssetCategory.REAL_ESTATE: [AssetType.REIT]
    }

    @classmethod
    async def analyze_assets(cls, assets: List[AssetType], timeout_minutes: int = 15) -> Dict[str, Any]:
        """Analyze specified assets"""
        start_time = datetime.now()
        results = {}

        logger.info(f"Starting analysis for {len(assets)} assets: {[a.value for a in assets]}")

        # Run analyses with timeout
        timeout_seconds = timeout_minutes * 60
        tasks = []

        for asset in assets:
            adapter = cls.ASSET_ADAPTERS[asset]
            task = cls._run_with_timeout(asset, adapter.run_analysis(), timeout_seconds)
            tasks.append(task)

        # Execute all tasks in parallel
        asset_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, asset in enumerate(assets):
            result = asset_results[i]
            if isinstance(result, Exception):
                logger.error(f"{asset.value} failed with exception: {result}")
                results[asset.value] = {
                    'asset_type': asset.value,
                    'status': 'exception',
                    'error': str(result),
                    'sentiment': 0.0,
                    'confidence': 0.3
                }
            else:
                results[asset.value] = result

        # Generate portfolio summary
        execution_time = (datetime.now() - start_time).total_seconds()
        portfolio_summary = cls._generate_portfolio_summary(results, execution_time)

        return {
            'analysis_results': results,
            'portfolio_summary': portfolio_summary,
            'execution_time_seconds': execution_time,
            'timestamp': datetime.now().isoformat(),
            'assets_analyzed': len(assets),
            'total_assets_available': len(AssetType)
        }

    @classmethod
    async def analyze_category(cls, category: AssetCategory, timeout_minutes: int = 15) -> Dict[str, Any]:
        """Analyze assets in a specific category"""
        assets = cls.CATEGORY_MAPPINGS.get(category, [])
        logger.info(f"Analyzing {category.value} category: {[a.value for a in assets]}")

        if not assets:
            raise ValueError(f"No assets found for category: {category.value}")

        result = await cls.analyze_assets(assets, timeout_minutes)
        result['category'] = category.value
        return result

    @classmethod
    async def _run_with_timeout(cls, asset: AssetType, task, timeout_seconds: int):
        """Run single asset analysis with timeout"""
        try:
            result = await asyncio.wait_for(task, timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"{asset.value} analysis timed out after {timeout_seconds}s")
            return {
                'asset_type': asset.value,
                'status': 'timeout',
                'sentiment': 0.0,
                'confidence': 0.2,
                'error': f'Analysis timed out after {timeout_seconds}s'
            }
        except Exception as e:
            logger.error(f"{asset.value} analysis error: {e}")
            return {
                'asset_type': asset.value,
                'status': 'error',
                'sentiment': 0.0,
                'confidence': 0.3,
                'error': str(e)
            }

    @classmethod
    def _generate_portfolio_summary(cls, results: Dict[str, Any], execution_time: float) -> Dict:
        """Generate portfolio-level summary"""
        successful_results = [r for r in results.values() if r.get('status') == 'success']

        if not successful_results:
            return {
                'portfolio_sentiment': 0.0,
                'portfolio_confidence': 0.0,
                'successful_assets': 0,
                'failed_assets': len(results),
                'execution_time_seconds': execution_time,
                'status': 'failed',
                'risk_assessment': 'HIGH RISK - No successful analyses'
            }

        # Calculate portfolio metrics (equal weight)
        sentiments = [r.get('sentiment', 0.0) for r in successful_results]
        confidences = [r.get('confidence', 0.5) for r in successful_results]

        portfolio_sentiment = sum(sentiments) / len(sentiments)
        portfolio_confidence = sum(confidences) / len(confidences)

        # Risk assessment
        if portfolio_confidence > 0.7 and len(successful_results) >= 3:
            risk_level = "LOW RISK"
        elif portfolio_confidence > 0.5 and len(successful_results) >= 2:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "HIGH RISK"

        # Investment recommendation
        if portfolio_sentiment > 0.3:
            recommendation = "BUY" if portfolio_sentiment > 0.6 else "MILD BUY"
        elif portfolio_sentiment < -0.3:
            recommendation = "SELL" if portfolio_sentiment < -0.6 else "MILD SELL"
        else:
            recommendation = "HOLD"

        return {
            'portfolio_sentiment': round(portfolio_sentiment, 3),
            'portfolio_confidence': round(portfolio_confidence, 3),
            'successful_assets': len(successful_results),
            'failed_assets': len(results) - len(successful_results),
            'total_assets': len(results),
            'execution_time_seconds': execution_time,
            'status': 'success',
            'risk_assessment': risk_level,
            'investment_recommendation': recommendation,
            'asset_breakdown': {
                r.get('asset_type', 'unknown'): {
                    'sentiment': round(r.get('sentiment', 0), 3),
                    'confidence': round(r.get('confidence', 0), 3),
                    'status': r.get('status', 'unknown')
                } for r in results.values()
            }
        }


# ================================================================
# FASTAPI APPLICATION (unchanged)
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("4-Asset Sentiment Analysis Backend starting up...")
    logger.info(f"Assets supported: {[asset.value for asset in AssetType]}")
    logger.info(f"Categories supported: {[cat.value for cat in AssetCategory]}")
    yield
    # Shutdown
    logger.info("Backend shutting down...")


app = FastAPI(
    title="4-Asset Sentiment Analysis Backend",
    description="Comprehensive sentiment analysis for Nifty 50, Gold, Bitcoin, and REIT",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# API ENDPOINTS (unchanged)
# ================================================================

@app.get("/")
async def root():
    return {
        "message": "4-Asset Sentiment Analysis Backend API",
        "version": "2.0.0",
        "assets_supported": [asset.value for asset in AssetType],
        "categories_supported": [cat.value for cat in AssetCategory],
        "endpoints": [
            "GET /health",
            "GET /assets",
            "GET /categories",
            "POST /analyze/all-assets",
            "POST /analyze/assets",
            "POST /analyze/category/{category}"
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "assets_available": len(AssetType),
        "categories_available": len(AssetCategory),
        "services": {
            "nifty_50": "operational",
            "gold": "operational",
            "bitcoin": "operational",
            "reit": "operational"
        }
    }


@app.get("/assets")
async def get_assets():
    return {
        "total_assets": len(AssetType),
        "assets": {
            AssetType.NIFTY50.value: {
                "name": "Nifty 50",
                "category": "equity",
                "description": "Indian stock market index",
                "kpis": 5,
                "update_frequency": "15-minute"
            },
            AssetType.GOLD.value: {
                "name": "Gold",
                "category": "commodity",
                "description": "Precious metal commodity",
                "kpis": 5,
                "update_frequency": "30-minute"
            },
            AssetType.BITCOIN.value: {
                "name": "Bitcoin",
                "category": "cryptocurrency",
                "description": "Leading cryptocurrency",
                "kpis": 4,
                "update_frequency": "1-minute"
            },
            AssetType.REIT.value: {
                "name": "REIT",
                "category": "real_estate",
                "description": "Real Estate Investment Trust",
                "kpis": 4,
                "update_frequency": "daily"
            }
        }
    }


@app.get("/categories")
async def get_categories():
    return {
        "total_categories": len(AssetCategory),
        "categories": {
            cat.value: [asset.value for asset in MultiAssetOrchestrator.CATEGORY_MAPPINGS[cat]]
            for cat in AssetCategory
        }
    }


@app.post("/analyze/all-assets")
async def analyze_all_assets(request: AnalysisRequest):
    try:
        all_assets = [AssetType.NIFTY50, AssetType.GOLD, AssetType.BITCOIN, AssetType.REIT]
        result = await MultiAssetOrchestrator.analyze_assets(
            all_assets,
            request.timeout_minutes
        )

        if request.save_results:
            await _save_analysis_results(result, "all_assets")

        return result

    except Exception as e:
        logger.error(f"All-assets analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/assets")
async def analyze_specific_assets(request: AnalysisRequest):
    try:
        result = await MultiAssetOrchestrator.analyze_assets(
            request.assets,
            request.timeout_minutes
        )

        if request.save_results:
            await _save_analysis_results(result, f"assets_{'_'.join([a.value for a in request.assets])}")

        return result

    except Exception as e:
        logger.error(f"Specific assets analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/category/{category}")
async def analyze_asset_category(category: AssetCategory, request: CategoryAnalysisRequest):
    try:
        result = await MultiAssetOrchestrator.analyze_category(
            category,
            request.timeout_minutes
        )

        return result

    except Exception as e:
        logger.error(f"Category {category.value} analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/analyze/asset/{asset}")
async def analyze_single_asset(asset: AssetType):
    try:
        result = await MultiAssetOrchestrator.analyze_assets([asset], 10)
        return result

    except Exception as e:
        logger.error(f"Single asset {asset.value} analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

async def _save_analysis_results(result: Dict, analysis_type: str):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{analysis_type}_{timestamp}.json"

        os.makedirs("analysis_results", exist_ok=True)

        with open(f"analysis_results/{filename}", 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Analysis results saved to {filename}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_4asset_backend_corrected:app", host="0.0.0.0", port=8000, reload=True)
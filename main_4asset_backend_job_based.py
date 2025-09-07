# COMPLETE BACKEND WITH JOB-BASED PIPELINE INTEGRATION v2.0
# ==========================================================
# Full job-based pipeline system with direct API access for downstream models

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Union
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# Add paths for all asset KPIs
sys.path.append(str(Path(__file__).parent / "nifty_50_kpis"))
sys.path.append(str(Path(__file__).parent / "gold_kpis"))
sys.path.append(str(Path(__file__).parent / "bitcoin_kpis"))
sys.path.append(str(Path(__file__).parent / "reit_kpis"))

# JOB-BASED PIPELINE INTEGRATION
from job_based_pipeline_generator import generate_job_based_outputs

# Configure logging
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
    TRADITIONAL = "traditional"
    ALTERNATIVE = "alternative"
    EQUITY = "equity"
    COMMODITY = "commodity"
    CRYPTOCURRENCY = "crypto"
    REAL_ESTATE = "real_estate"

# Updated Pydantic Models with job tracking
class AnalysisRequest(BaseModel):
    assets: List[AssetType]
    timeout_minutes: Optional[int] = 15
    include_portfolio_view: Optional[bool] = True
    save_results: Optional[bool] = True
    generate_pipeline_outputs: Optional[bool] = True
    custom_job_id: Optional[str] = None  # Allow custom job ID

class CategoryAnalysisRequest(BaseModel):
    timeout_minutes: Optional[int] = 15
    include_detailed_breakdown: Optional[bool] = True
    generate_pipeline_outputs: Optional[bool] = True
    custom_job_id: Optional[str] = None

# ================================================================
# ASSET INTEGRATION ADAPTERS (same as before)
# ================================================================
app = FastAPI(
    title="4-Asset Sentiment Analysis Backend with Job-Based Pipeline v2.0",
    description="Job-tracked sentiment analysis with consolidated CSV + asset-specific JSON outputs",
    version="2.0.0",
)

# Prefer CORSMiddleware for expose_headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Job-ID", "X-Request-ID"]  # visible to browsers
)
class NiftyIntegrationAdapter:
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            logger.info("Starting Nifty 50 analysis...")
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
                'sentiment': 0.0, 'confidence': 0.3, 'status': 'error',
                'error': str(e), 'timestamp': datetime.now().isoformat()
            }

class GoldIntegrationAdapter:
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            logger.info("Starting Gold analysis...")
            result = None
            
            try:
                from gold_kpis.integrated_gold_sentiment_v5 import run_integrated_analysis
                result = await run_integrated_analysis()
                logger.info("Gold analysis: Used run_integrated_analysis function")
            except (ImportError, AttributeError) as e:
                try:
                    from gold_kpis.integrated_gold_sentiment_v5 import IntegratedGoldSentimentSystem
                    system = IntegratedGoldSentimentSystem() 
                    result = await system.run_comprehensive_analysis()
                    logger.info("Gold analysis: Used IntegratedGoldSentimentSystem class")
                except (ImportError, AttributeError) as e:
                    try:
                        from gold_kpis.integrated_gold_sentiment_v5 import run_integrated_analysis
                        result = await run_integrated_analysis()
                    except Exception:
                        raise ImportError("All Gold analysis import methods failed")
            
            if isinstance(result, dict):
                executive = result.get('executive_summary', {})
                return {
                    'asset_type': AssetType.GOLD,
                    'sentiment': executive.get('overall_sentiment', 0.0),
                    'confidence': executive.get('confidence_level', 0.5),
                    'execution_time': result.get('performance_metrics', {}).get('execution_time_seconds', 0.0),
                    'component_details': result.get('component_analysis', {}),
                    'market_context': result.get('market_context', {}),
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'Multiple_Global_APIs'
                }
        except Exception as e:
            logger.error(f"Gold analysis failed: {e}")
            return {
                'asset_type': AssetType.GOLD,
                'sentiment': 0.0, 'confidence': 0.3, 'status': 'error',
                'error': str(e), 'timestamp': datetime.now().isoformat()
            }

class BitcoinIntegrationAdapter:
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            logger.info("Starting Bitcoin analysis...")
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
                'sentiment': 0.0, 'confidence': 0.3, 'status': 'error',
                'error': str(e), 'timestamp': datetime.now().isoformat()
            }

class REITIntegrationAdapter:
    @staticmethod
    async def run_analysis() -> Dict[str, Any]:
        try:
            logger.info("Starting REIT analysis...")
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
                'sentiment': 0.0, 'confidence': 0.3, 'status': 'error',
                'error': str(e), 'timestamp': datetime.now().isoformat()
            }

# ================================================================
# MULTI-ASSET ORCHESTRATOR (enhanced with job tracking)
# ================================================================

class MultiAssetOrchestrator:
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
    async def analyze_assets_with_job_id(cls, job_id: str, assets: List[AssetType], timeout_minutes: int = 15) -> Dict[str, Any]:
        """Enhanced analysis with job ID tracking"""
        start_time = datetime.now()
        results = {}
        
        logger.info(f"Starting job {job_id} - analyzing {len(assets)} assets: {[a.value for a in assets]}")
        
        timeout_seconds = timeout_minutes * 60
        tasks = []
        
        for asset in assets:
            adapter = cls.ASSET_ADAPTERS[asset]
            task = cls._run_with_timeout(asset, adapter.run_analysis(), timeout_seconds)
            tasks.append(task)
        
        asset_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, asset in enumerate(assets):
            result = asset_results[i]
            if isinstance(result, Exception):
                logger.error(f"Job {job_id} - {asset.value} failed with exception: {result}")
                results[asset.value] = {
                    'asset_type': asset.value, 'status': 'exception',
                    'error': str(result), 'sentiment': 0.0, 'confidence': 0.3
                }
            else:
                results[asset.value] = result
        
        execution_time = (datetime.now() - start_time).total_seconds()
        portfolio_summary = cls._generate_portfolio_summary(results, execution_time)
        
        return {
            'job_id': job_id,
            'analysis_results': results,
            'portfolio_summary': portfolio_summary,
            'execution_time_seconds': execution_time,
            'timestamp': datetime.now().isoformat(),
            'assets_analyzed': len(assets),
            'total_assets_available': len(AssetType)
        }
    
    @classmethod
    async def analyze_category_with_job_id(cls, job_id: str, category: AssetCategory, timeout_minutes: int = 15) -> Dict[str, Any]:
        """Enhanced category analysis with job ID"""
        assets = cls.CATEGORY_MAPPINGS.get(category, [])
        if not assets:
            raise ValueError(f"No assets found for category: {category.value}")
        result = await cls.analyze_assets_with_job_id(job_id, assets, timeout_minutes)
        result['category'] = category.value
        return result
    
    @classmethod
    async def _run_with_timeout(cls, asset: AssetType, task, timeout_seconds: int):
        try:
            result = await asyncio.wait_for(task, timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError:
            return {
                'asset_type': asset.value, 'status': 'timeout',
                'sentiment': 0.0, 'confidence': 0.2,
                'error': f'Analysis timed out after {timeout_seconds}s'
            }
        except Exception as e:
            return {
                'asset_type': asset.value, 'status': 'error',
                'sentiment': 0.0, 'confidence': 0.3, 'error': str(e)
            }
    
    @classmethod
    def _generate_portfolio_summary(cls, results: Dict[str, Any], execution_time: float) -> Dict:
        successful_results = [r for r in results.values() if r.get('status') == 'success']
        
        if not successful_results:
            return {
                'portfolio_sentiment': 0.0, 'portfolio_confidence': 0.0,
                'successful_assets': 0, 'failed_assets': len(results),
                'execution_time_seconds': execution_time, 'status': 'failed',
                'risk_assessment': 'HIGH RISK - No successful analyses'
            }
        
        sentiments = [r.get('sentiment', 0.0) for r in successful_results]
        confidences = [r.get('confidence', 0.5) for r in successful_results]
        
        portfolio_sentiment = sum(sentiments) / len(sentiments)
        portfolio_confidence = sum(confidences) / len(confidences)
        
        if portfolio_confidence > 0.7 and len(successful_results) >= 3:
            risk_level = "LOW RISK"
        elif portfolio_confidence > 0.5 and len(successful_results) >= 2:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "HIGH RISK"
        
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
# JOB TRACKING MIDDLEWARE
# ================================================================

@app.middleware("http")
async def add_job_tracking_headers(request, call_next):
    """Add job tracking headers"""
    response = await call_next(request)
    
    # Add CORS headers for job tracking
    response.headers["Access-Control-Expose-Headers"] = "X-Job-ID, X-Request-ID"
    
    return response

# ================================================================
# FASTAPI APPLICATION WITH JOB-BASED PIPELINE
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("4-Asset Sentiment Analysis Backend v2.0 starting up...")
    logger.info("Job-based pipeline system enabled for downstream backend models")
    logger.info("Direct API access available for file retrieval")
    yield
    logger.info("Backend shutting down...")

app = FastAPI(
    title="4-Asset Sentiment Analysis Backend with Job-Based Pipeline v2.0",
    description="Job-tracked sentiment analysis with consolidated CSV + asset-specific JSON outputs",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Job-ID", "X-Request-ID"]
)

# ================================================================
# ENHANCED API ENDPOINTS WITH JOB TRACKING
# ================================================================

@app.get("/")
async def root():
    return {
        "message": "4-Asset Sentiment Analysis Backend with Job-Based Pipeline v2.0",
        "version": "2.0.0",
        "assets_supported": [asset.value for asset in AssetType],
        "pipeline_system": {
            "job_tracking": "UUID-based job IDs for request correlation",
            "consolidated_csv": "Single CSV with all 4 assets (tidy format)",
            "asset_contexts": "4 asset-specific JSON files per job",
            "direct_api_access": "Download files via API endpoints"
        },
        "endpoints": [
            "GET /health", "GET /assets", "GET /categories",
            "POST /analyze/all-assets", "POST /analyze/assets", "POST /analyze/category/{category}",
            "GET /jobs", "GET /jobs/{job_id}", "GET /jobs/{job_id}/files",
            "GET /jobs/{job_id}/csv", "GET /jobs/{job_id}/context/{asset}",
            "GET /jobs/{job_id}/manifest"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "assets_available": len(AssetType),
        "job_tracking": "enabled",
        "pipeline_integration": "v2.0",
        "services": {
            "nifty_50": "operational", "gold": "operational",
            "bitcoin": "operational", "reit": "operational"
        }
    }

@app.post("/analyze/all-assets")
async def analyze_all_assets(request: AnalysisRequest):
    """Analyze all 4 assets with job-based pipeline outputs"""
    try:
        # Generate or use provided job ID
        job_id = request.custom_job_id or str(uuid.uuid4())
        
        all_assets = [AssetType.NIFTY50, AssetType.GOLD, AssetType.BITCOIN, AssetType.REIT]
        result = await MultiAssetOrchestrator.analyze_assets_with_job_id(
            job_id, all_assets, request.timeout_minutes
        )

        # Save traditional results
        if request.save_results:
            await _save_analysis_results(result, "all_assets", job_id)

        # Generate job-based pipeline outputs
        if request.generate_pipeline_outputs:
            pipeline_outputs = await generate_job_based_outputs(job_id, result)
            result['pipeline_outputs'] = pipeline_outputs

        # Add job tracking to response headers
        response = JSONResponse(content=result)
        response.headers["X-Job-ID"] = job_id
        return response

    except Exception as e:
        logger.error(f"All-assets analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/assets")
async def analyze_specific_assets(request: AnalysisRequest):
    """Analyze specific assets with job-based pipeline outputs"""
    try:
        job_id = request.custom_job_id or str(uuid.uuid4())
        
        result = await MultiAssetOrchestrator.analyze_assets_with_job_id(
            job_id, request.assets, request.timeout_minutes
        )

        if request.save_results:
            await _save_analysis_results(result, f"assets_{'_'.join([a.value for a in request.assets])}", job_id)

        if request.generate_pipeline_outputs:
            pipeline_outputs = await generate_job_based_outputs(job_id, result)
            result['pipeline_outputs'] = pipeline_outputs

        response = JSONResponse(content=result)
        response.headers["X-Job-ID"] = job_id
        return response

    except Exception as e:
        logger.error(f"Specific assets analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/category/{category}")
async def analyze_asset_category(category: AssetCategory, request: CategoryAnalysisRequest):
    """Analyze assets in a specific category with job tracking"""
    try:
        job_id = request.custom_job_id or str(uuid.uuid4())
        
        result = await MultiAssetOrchestrator.analyze_category_with_job_id(
            job_id, category, request.timeout_minutes
        )

        if request.generate_pipeline_outputs:
            pipeline_outputs = await generate_job_based_outputs(job_id, result)
            result['pipeline_outputs'] = pipeline_outputs

        response = JSONResponse(content=result)
        response.headers["X-Job-ID"] = job_id
        return response

    except Exception as e:
        logger.error(f"Category {category.value} analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ================================================================
# JOB MANAGEMENT ENDPOINTS
# ================================================================

@app.get("/jobs")
async def list_all_jobs():
    """List all available jobs"""
    try:
        base_dir = "pipeline_outputs"
        if not os.path.exists(base_dir):
            return {"jobs": [], "total_jobs": 0}
        
        jobs = []
        for job_id in os.listdir(base_dir):
            job_path = f"{base_dir}/{job_id}"
            if os.path.isdir(job_path):
                # Get job info from manifest if available
                manifest_file = f"{job_path}/job_manifest_{job_id}.json"
                if os.path.exists(manifest_file):
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    jobs.append({
                        'job_id': job_id,
                        'created_at': manifest.get('generated_at'),
                        'successful_assets': manifest.get('analysis_summary', {}).get('successful_assets', 0),
                        'files_available': len(manifest.get('files', {}).get('asset_contexts', [])) + 1
                    })
                else:
                    # Basic job info
                    jobs.append({
                        'job_id': job_id,
                        'created_at': datetime.fromtimestamp(os.path.getctime(job_path)).isoformat(),
                        'files_available': len([f for f in os.listdir(job_path) if f.endswith(('.csv', '.json'))])
                    })
        
        # Sort by creation date (newest first)
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return {
            "jobs": jobs,
            "total_jobs": len(jobs),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job_details(job_id: str):
    """Get detailed information about a specific job"""
    try:
        job_path = f"pipeline_outputs/{job_id}"
        if not os.path.exists(job_path):
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Read manifest file
        manifest_file = f"{job_path}/job_manifest_{job_id}.json"
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            return manifest
        else:
            # Generate basic job info
            files = [f for f in os.listdir(job_path) if f.endswith(('.csv', '.json'))]
            return {
                'job_id': job_id,
                'status': 'completed',
                'files_available': files,
                'total_files': len(files),
                'directory': job_path
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job details: {str(e)}")

@app.get("/jobs/{job_id}/files")
async def list_job_files(job_id: str):
    """List all files for a specific job"""
    try:
        job_path = f"pipeline_outputs/{job_id}"
        if not os.path.exists(job_path):
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        files = []
        for filename in os.listdir(job_path):
            file_path = f"{job_path}/{filename}"
            if os.path.isfile(file_path):
                file_info = {
                    'filename': filename,
                    'size_bytes': os.path.getsize(file_path),
                    'size_kb': round(os.path.getsize(file_path) / 1024, 2),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
                
                if filename.endswith('.csv'):
                    file_info['type'] = 'consolidated_csv'
                    file_info['description'] = 'Consolidated data for Backend Model A'
                    file_info['access_endpoint'] = f'/jobs/{job_id}/csv'
                elif filename.endswith('_context_'):
                    asset_name = filename.split('_context_')[0]
                    file_info['type'] = 'asset_context'
                    file_info['asset'] = asset_name
                    file_info['description'] = f'{asset_name} context for Backend Model B'
                    file_info['access_endpoint'] = f'/jobs/{job_id}/context/{asset_name}'
                elif filename.startswith('job_manifest_'):
                    file_info['type'] = 'job_manifest'
                    file_info['description'] = 'Job metadata and file index'
                    file_info['access_endpoint'] = f'/jobs/{job_id}/manifest'
                
                files.append(file_info)
        
        return {
            'job_id': job_id,
            'files': files,
            'total_files': len(files),
            'directory': job_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list job files: {str(e)}")

# ================================================================
# FILE ACCESS ENDPOINTS FOR DOWNSTREAM MODELS
# ================================================================

@app.get("/jobs/{job_id}/csv")
async def get_job_csv_data(job_id: str, download: bool = False):
    """Get consolidated CSV data for Backend Model A"""
    try:
        csv_file = f"pipeline_outputs/{job_id}/assets_data_{job_id}.csv"
        if not os.path.exists(csv_file):
            raise HTTPException(status_code=404, detail=f"CSV file for job {job_id} not found")
        
        if download:
            return FileResponse(
                csv_file,
                media_type='text/csv',
                filename=f"assets_data_{job_id}.csv"
            )
        else:
            # Return JSON format for API consumption
            import pandas as pd
            df = pd.read_csv(csv_file)
            return {
                'job_id': job_id,
                'data': df.to_dict('records'),
                'metadata': {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'assets_included': df['asset'].unique().tolist(),
                    'file_size_kb': round(os.path.getsize(csv_file) / 1024, 2)
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing CSV: {str(e)}")

@app.get("/jobs/{job_id}/context/{asset}")
async def get_job_asset_context(job_id: str, asset: str, download: bool = False):
    """Get asset-specific context JSON for Backend Model B"""
    try:
        # Validate asset
        if asset.upper() not in ['NIFTY50', 'GOLD', 'BITCOIN', 'REIT']:
            raise HTTPException(status_code=400, detail=f"Invalid asset: {asset}")
        
        context_file = f"pipeline_outputs/{job_id}/{asset.upper()}_context_{job_id}.json"
        if not os.path.exists(context_file):
            raise HTTPException(status_code=404, detail=f"Context file for {asset} in job {job_id} not found")
        
        if download:
            return FileResponse(
                context_file,
                media_type='application/json',
                filename=f"{asset.upper()}_context_{job_id}.json"
            )
        else:
            # Return JSON data directly
            with open(context_file, 'r') as f:
                context_data = json.load(f)
            
            return {
                'job_id': job_id,
                'asset': asset.upper(),
                'context_data': context_data,
                'metadata': {
                    'file_size_kb': round(os.path.getsize(context_file) / 1024, 2),
                    'format_version': context_data.get('pipeline_metadata', {}).get('format_version', '2.0')
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing context file: {str(e)}")

@app.get("/jobs/{job_id}/manifest")
async def get_job_manifest(job_id: str):
    """Get job manifest with file index"""
    try:
        manifest_file = f"pipeline_outputs/{job_id}/job_manifest_{job_id}.json"
        if not os.path.exists(manifest_file):
            raise HTTPException(status_code=404, detail=f"Manifest for job {job_id} not found")
        
        with open(manifest_file, 'r') as f:
            manifest_data = json.load(f)
        
        return manifest_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing manifest: {str(e)}")

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

async def _save_analysis_results(result: Dict, analysis_type: str, job_id: str):
    """Save traditional analysis results with job tracking"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{analysis_type}_{job_id}_{timestamp}.json"
        os.makedirs("analysis_results", exist_ok=True)
        with open(f"analysis_results/{filename}", 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Analysis results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_4asset_backend_job_based:app", host="0.0.0.0", port=8000, reload=True)
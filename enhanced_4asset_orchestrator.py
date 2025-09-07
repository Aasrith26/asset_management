# ENHANCED 4-ASSET ORCHESTRATOR
# ==============================
# Complete orchestrator supporting Nifty 50, Gold, Bitcoin, and REIT

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import asyncio
import uuid
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import aiofiles
import httpx
from dataclasses import dataclass
from enum import Enum
import traceback

# Import configurations and integrations
try:
    from config import config, CSV_PROCESSOR_URL, JSON_PROCESSOR_URL
except ImportError:
    # Fallback configuration
    class MockConfig:
        APP_NAME = "Multi-Asset Sentiment Analysis Orchestrator"
        VERSION = "2.0.0"
        LOG_LEVEL = "INFO"
        DEFAULT_TIMEOUT_MINUTES = 15
        CSV_DIR = Path("./data/csv")
        JSON_DIR = Path("./data/json")
        LOGS_DIR = Path("./data/logs")
        BASE_DATA_DIR = Path("./data")
        
        @classmethod
        def ensure_directories(cls):
            for directory in [cls.CSV_DIR, cls.JSON_DIR, cls.LOGS_DIR]:
                directory.mkdir(parents=True, exist_ok=True)
        
        @classmethod
        def get_asset_config(cls, asset: str):
            configs = {
                "NIFTY50": {"timeout_minutes": 15, "priority": "high"},
                "GOLD": {"timeout_minutes": 20, "priority": "high"},
                "BITCOIN": {"timeout_minutes": 5, "priority": "medium"},
                "REIT": {"timeout_minutes": 10, "priority": "low"}
            }
            return configs.get(asset, {"timeout_minutes": 15, "priority": "normal"})
    
    config = MockConfig()
    CSV_PROCESSOR_URL = "http://localhost:8001/process-csv"
    JSON_PROCESSOR_URL = "http://localhost:8002/process-context"

# Import all asset integration adapters
try:
    from asset_integration import (
        NiftyIntegrationAdapter, 
        GoldIntegrationAdapter,
        BitcoinIntegrationAdapter,  # New
        REITIntegrationAdapter      # New
    )
    print("✅ Successfully imported all 4 asset integration adapters")
except ImportError as e:
    print(f"❌ Could not import some asset integrations: {e}")
    # Create mock adapters for missing ones
    
    class MockBitcoinAdapter:
        @staticmethod
        async def run_analysis():
            return {
                'component_sentiment': 0.45,
                'component_confidence': 0.72,
                'method': 'mock_btc',
                'timestamp': datetime.now().isoformat()
            }
    
    class MockREITAdapter:
        @staticmethod
        async def run_analysis():
            return {
                'component_sentiment': 0.25,
                'component_confidence': 0.68,
                'method': 'mock_reit',
                'timestamp': datetime.now().isoformat()
            }
    
    # Use existing adapters or mocks
    try:
        from asset_integration import NiftyIntegrationAdapter, GoldIntegrationAdapter
    except:
        class MockNiftyAdapter:
            @staticmethod
            async def run_analysis():
                return {'component_sentiment': 0.5, 'component_confidence': 0.7}
        class MockGoldAdapter:
            @staticmethod
            async def run_analysis():
                return {'component_sentiment': 0.6, 'component_confidence': 0.75}
        NiftyIntegrationAdapter = MockNiftyAdapter
        GoldIntegrationAdapter = MockGoldAdapter
    
    BitcoinIntegrationAdapter = MockBitcoinAdapter
    REITIntegrationAdapter = MockREITAdapter

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title=config.APP_NAME,
    description="Enterprise-grade parallel sentiment analysis for 4 asset classes",
    version=config.VERSION
)

# ================================================================
# DATA MODELS - ENHANCED FOR 4 ASSETS
# ================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class AssetType(str, Enum):
    NIFTY50 = "NIFTY50"
    GOLD = "GOLD"
    BITCOIN = "BITCOIN"    # New
    REIT = "REIT"          # New

class AssetCategory(str, Enum):
    TRADITIONAL = "traditional"  # Nifty + Gold
    ALTERNATIVE = "alternative"  # Bitcoin + REIT
    EQUITY = "equity"           # Nifty
    COMMODITY = "commodity"     # Gold
    CRYPTO = "crypto"          # Bitcoin
    REAL_ESTATE = "real_estate" # REIT

@dataclass
class AssetAnalysisResult:
    asset: str
    sentiment_score: float
    confidence: float
    execution_time: float
    timestamp: datetime
    context_data: Dict[str, Any]
    status: str
    error_message: Optional[str] = None

class AnalysisRequest(BaseModel):
    assets: List[AssetType] = [AssetType.NIFTY50, AssetType.GOLD, AssetType.BITCOIN, AssetType.REIT]
    timeout_minutes: int = config.DEFAULT_TIMEOUT_MINUTES
    priority: str = "normal"
    include_portfolio_view: bool = True

class CategoryRequest(BaseModel):
    category: AssetCategory
    timeout_minutes: int = config.DEFAULT_TIMEOUT_MINUTES

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    assets_requested: List[str]
    created_at: datetime
    estimated_completion: Optional[datetime] = None

# ================================================================
# ENHANCED ASSET ANALYZERS - ALL 4 ASSETS
# ================================================================

class Nifty50Analyzer:
    """Nifty 50 analyzer using integration adapter"""
    
    def __init__(self):
        self.asset_name = "NIFTY50"
        self.asset_type = "equity_index"
    
    async def analyze(self, timeout_minutes: int = 15) -> AssetAnalysisResult:
        start_time = datetime.now()
        try:
            result = await asyncio.wait_for(
                NiftyIntegrationAdapter.run_analysis(),
                timeout=timeout_minutes * 60
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            sentiment_score = result.get('composite_sentiment', 0.0)
            confidence = result.get('composite_confidence', 0.5)
            
            return AssetAnalysisResult(
                asset=self.asset_name,
                sentiment_score=sentiment_score,
                confidence=confidence,
                execution_time=execution_time,
                timestamp=datetime.now(),
                context_data=result,
                status="success"
            )
        except Exception as e:
            return self._create_error_result(str(e), (datetime.now() - start_time).total_seconds())
    
    def _create_error_result(self, error_message: str, execution_time: float) -> AssetAnalysisResult:
        return AssetAnalysisResult(
            asset=self.asset_name,
            sentiment_score=0.0,
            confidence=0.3,
            execution_time=execution_time,
            timestamp=datetime.now(),
            context_data={},
            status="error",
            error_message=error_message
        )

class GoldAnalyzer:
    """Gold analyzer using integration adapter"""
    
    def __init__(self):
        self.asset_name = "GOLD"
        self.asset_type = "commodity"
    
    async def analyze(self, timeout_minutes: int = 20) -> AssetAnalysisResult:
        start_time = datetime.now()
        try:
            result = await asyncio.wait_for(
                GoldIntegrationAdapter.run_analysis(),
                timeout=timeout_minutes * 60
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            executive_summary = result.get('executive_summary', {})
            sentiment_score = executive_summary.get('overall_sentiment', 0.0)
            confidence = executive_summary.get('confidence_level', 0.5)
            
            return AssetAnalysisResult(
                asset=self.asset_name,
                sentiment_score=sentiment_score,
                confidence=confidence,
                execution_time=execution_time,
                timestamp=datetime.now(),
                context_data=result,
                status="success"
            )
        except Exception as e:
            return self._create_error_result(str(e), (datetime.now() - start_time).total_seconds())
    
    def _create_error_result(self, error_message: str, execution_time: float) -> AssetAnalysisResult:
        return AssetAnalysisResult(
            asset=self.asset_name,
            sentiment_score=0.0,
            confidence=0.3,
            execution_time=execution_time,
            timestamp=datetime.now(),
            context_data={},
            status="error",
            error_message=error_message
        )

class BitcoinAnalyzer:
    """Bitcoin analyzer using integration adapter"""
    
    def __init__(self):
        self.asset_name = "BITCOIN"
        self.asset_type = "cryptocurrency"
    
    async def analyze(self, timeout_minutes: int = 5) -> AssetAnalysisResult:
        start_time = datetime.now()
        try:
            result = await asyncio.wait_for(
                BitcoinIntegrationAdapter.run_analysis(),
                timeout=timeout_minutes * 60
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            sentiment_score = result.get('component_sentiment', 0.0)
            confidence = result.get('component_confidence', 0.5)
            
            return AssetAnalysisResult(
                asset=self.asset_name,
                sentiment_score=sentiment_score,
                confidence=confidence,
                execution_time=execution_time,
                timestamp=datetime.now(),
                context_data=result,
                status="success"
            )
        except Exception as e:
            return self._create_error_result(str(e), (datetime.now() - start_time).total_seconds())
    
    def _create_error_result(self, error_message: str, execution_time: float) -> AssetAnalysisResult:
        return AssetAnalysisResult(
            asset=self.asset_name,
            sentiment_score=0.0,
            confidence=0.3,
            execution_time=execution_time,
            timestamp=datetime.now(),
            context_data={},
            status="error",
            error_message=error_message
        )

class REITAnalyzer:
    """REIT analyzer using integration adapter"""
    
    def __init__(self):
        self.asset_name = "REIT"
        self.asset_type = "real_estate_investment_trust"
    
    async def analyze(self, timeout_minutes: int = 10) -> AssetAnalysisResult:
        start_time = datetime.now()
        try:
            result = await asyncio.wait_for(
                REITIntegrationAdapter.run_analysis(),
                timeout=timeout_minutes * 60
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            sentiment_score = result.get('component_sentiment', 0.0)
            confidence = result.get('component_confidence', 0.5)
            
            return AssetAnalysisResult(
                asset=self.asset_name,
                sentiment_score=sentiment_score,
                confidence=confidence,
                execution_time=execution_time,
                timestamp=datetime.now(),
                context_data=result,
                status="success"
            )
        except Exception as e:
            return self._create_error_result(str(e), (datetime.now() - start_time).total_seconds())
    
    def _create_error_result(self, error_message: str, execution_time: float) -> AssetAnalysisResult:
        return AssetAnalysisResult(
            asset=self.asset_name,
            sentiment_score=0.0,
            confidence=0.3,
            execution_time=execution_time,
            timestamp=datetime.now(),
            context_data={},
            status="error",
            error_message=error_message
        )

# ================================================================
# FILE MANAGEMENT - ENHANCED FOR 4 ASSETS
# ================================================================

class FileManager:
    """Enhanced file management for 4 assets"""
    
    @staticmethod
    async def generate_csv_file(results: List[AssetAnalysisResult]) -> str:
        """Generate consolidated CSV file for all assets"""
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"multi_asset_sentiment_{timestamp_str}.csv"
        filepath = config.CSV_DIR / filename
        
        config.CSV_DIR.mkdir(parents=True, exist_ok=True)
        
        csv_data = []
        for result in results:
            csv_data.append({
                'timestamp': result.timestamp.isoformat(),
                'asset': result.asset,
                'asset_type': getattr(result, 'asset_type', 'unknown'),
                'sentiment_score': result.sentiment_score,
                'confidence': result.confidence,
                'execution_time': result.execution_time,
                'status': result.status
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Generated multi-asset CSV file: {filename}")
        return str(filepath)
    
    @staticmethod
    async def generate_portfolio_summary(results: List[AssetAnalysisResult]) -> Dict:
        """Generate portfolio-level summary"""
        successful_results = [r for r in results if r.status == 'success']
        
        if not successful_results:
            return {"error": "No successful analyses"}
        
        # Portfolio weights (equal weight for simplicity)
        equal_weight = 1.0 / len(successful_results)
        
        portfolio_sentiment = sum(r.sentiment_score * equal_weight for r in successful_results)
        portfolio_confidence = sum(r.confidence * equal_weight for r in successful_results)
        
        # Asset breakdown
        asset_breakdown = {}
        for result in successful_results:
            asset_breakdown[result.asset] = {
                "sentiment": result.sentiment_score,
                "confidence": result.confidence,
                "weight": equal_weight,
                "contribution": result.sentiment_score * equal_weight
            }
        
        return {
            "portfolio_sentiment": portfolio_sentiment,
            "portfolio_confidence": portfolio_confidence,
            "total_assets": len(results),
            "successful_assets": len(successful_results),
            "asset_breakdown": asset_breakdown,
            "timestamp": datetime.now().isoformat()
        }

# ================================================================
# MAIN ORCHESTRATOR - ENHANCED FOR 4 ASSETS
# ================================================================

class MultiAssetOrchestrator:
    """Enhanced orchestrator for 4 asset classes"""
    
    ANALYZERS = {
        AssetType.NIFTY50: Nifty50Analyzer,
        AssetType.GOLD: GoldAnalyzer,
        AssetType.BITCOIN: BitcoinAnalyzer,  # New
        AssetType.REIT: REITAnalyzer         # New
    }
    
    CATEGORY_MAPPINGS = {
        AssetCategory.TRADITIONAL: [AssetType.NIFTY50, AssetType.GOLD],
        AssetCategory.ALTERNATIVE: [AssetType.BITCOIN, AssetType.REIT],
        AssetCategory.EQUITY: [AssetType.NIFTY50],
        AssetCategory.COMMODITY: [AssetType.GOLD],
        AssetCategory.CRYPTO: [AssetType.BITCOIN],
        AssetCategory.REAL_ESTATE: [AssetType.REIT]
    }
    
    @classmethod
    async def run_parallel_analysis(cls, request: AnalysisRequest, job_id: str) -> List[AssetAnalysisResult]:
        """Run parallel analysis for requested assets"""
        tasks = []
        for asset_type in request.assets:
            if asset_type in cls.ANALYZERS:
                analyzer = cls.ANALYZERS[asset_type]()
                asset_config = config.get_asset_config(asset_type.value)
                timeout = asset_config.get('timeout_minutes', request.timeout_minutes)
                
                task = cls._analyze_with_progress_tracking(analyzer, job_id, timeout)
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    asset_name = request.assets[i].value
                    error_result = AssetAnalysisResult(
                        asset=asset_name,
                        sentiment_score=0.0,
                        confidence=0.3,
                        execution_time=0.0,
                        timestamp=datetime.now(),
                        context_data={},
                        status="exception",
                        error_message=str(result)
                    )
                    final_results.append(error_result)
                else:
                    final_results.append(result)
            
            return final_results
        
        return []
    
    @classmethod
    async def run_category_analysis(cls, category: AssetCategory, timeout_minutes: int, job_id: str) -> List[AssetAnalysisResult]:
        """Run analysis for a specific asset category"""
        assets = cls.CATEGORY_MAPPINGS.get(category, [])
        request = AnalysisRequest(assets=assets, timeout_minutes=timeout_minutes)
        return await cls.run_parallel_analysis(request, job_id)

# ================================================================
# API ENDPOINTS - ENHANCED FOR 4 ASSETS
# ================================================================

@app.post("/analyze/all-assets", response_model=JobResponse)
async def analyze_all_assets(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis for all requested assets (up to 4)"""
    try:
        job_id = JobManager.create_job(request)
        background_tasks.add_task(process_analysis_job, job_id, request)
        
        estimated_completion = datetime.now() + timedelta(minutes=request.timeout_minutes + 10)
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            assets_requested=[asset.value for asset in request.assets],
            created_at=datetime.now(),
            estimated_completion=estimated_completion
        )
    except Exception as e:
        logger.error(f"Failed to start analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/category/{category}")
async def analyze_asset_category(category: AssetCategory, request: CategoryRequest, background_tasks: BackgroundTasks):
    """Analyze assets by category (traditional, alternative, etc.)"""
    try:
        assets = MultiAssetOrchestrator.CATEGORY_MAPPINGS.get(category, [])
        if not assets:
            raise HTTPException(status_code=400, detail=f"Unknown category: {category}")
        
        analysis_request = AnalysisRequest(assets=assets, timeout_minutes=request.timeout_minutes)
        job_id = JobManager.create_job(analysis_request)
        background_tasks.add_task(process_analysis_job, job_id, analysis_request)
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            assets_requested=[asset.value for asset in assets],
            created_at=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to start category analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check for 4 assets"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': config.VERSION,
        'available_assets': ['NIFTY50', 'GOLD', 'BITCOIN', 'REIT'],
        'asset_categories': ['traditional', 'alternative', 'equity', 'commodity', 'crypto', 'real_estate'],
        'total_assets': 4,
        'active_jobs': len(JobManager.jobs)
    }

@app.get("/assets")
async def list_assets():
    """List all 4 available assets with details"""
    return {
        'available_assets': {
            'NIFTY50': {'type': 'equity_index', 'category': 'traditional', 'update_frequency': '15min'},
            'GOLD': {'type': 'commodity', 'category': 'traditional', 'update_frequency': '30min'},
            'BITCOIN': {'type': 'cryptocurrency', 'category': 'alternative', 'update_frequency': '1min'},
            'REIT': {'type': 'real_estate', 'category': 'alternative', 'update_frequency': 'daily'}
        },
        'categories': {
            'traditional': ['NIFTY50', 'GOLD'],
            'alternative': ['BITCOIN', 'REIT']
        },
        'total_assets': 4
    }

# Include existing endpoints (status, results, etc.) with JobManager class
class JobManager:
    """Enhanced job management for 4-asset system"""
    jobs: Dict[str, Dict] = {}
    
    @classmethod
    def create_job(cls, request: AnalysisRequest) -> str:
        job_id = str(uuid.uuid4())
        cls.jobs[job_id] = {
            'id': job_id,
            'status': JobStatus.PENDING,
            'request': request,
            'created_at': datetime.now(),
            'progress': {asset.value: 'pending' for asset in request.assets},
            'results': None,
            'files': None,
            'error_message': None
        }
        return job_id
    
    @classmethod
    def update_job_status(cls, job_id: str, status: JobStatus, **kwargs):
        if job_id in cls.jobs:
            cls.jobs[job_id]['status'] = status
            for key, value in kwargs.items():
                cls.jobs[job_id][key] = value
    
    @classmethod
    def get_job(cls, job_id: str) -> Optional[Dict]:
        return cls.jobs.get(job_id)
    
    @classmethod
    def update_asset_progress(cls, job_id: str, asset: str, status: str):
        if job_id in cls.jobs:
            cls.jobs[job_id]['progress'][asset] = status

# Background processing function
async def process_analysis_job(job_id: str, request: AnalysisRequest):
    """Background task to process 4-asset analysis job"""
    try:
        logger.info(f"Starting 4-asset job {job_id}")
        JobManager.update_job_status(job_id, JobStatus.RUNNING)
        
        # Run parallel analysis
        results = await MultiAssetOrchestrator.run_parallel_analysis(request, job_id)
        
        # Generate files
        csv_file = await FileManager.generate_csv_file(results)
        json_files = await FileManager.generate_json_files(results)
        
        # Generate portfolio summary if requested
        portfolio_summary = None
        if request.include_portfolio_view:
            portfolio_summary = await FileManager.generate_portfolio_summary(results)
        
        # Determine final status
        successful = [r for r in results if r.status == 'success']
        if len(successful) == len(results):
            final_status = JobStatus.COMPLETED
        elif len(successful) > 0:
            final_status = JobStatus.PARTIAL
        else:
            final_status = JobStatus.FAILED
        
        # Update job with results
        JobManager.update_job_status(
            job_id,
            final_status,
            results={
                'analysis_results': [{
                    'asset': r.asset,
                    'sentiment_score': r.sentiment_score,
                    'confidence': r.confidence,
                    'status': r.status,
                    'execution_time': r.execution_time
                } for r in results],
                'portfolio_summary': portfolio_summary
            },
            files={'csv_file': csv_file, 'json_files': json_files}
        )
        
        logger.info(f"Completed 4-asset job {job_id} with status {final_status}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        JobManager.update_job_status(job_id, JobStatus.FAILED, error_message=str(e))

# Include remaining endpoints and startup code...
@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'results': job['results'],
        'files': job['files']
    }

@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job['status'] not in [JobStatus.COMPLETED, JobStatus.PARTIAL]:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    return job

@app.on_event("startup")
async def startup():
    logger.info(f"Starting {config.APP_NAME} v{config.VERSION}")
    config.ensure_directories()
    logger.info("4-Asset Orchestrator ready for requests")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "enhanced_4asset_orchestrator:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# FIXED ORCHESTRATOR WITH PROPER INTEGRATIONS v1.1
# ==================================================
# Updated orchestrator that uses your asset integration adapters

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
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

# Import our configurations and integrations
try:
    from config import config, CSV_PROCESSOR_URL, JSON_PROCESSOR_URL
except ImportError:
    # Fallback configuration if config.py has issues
    class MockConfig:
        APP_NAME = "Multi-Asset Sentiment Analysis Orchestrator"
        VERSION = "1.1.0"
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
            return {
                "timeout_minutes": 15,
                "cache_ttl_minutes": 30,
                "priority": "high",
                "max_retries": 2
            }
    
    config = MockConfig()
    CSV_PROCESSOR_URL = "http://localhost:8001/process-csv"
    JSON_PROCESSOR_URL = "http://localhost:8002/process-context"

try:
    from asset_integration import NiftyIntegrationAdapter, GoldIntegrationAdapter
    print("Successfully imported asset integration adapters")
except ImportError as e:
    print(f"Could not import asset integrations: {e}")
    print("Make sure asset_integration.py is in the same directory")
    
    # Create mock adapters as fallback
    class MockNiftyAdapter:
        @staticmethod
        async def run_analysis():
            return {
                'composite_sentiment': 0.5,
                'composite_confidence': 0.7,
                'method': 'mock_fallback',
                'timestamp': datetime.now().isoformat()
            }
    
    class MockGoldAdapter:
        @staticmethod
        async def run_analysis():
            return {
                'executive_summary': {
                    'overall_sentiment': 0.6,
                    'confidence_level': 0.75,
                    'interpretation': 'Mock analysis'
                },
                'method': 'mock_fallback',
                'timestamp': datetime.now().isoformat()
            }
    
    NiftyIntegrationAdapter = MockNiftyAdapter
    GoldIntegrationAdapter = MockGoldAdapter

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title=config.APP_NAME,
    description="Enterprise-grade parallel sentiment analysis for multiple asset classes",
    version=config.VERSION
)

# ================================================================
# DATA MODELS
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
    assets: List[AssetType] = [AssetType.NIFTY50, AssetType.GOLD]
    timeout_minutes: int = config.DEFAULT_TIMEOUT_MINUTES
    priority: str = "normal"

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    assets_requested: List[str]
    created_at: datetime
    estimated_completion: Optional[datetime] = None

# ================================================================
# ASSET ANALYZERS (Using Integration Adapters)
# ================================================================

class Nifty50Analyzer:
    """Nifty 50 analyzer using integration adapter"""
    
    def __init__(self):
        self.asset_name = "NIFTY50"
    
    async def analyze(self, timeout_minutes: int = 15) -> AssetAnalysisResult:
        """Run Nifty 50 analysis using integration adapter"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting {self.asset_name} analysis...")
            
            # Use the integration adapter
            result = await asyncio.wait_for(
                NiftyIntegrationAdapter.run_analysis(),
                timeout=timeout_minutes * 60
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract sentiment from your Nifty system's output format
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
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{self.asset_name} analysis timed out after {timeout_minutes} minutes")
            
            return AssetAnalysisResult(
                asset=self.asset_name,
                sentiment_score=0.0,
                confidence=0.3,
                execution_time=execution_time,
                timestamp=datetime.now(),
                context_data={},
                status="timeout",
                error_message=f"Analysis timed out after {timeout_minutes} minutes"
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{self.asset_name} analysis failed: {str(e)}")
            
            return AssetAnalysisResult(
                asset=self.asset_name,
                sentiment_score=0.0,
                confidence=0.3,
                execution_time=execution_time,
                timestamp=datetime.now(),
                context_data={},
                status="error",
                error_message=str(e)
            )

class GoldAnalyzer:
    """Gold analyzer using integration adapter"""
    
    def __init__(self):
        self.asset_name = "GOLD"
    
    async def analyze(self, timeout_minutes: int = 20) -> AssetAnalysisResult:
        """Run Gold analysis using integration adapter"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting {self.asset_name} analysis...")
            
            # Use the integration adapter
            result = await asyncio.wait_for(
                GoldIntegrationAdapter.run_analysis(),
                timeout=timeout_minutes * 60
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract sentiment from your Gold system's output format
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
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{self.asset_name} analysis timed out after {timeout_minutes} minutes")
            
            return AssetAnalysisResult(
                asset=self.asset_name,
                sentiment_score=0.0,
                confidence=0.3,
                execution_time=execution_time,
                timestamp=datetime.now(),
                context_data={},
                status="timeout",
                error_message=f"Analysis timed out after {timeout_minutes} minutes"
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{self.asset_name} analysis failed: {str(e)}")
            
            return AssetAnalysisResult(
                asset=self.asset_name,
                sentiment_score=0.0,
                confidence=0.3,
                execution_time=execution_time,
                timestamp=datetime.now(),
                context_data={},
                status="error",
                error_message=str(e)
            )

# ================================================================
# FILE MANAGEMENT
# ================================================================

class FileManager:
    """Handles CSV and JSON file generation"""
    
    @staticmethod
    async def generate_csv_file(results: List[AssetAnalysisResult]) -> str:
        """Generate consolidated CSV file"""
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"sentiment_scores_{timestamp_str}.csv"
        filepath = config.CSV_DIR / filename
        
        # Ensure directory exists
        config.CSV_DIR.mkdir(parents=True, exist_ok=True)
        
        # Prepare CSV data
        csv_data = []
        for result in results:
            csv_data.append({
                'timestamp': result.timestamp.isoformat(),
                'asset': result.asset,
                'sentiment_score': result.sentiment_score,
                'confidence': result.confidence,
                'execution_time': result.execution_time,
                'status': result.status
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Generated CSV file: {filename}")
        return str(filepath)
    
    @staticmethod
    async def generate_json_files(results: List[AssetAnalysisResult]) -> Dict[str, str]:
        """Generate individual JSON context files"""
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_files = {}
        
        # Ensure directory exists
        config.JSON_DIR.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            filename = f"{result.asset.lower()}_context_{timestamp_str}.json"
            filepath = config.JSON_DIR / filename
            
            # Prepare JSON context
            context = {
                'asset_metadata': {
                    'asset_name': result.asset,
                    'analysis_timestamp': result.timestamp.isoformat(),
                    'execution_time_seconds': result.execution_time,
                    'status': result.status
                },
                'composite_results': {
                    'sentiment_score': result.sentiment_score,
                    'confidence': result.confidence,
                    'status': result.status
                },
                'detailed_context': result.context_data,
                'system_metadata': {
                    'orchestrator_version': config.VERSION,
                    'generated_at': datetime.now().isoformat()
                }
            }
            
            if result.error_message:
                context['error_details'] = {
                    'error_message': result.error_message,
                    'error_type': result.status
                }
            
            # Save JSON file
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(context, indent=2, default=str))
            
            json_files[result.asset] = str(filepath)
            logger.info(f"Generated JSON context file: {filename}")
        
        return json_files

# ================================================================
# INTEGRATION DISPATCHER
# ================================================================

class IntegrationDispatcher:
    """Routes files to downstream systems"""
    
    @staticmethod
    async def send_csv_to_processor(csv_file_path: str) -> bool:
        """Send CSV file to Model A"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                with open(csv_file_path, 'rb') as f:
                    files = {'file': f}
                    response = await client.post(CSV_PROCESSOR_URL, files=files)
                
                if response.status_code == 200:
                    logger.info(f"Successfully sent CSV to processor: {csv_file_path}")
                    return True
                else:
                    logger.warning(f"CSV processor returned status {response.status_code}")
                    return False
        
        except httpx.ConnectError:
            logger.warning(f"Cannot connect to CSV processor at {CSV_PROCESSOR_URL}")
            return False
        except Exception as e:
            logger.error(f"Error sending CSV: {str(e)}")
            return False
    
    @staticmethod
    async def send_json_to_api(json_file_path: str, asset: str) -> bool:
        """Send JSON context to Model B API"""
        try:
            async with aiofiles.open(json_file_path, 'r') as f:
                json_content = await f.read()
                context_data = json.loads(json_content)
            
            async with httpx.AsyncClient(timeout=45.0) as client:
                payload = {
                    'asset': asset,
                    'context': context_data
                }
                response = await client.post(JSON_PROCESSOR_URL, json=payload)
                
                if response.status_code == 200:
                    logger.info(f"Successfully sent JSON context for {asset}")
                    return True
                else:
                    logger.warning(f"JSON processor returned status {response.status_code}")
                    return False
        
        except httpx.ConnectError:
            logger.warning(f"Cannot connect to JSON processor at {JSON_PROCESSOR_URL}")
            return False
        except Exception as e:
            logger.error(f"Error sending JSON for {asset}: {str(e)}")
            return False

# ================================================================
# JOB MANAGEMENT
# ================================================================

class JobManager:
    """Manages analysis jobs and their status"""
    jobs: Dict[str, Dict] = {}
    
    @classmethod
    def create_job(cls, request: AnalysisRequest) -> str:
        """Create a new analysis job"""
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
        """Update job status"""
        if job_id in cls.jobs:
            cls.jobs[job_id]['status'] = status
            for key, value in kwargs.items():
                cls.jobs[job_id][key] = value
    
    @classmethod
    def get_job(cls, job_id: str) -> Optional[Dict]:
        """Get job by ID"""
        return cls.jobs.get(job_id)
    
    @classmethod
    def update_asset_progress(cls, job_id: str, asset: str, status: str):
        """Update progress for specific asset"""
        if job_id in cls.jobs:
            cls.jobs[job_id]['progress'][asset] = status

# ================================================================
# MAIN ORCHESTRATOR
# ================================================================

class MultiAssetOrchestrator:
    """Main orchestrator for parallel asset analysis"""
    
    # Asset analyzer mapping
    ANALYZERS = {
        AssetType.NIFTY50: Nifty50Analyzer,
        AssetType.GOLD: GoldAnalyzer
    }
    
    @classmethod
    async def run_parallel_analysis(cls, request: AnalysisRequest, job_id: str) -> List[AssetAnalysisResult]:
        """Run parallel analysis for all requested assets"""
        
        tasks = []
        for asset_type in request.assets:
            if asset_type in cls.ANALYZERS:
                analyzer = cls.ANALYZERS[asset_type]()
                asset_config = config.get_asset_config(asset_type.value)
                timeout = asset_config.get('timeout_minutes', request.timeout_minutes)
                
                task = cls._analyze_with_progress_tracking(analyzer, job_id, timeout)
                tasks.append(task)
            else:
                logger.warning(f"No analyzer available for asset type: {asset_type}")
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
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
    async def _analyze_with_progress_tracking(cls, analyzer, job_id: str, timeout_minutes: int) -> AssetAnalysisResult:
        """Analyze asset with progress tracking"""
        asset_name = analyzer.asset_name
        
        try:
            JobManager.update_asset_progress(job_id, asset_name, 'running')
            result = await analyzer.analyze(timeout_minutes)
            
            if result.status == 'success':
                JobManager.update_asset_progress(job_id, asset_name, 'completed')
            else:
                JobManager.update_asset_progress(job_id, asset_name, 'failed')
            
            return result
            
        except Exception as e:
            JobManager.update_asset_progress(job_id, asset_name, 'error')
            logger.error(f"Error in {asset_name} analysis: {str(e)}")
            
            return AssetAnalysisResult(
                asset=asset_name,
                sentiment_score=0.0,
                confidence=0.3,
                execution_time=0.0,
                timestamp=datetime.now(),
                context_data={},
                status="error",
                error_message=str(e)
            )

# ================================================================
# API ENDPOINTS
# ================================================================

@app.post("/analyze/all-assets", response_model=JobResponse)
async def analyze_all_assets(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis for all requested assets"""
    try:
        job_id = JobManager.create_job(request)
        background_tasks.add_task(process_analysis_job, job_id, request)
        
        estimated_completion = datetime.now() + timedelta(minutes=request.timeout_minutes + 5)
        
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

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
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
    """Get detailed job results"""
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] not in [JobStatus.COMPLETED, JobStatus.PARTIAL]:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return job

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': config.VERSION,
        'available_assets': ['NIFTY50', 'GOLD'],
        'active_jobs': len(JobManager.jobs)
    }

@app.get("/assets")
async def list_assets():
    """List available assets"""
    return {
        'available_assets': ['NIFTY50', 'GOLD'],
        'total_assets': 2
    }

# ================================================================
# BACKGROUND PROCESSING
# ================================================================

async def process_analysis_job(job_id: str, request: AnalysisRequest):
    """Background task to process analysis job"""
    try:
        logger.info(f"Starting job {job_id}")
        JobManager.update_job_status(job_id, JobStatus.RUNNING)
        
        # Run parallel analysis
        results = await MultiAssetOrchestrator.run_parallel_analysis(request, job_id)
        
        # Generate files
        csv_file = await FileManager.generate_csv_file(results)
        json_files = await FileManager.generate_json_files(results)
        
        # Route files (with error handling)
        csv_sent = await IntegrationDispatcher.send_csv_to_processor(csv_file)
        json_routing = {}
        for asset, json_file in json_files.items():
            json_routing[asset] = await IntegrationDispatcher.send_json_to_api(json_file, asset)
        
        # Determine final status
        successful = [r for r in results if r.status == 'success']
        if len(successful) == len(results):
            final_status = JobStatus.COMPLETED
        elif len(successful) > 0:
            final_status = JobStatus.PARTIAL
        else:
            final_status = JobStatus.FAILED
        
        # Update job
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
                'integration_status': {
                    'csv_sent': csv_sent,
                    'json_routing': json_routing
                }
            },
            files={'csv_file': csv_file, 'json_files': json_files}
        )
        
        logger.info(f"Completed job {job_id} with status {final_status}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        JobManager.update_job_status(job_id, JobStatus.FAILED, error_message=str(e))

# ================================================================
# STARTUP
# ================================================================

@app.on_event("startup")
async def startup():
    """Initialize application"""
    logger.info(f"Starting {config.APP_NAME} v{config.VERSION}")
    config.ensure_directories()
    logger.info("Orchestrator ready for requests")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fixed_orchestrator:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
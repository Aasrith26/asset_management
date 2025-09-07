# MULTI-ASSET SENTIMENT ANALYSIS ORCHESTRATOR v1.0
# ===================================================
# FastAPI-based orchestrator for parallel asset sentiment analysis
# Supports Nifty 50, Gold, and future asset expansions

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Multi-Asset Sentiment Analysis API",
    description="Enterprise-grade parallel sentiment analysis for multiple asset classes",
    version="1.0.0"
)

# ===================================================
# DATA MODELS & ENUMS
# ===================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class AssetType(str, Enum):
    NIFTY50 = "NIFTY50"
    GOLD = "GOLD"
    CRYPTO = "CRYPTO"  # Future
    FOREX = "FOREX"    # Future
    COMMODITIES = "COMMODITIES"  # Future

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
    priority: str = "normal"  # normal, high, low
    include_context: bool = True
    timeout_minutes: int = 15

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    assets_requested: List[str]
    created_at: datetime
    estimated_completion: Optional[datetime] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Dict[str, str]  # asset -> status
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    files_generated: Optional[Dict[str, str]] = None

# ===================================================
# CONFIGURATION MANAGEMENT
# ===================================================

class Config:
    """Centralized configuration for the orchestrator"""
    
    # File paths
    BASE_DATA_DIR = Path("./data")
    CSV_DIR = BASE_DATA_DIR / "csv"
    JSON_DIR = BASE_DATA_DIR / "json"
    LOGS_DIR = BASE_DATA_DIR / "logs"
    
    # Downstream integration
    CSV_PROCESSOR_URL = "http://localhost:8001/process-csv"  # Model A
    JSON_PROCESSOR_URL = "http://localhost:8002/process-context"  # Model B API
    
    # Performance settings
    MAX_PARALLEL_ASSETS = 10
    DEFAULT_TIMEOUT_MINUTES = 15
    RETRY_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 5
    
    # File retention
    CLEANUP_AFTER_HOURS = 24
    ARCHIVE_AFTER_DAYS = 7
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        for directory in [cls.CSV_DIR, cls.JSON_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
Config.ensure_directories()

# ===================================================
# ASSET ANALYZER INTERFACE
# ===================================================

class AssetAnalyzerInterface:
    """Standardized interface for all asset analyzers"""
    
    async def analyze(self, timeout_minutes: int = 15) -> AssetAnalysisResult:
        """Run asset analysis and return standardized result"""
        raise NotImplementedError("Subclass must implement analyze method")
    
    def get_asset_name(self) -> str:
        """Return asset name identifier"""
        raise NotImplementedError("Subclass must implement get_asset_name method")

class Nifty50Analyzer(AssetAnalyzerInterface):
    """Nifty 50 sentiment analyzer wrapper"""
    
    def __init__(self):
        self.asset_name = "NIFTY50"
    
    async def analyze(self, timeout_minutes: int = 15) -> AssetAnalysisResult:
        """Run Nifty 50 analysis"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting {self.asset_name} analysis...")
            
            # Import and run your existing Nifty analyzer
            from your_nifty_system import run_nifty_analysis  # Update with actual import
            
            # Run with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(run_nifty_analysis),
                timeout=timeout_minutes * 60
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract standardized fields from your result
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
    
    def get_asset_name(self) -> str:
        return self.asset_name

class GoldAnalyzer(AssetAnalyzerInterface):
    """Gold sentiment analyzer wrapper"""
    
    def __init__(self):
        self.asset_name = "GOLD"
    
    async def analyze(self, timeout_minutes: int = 15) -> AssetAnalysisResult:
        """Run Gold analysis"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting {self.asset_name} analysis...")
            
            # Import and run your existing Gold analyzer
            from your_gold_system import run_gold_analysis  # Update with actual import
            
            # Run with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(run_gold_analysis),
                timeout=timeout_minutes * 60
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract standardized fields from your result
            sentiment_score = result.get('executive_summary', {}).get('overall_sentiment', 0.0)
            confidence = result.get('executive_summary', {}).get('confidence_level', 0.5)
            
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
    
    def get_asset_name(self) -> str:
        return self.asset_name

# ===================================================
# ASSET ANALYZER FACTORY
# ===================================================

class AssetAnalyzerFactory:
    """Factory to create asset analyzers"""
    
    _analyzers = {
        AssetType.NIFTY50: Nifty50Analyzer,
        AssetType.GOLD: GoldAnalyzer,
        # Future assets can be added here
    }
    
    @classmethod
    def create_analyzer(cls, asset_type: AssetType) -> AssetAnalyzerInterface:
        """Create analyzer instance for given asset type"""
        if asset_type not in cls._analyzers:
            raise ValueError(f"No analyzer available for asset type: {asset_type}")
        
        return cls._analyzers[asset_type]()
    
    @classmethod
    def get_available_assets(cls) -> List[AssetType]:
        """Get list of available asset types"""
        return list(cls._analyzers.keys())

# ===================================================
# FILE MANAGEMENT SYSTEM
# ===================================================

class FileManager:
    """Handles CSV and JSON file generation and management"""
    
    @staticmethod
    async def generate_csv_file(results: List[AssetAnalysisResult]) -> str:
        """Generate consolidated CSV file from analysis results"""
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"sentiment_scores_{timestamp_str}.csv"
        filepath = Config.CSV_DIR / filename
        
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
        """Generate individual JSON context files for each asset"""
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_files = {}
        
        for result in results:
            filename = f"{result.asset.lower()}_context_{timestamp_str}.json"
            filepath = Config.JSON_DIR / filename
            
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
                    'orchestrator_version': '1.0.0',
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
    
    @staticmethod
    async def cleanup_old_files():
        """Clean up old files based on retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=Config.CLEANUP_AFTER_HOURS)
        
        for directory in [Config.CSV_DIR, Config.JSON_DIR]:
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")

# ===================================================
# INTEGRATION DISPATCHER
# ===================================================

class IntegrationDispatcher:
    """Handles routing of files to downstream systems"""
    
    @staticmethod
    async def send_csv_to_processor(csv_file_path: str) -> bool:
        """Send CSV file to Model A (CSV processor)"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                with open(csv_file_path, 'rb') as f:
                    files = {'file': f}
                    response = await client.post(Config.CSV_PROCESSOR_URL, files=files)
                    
                if response.status_code == 200:
                    logger.info(f"Successfully sent CSV to processor: {csv_file_path}")
                    return True
                else:
                    logger.error(f"Failed to send CSV to processor. Status: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending CSV to processor: {str(e)}")
            return False
    
    @staticmethod
    async def send_json_to_api(json_file_path: str, asset: str) -> bool:
        """Send JSON context file to Model B API"""
        try:
            async with aiofiles.open(json_file_path, 'r') as f:
                json_content = await f.read()
                context_data = json.loads(json_content)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    'asset': asset,
                    'context': context_data
                }
                response = await client.post(Config.JSON_PROCESSOR_URL, json=payload)
                
                if response.status_code == 200:
                    logger.info(f"Successfully sent JSON context for {asset}")
                    return True
                else:
                    logger.error(f"Failed to send JSON for {asset}. Status: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending JSON for {asset}: {str(e)}")
            return False
    
    @staticmethod
    async def route_all_files(csv_file: str, json_files: Dict[str, str]) -> Dict[str, bool]:
        """Route all generated files to their destinations"""
        routing_results = {}
        
        # Send CSV file
        routing_results['csv_sent'] = await IntegrationDispatcher.send_csv_to_processor(csv_file)
        
        # Send JSON files
        for asset, json_file in json_files.items():
            routing_results[f'json_{asset}_sent'] = await IntegrationDispatcher.send_json_to_api(json_file, asset)
        
        return routing_results

# ===================================================
# JOB MANAGEMENT
# ===================================================

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
        """Update job status and additional fields"""
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

# ===================================================
# MAIN ORCHESTRATOR
# ===================================================

class MultiAssetOrchestrator:
    """Main orchestrator for parallel asset analysis"""
    
    @staticmethod
    async def analyze_single_asset(analyzer: AssetAnalyzerInterface, job_id: str, timeout_minutes: int) -> AssetAnalysisResult:
        """Analyze a single asset with progress tracking"""
        asset_name = analyzer.get_asset_name()
        
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
            logger.error(f"Unexpected error in {asset_name} analysis: {str(e)}")
            
            return AssetAnalysisResult(
                asset=asset_name,
                sentiment_score=0.0,
                confidence=0.3,
                execution_time=0.0,
                timestamp=datetime.now(),
                context_data={},
                status="error",
                error_message=f"Unexpected error: {str(e)}"
            )
    
    @staticmethod
    async def run_parallel_analysis(request: AnalysisRequest, job_id: str) -> List[AssetAnalysisResult]:
        """Run parallel analysis for all requested assets"""
        analyzers = []
        
        # Create analyzers for requested assets
        for asset_type in request.assets:
            try:
                analyzer = AssetAnalyzerFactory.create_analyzer(asset_type)
                analyzers.append(analyzer)
            except ValueError as e:
                logger.error(f"Cannot create analyzer for {asset_type}: {str(e)}")
                # Create error result for unsupported asset
                error_result = AssetAnalysisResult(
                    asset=asset_type.value,
                    sentiment_score=0.0,
                    confidence=0.0,
                    execution_time=0.0,
                    timestamp=datetime.now(),
                    context_data={},
                    status="unsupported",
                    error_message=f"Asset type {asset_type} not supported"
                )
                analyzers.append(error_result)
        
        # Run analyses in parallel
        if analyzers:
            tasks = [
                MultiAssetOrchestrator.analyze_single_asset(analyzer, job_id, request.timeout_minutes)
                for analyzer in analyzers
                if isinstance(analyzer, AssetAnalyzerInterface)
            ]
            
            # Add any error results
            error_results = [analyzer for analyzer in analyzers if isinstance(analyzer, AssetAnalysisResult)]
            
            if tasks:
                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions in parallel execution
                final_results = []
                for i, result in enumerate(parallel_results):
                    if isinstance(result, Exception):
                        asset_name = analyzers[i].get_asset_name() if hasattr(analyzers[i], 'get_asset_name') else f"Asset_{i}"
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
                
                final_results.extend(error_results)
                return final_results
            else:
                return error_results
        
        return []

# ===================================================
# API ENDPOINTS
# ===================================================

@app.post("/analyze/all-assets", response_model=JobResponse)
async def analyze_all_assets(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis for all requested assets"""
    try:
        # Create job
        job_id = JobManager.create_job(request)
        
        # Start background processing
        background_tasks.add_task(process_analysis_job, job_id, request)
        
        # Calculate estimated completion
        estimated_completion = datetime.now() + timedelta(minutes=request.timeout_minutes + 2)
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            assets_requested=[asset.value for asset in request.assets],
            created_at=datetime.now(),
            estimated_completion=estimated_completion
        )
        
    except Exception as e:
        logger.error(f"Failed to start analysis job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.post("/analyze/asset/{asset_name}")
async def analyze_single_asset_endpoint(asset_name: str, background_tasks: BackgroundTasks):
    """Analyze a single specific asset"""
    try:
        # Validate asset name
        asset_type = AssetType(asset_name.upper())
        
        # Create single-asset request
        request = AnalysisRequest(assets=[asset_type])
        job_id = JobManager.create_job(request)
        
        # Start background processing
        background_tasks.add_task(process_analysis_job, job_id, request)
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            assets_requested=[asset_name.upper()],
            created_at=datetime.now()
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Asset '{asset_name}' is not supported")
    except Exception as e:
        logger.error(f"Failed to start single asset analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of analysis job"""
    job = JobManager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job['progress'],
        results=job['results'],
        error_message=job['error_message'],
        files_generated=job['files']
    )

@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get detailed results of completed job"""
    job = JobManager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] not in [JobStatus.COMPLETED, JobStatus.PARTIAL]:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return {
        'job_id': job_id,
        'status': job['status'],
        'results': job['results'],
        'files': job['files'],
        'created_at': job['created_at'],
        'progress': job['progress']
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'available_assets': [asset.value for asset in AssetAnalyzerFactory.get_available_assets()],
        'active_jobs': len(JobManager.jobs)
    }

@app.get("/assets")
async def list_available_assets():
    """List all available assets for analysis"""
    return {
        'available_assets': [asset.value for asset in AssetAnalyzerFactory.get_available_assets()],
        'total_assets': len(AssetAnalyzerFactory.get_available_assets())
    }

# ===================================================
# BACKGROUND PROCESSING
# ===================================================

async def process_analysis_job(job_id: str, request: AnalysisRequest):
    """Background task to process analysis job"""
    try:
        logger.info(f"Starting background processing for job {job_id}")
        JobManager.update_job_status(job_id, JobStatus.RUNNING)
        
        # Run parallel analysis
        results = await MultiAssetOrchestrator.run_parallel_analysis(request, job_id)
        
        # Generate files
        csv_file = await FileManager.generate_csv_file(results)
        json_files = await FileManager.generate_json_files(results)
        
        # Route files to downstream systems
        routing_results = await IntegrationDispatcher.route_all_files(csv_file, json_files)
        
        # Determine final status
        successful_analyses = [r for r in results if r.status == 'success']
        failed_analyses = [r for r in results if r.status != 'success']
        
        if len(successful_analyses) == len(results):
            final_status = JobStatus.COMPLETED
        elif len(successful_analyses) > 0:
            final_status = JobStatus.PARTIAL
        else:
            final_status = JobStatus.FAILED
        
        # Update job with results
        JobManager.update_job_status(
            job_id,
            final_status,
            results={
                'analysis_results': [
                    {
                        'asset': r.asset,
                        'sentiment_score': r.sentiment_score,
                        'confidence': r.confidence,
                        'status': r.status,
                        'execution_time': r.execution_time
                    } for r in results
                ],
                'routing_results': routing_results,
                'summary': {
                    'total_assets': len(results),
                    'successful': len(successful_analyses),
                    'failed': len(failed_analyses)
                }
            },
            files={
                'csv_file': csv_file,
                'json_files': json_files
            }
        )
        
        logger.info(f"Completed processing for job {job_id} with status {final_status}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        JobManager.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=str(e)
        )

# ===================================================
# STARTUP & SHUTDOWN
# ===================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Multi-Asset Sentiment Analysis Orchestrator starting up...")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Clean up old files
    await FileManager.cleanup_old_files()
    
    logger.info("Orchestrator ready to receive requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Orchestrator shutting down...")
    
    # Perform any necessary cleanup
    await FileManager.cleanup_old_files()

# ===================================================
# MAIN APPLICATION
# ===================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Multi-Asset Sentiment Analysis Orchestrator")
    
    uvicorn.run(
        "orchestrator:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
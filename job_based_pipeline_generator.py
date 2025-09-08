# JOB-BASED PIPELINE OUTPUT GENERATOR v2.0
# ========================================
# Generate job-specific outputs: 1 consolidated CSV + 4 asset context JSONs

import csv
import json
import pandas as pd
import uuid
import os
from datetime import datetime
from typing import Dict, Any, List

class JobBasedPipelineGenerator:
    """
    Job-based pipeline output generator for downstream backend models:
    - One consolidated CSV with all 4 assets (tidy format)
    - Four asset-specific context JSON files
    - Job-scoped directory structure
    """
    
    def __init__(self, base_output_dir: str = "pipeline_outputs"):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
    
    def generate_job_outputs(self, job_id: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all outputs for a specific job"""
        
        # Create job-specific directory
        job_dir = f"{self.base_output_dir}/{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        
        print(f"\nGENERATING JOB-BASED PIPELINE OUTPUTS")
        print(f"Job ID: {job_id}")
        print(f"Directory: {job_dir}")
        print("=" * 60)
        
        # Generate consolidated CSV (all 4 assets in one file)
        csv_filename = self.generate_consolidated_csv(job_id, job_dir, analysis_result)
        
        # Generate 4 asset-specific context JSON files
        context_files = self.generate_asset_context_files(job_id, job_dir, analysis_result)
        
        # Generate job manifest
        manifest_file = self.generate_job_manifest(job_id, job_dir, csv_filename, context_files, analysis_result)
        
        return {
            'job_id': job_id,
            'job_directory': job_dir,
            'consolidated_csv': csv_filename,
            'asset_context_files': context_files,
            'job_manifest': manifest_file,
            'files_generated': len(context_files) + 2,  # CSV + 4 JSONs + manifest
            'ready_for_downstream': True
        }
    
    def generate_consolidated_csv(self, job_id: str, job_dir: str, analysis_result: Dict[str, Any]) -> str:
        """Generate single consolidated CSV with all 4 assets (tidy format)"""
        
        csv_filename = f"{job_dir}/assets_data_{job_id}.csv"
        
        # Tidy CSV structure: one row per metric per asset
        csv_rows = []
        
        analysis_results = analysis_result.get('analysis_results', {})
        portfolio_summary = analysis_result.get('portfolio_summary', {})
        timestamp = analysis_result.get('timestamp', datetime.now().isoformat())
        
        # Process each asset
        for asset_name, asset_data in analysis_results.items():
            asset_status = asset_data.get('status', 'unknown')
            
            # Overall asset sentiment
            csv_rows.append({
                'job_id': job_id,
                'asset': asset_name,
                'metric_type': 'sentiment',
                'metric_name': 'overall_sentiment',
                'value': asset_data.get('sentiment', 0.0),
                'confidence': asset_data.get('confidence', 0.5),
                'status': asset_status,
                'timestamp': timestamp
            })
            
            # Asset confidence
            csv_rows.append({
                'job_id': job_id,
                'asset': asset_name,
                'metric_type': 'confidence',
                'metric_name': 'overall_confidence',
                'value': asset_data.get('confidence', 0.5),
                'confidence': 1.0,
                'status': asset_status,
                'timestamp': timestamp
            })
            
            # Execution time
            csv_rows.append({
                'job_id': job_id,
                'asset': asset_name,
                'metric_type': 'performance',
                'metric_name': 'execution_time_seconds',
                'value': asset_data.get('execution_time', 0.0),
                'confidence': 1.0,
                'status': asset_status,
                'timestamp': timestamp
            })
            
            # Component details
            component_details = asset_data.get('component_details', {})
            for component_name, component_data in component_details.items():
                if isinstance(component_data, dict):
                    # Component sentiment
                    csv_rows.append({
                        'job_id': job_id,
                        'asset': asset_name,
                        'metric_type': 'component_sentiment',
                        'metric_name': f'{component_name}_sentiment',
                        'value': component_data.get('sentiment', 0.0),
                        'confidence': component_data.get('confidence', 0.5),
                        'status': asset_status,
                        'timestamp': timestamp
                    })
                    
                    # Component weight
                    if 'framework_weight' in component_data:
                        csv_rows.append({
                            'job_id': job_id,
                            'asset': asset_name,
                            'metric_type': 'component_weight',
                            'metric_name': f'{component_name}_weight',
                            'value': component_data.get('framework_weight', 0.0),
                            'confidence': 1.0,
                            'status': asset_status,
                            'timestamp': timestamp
                        })
                    
                    # Component metadata (flattened numerical values)
                    metadata = component_data.get('metadata', {})
                    for key, value in self._flatten_dict(metadata).items():
                        if isinstance(value, (int, float)):
                            csv_rows.append({
                                'job_id': job_id,
                                'asset': asset_name,
                                'metric_type': 'component_metric',
                                'metric_name': f'{component_name}_{key}',
                                'value': value,
                                'confidence': component_data.get('confidence', 0.5),
                                'status': asset_status,
                                'timestamp': timestamp
                            })
        
        # Portfolio-level metrics
        csv_rows.append({
            'job_id': job_id,
            'asset': 'PORTFOLIO',
            'metric_type': 'portfolio_sentiment',
            'metric_name': 'overall_portfolio_sentiment',
            'value': portfolio_summary.get('portfolio_sentiment', 0.0),
            'confidence': portfolio_summary.get('portfolio_confidence', 0.5),
            'status': 'success',
            'timestamp': timestamp
        })
        
        csv_rows.append({
            'job_id': job_id,
            'asset': 'PORTFOLIO',
            'metric_type': 'portfolio_metrics',
            'metric_name': 'successful_assets',
            'value': portfolio_summary.get('successful_assets', 0),
            'confidence': 1.0,
            'status': 'success',
            'timestamp': timestamp
        })
        
        # Write consolidated CSV
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_filename, index=False)
        
        print(f"  Generated consolidated CSV: {csv_filename}")
        print(f"   Rows: {len(csv_rows)}, Assets: {len(analysis_results) + 1} (including portfolio)")
        
        return csv_filename
    
    def generate_asset_context_files(self, job_id: str, job_dir: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate 4 asset-specific context JSON files"""
        
        context_files = []
        analysis_results = analysis_result.get('analysis_results', {})
        
        # Define all 4 assets (ensure we create files even if analysis failed)
        all_assets = ['NIFTY50', 'GOLD', 'BITCOIN', 'REIT']
        
        for asset_name in all_assets:
            asset_data = analysis_results.get(asset_name, {})
            
            context_filename = f"{job_dir}/{asset_name}_context_{job_id}.json"
            
            # Create comprehensive context data
            context_data = {
                'job_metadata': {
                    'job_id': job_id,
                    'asset_name': asset_name,
                    'generated_at': datetime.now().isoformat(),
                    'belongs_to_analysis': analysis_result.get('timestamp', datetime.now().isoformat())
                },
                
                'asset_metadata': {
                    'asset_name': asset_name,
                    'asset_type': asset_data.get('asset_type', asset_name),
                    'analysis_timestamp': asset_data.get('timestamp', datetime.now().isoformat()),
                    'data_source': asset_data.get('data_source', 'unknown'),
                    'execution_time': asset_data.get('execution_time', 0.0),
                    'status': asset_data.get('status', 'unknown')
                },
                
                'sentiment_analysis': {
                    'overall_sentiment': asset_data.get('sentiment', 0.0),
                    'confidence_level': asset_data.get('confidence', 0.5),
                    'sentiment_interpretation': self._get_sentiment_interpretation(asset_data.get('sentiment', 0.0)),
                    'confidence_grade': self._get_confidence_grade(asset_data.get('confidence', 0.5)),
                    'analysis_quality': asset_data.get('status', 'unknown')
                },
                
                'component_breakdown': {},
                
                'market_context': asset_data.get('market_context', {}),
                
                'executive_summary': asset_data.get('executive_summary', {}),
                
                'risk_assessment': {
                    'data_quality': asset_data.get('status', 'unknown'),
                    'confidence_score': asset_data.get('confidence', 0.5),
                    'execution_success': asset_data.get('status') == 'success',
                    'error_message': asset_data.get('error', None)
                },
                
                'pipeline_metadata': {
                    'job_id': job_id,
                    'format_version': '2.0',
                    'downstream_compatible': True,
                    'context_file_type': 'asset_specific',
                    'related_csv': f"assets_data_{job_id}.csv"
                }
            }
            
            # Process component details
            # FIXED: Process component details with correct field names
            component_details = asset_data.get('component_details', {})
            for component_name, component_data in component_details.items():
                if isinstance(component_data, dict):
                    context_data['component_breakdown'][component_name] = {
                        'sentiment': component_data.get('component_sentiment', 0.0),  # ← FIXED
                        'confidence': component_data.get('component_confidence', 0.5),  # ← FIXED
                        'weight': component_data.get('component_weight', component_data.get('framework_weight', 0.0)),
                        'contribution': component_data.get('weighted_contribution', 0.0),
                        'description': component_data.get('interpretation', component_data.get('description', '')),
                        'key_metrics': self._extract_key_metrics(component_data.get('metadata', {})),
                        'status': component_data.get('status', 'success')
                    }

            # Save context JSON
            with open(context_filename, 'w') as f:
                json.dump(context_data, f, indent=2, default=str)
            
            context_files.append(context_filename)
            
            print(f"Generated {asset_name} context: {context_filename}")
        
        return context_files
    
    def generate_job_manifest(self, job_id: str, job_dir: str, csv_file: str, context_files: List[str], analysis_result: Dict[str, Any]) -> str:
        """Generate job manifest file for easy discovery"""
        
        manifest_filename = f"{job_dir}/job_manifest_{job_id}.json"
        
        manifest_data = {
            'job_id': job_id,
            'generated_at': datetime.now().isoformat(),
            'analysis_timestamp': analysis_result.get('timestamp', datetime.now().isoformat()),
            
            'files': {
                'consolidated_csv': {
                    'filename': os.path.basename(csv_file),
                    'full_path': csv_file,
                    'type': 'consolidated_data',
                    'description': 'Tidy CSV with all 4 assets for Backend Model A',
                    'format': 'CSV',
                    'columns': ['job_id', 'asset', 'metric_type', 'metric_name', 'value', 'confidence', 'status', 'timestamp']
                },
                'asset_contexts': []
            },
            
            'analysis_summary': {
                'total_assets': len(analysis_result.get('analysis_results', {})),
                'successful_assets': analysis_result.get('portfolio_summary', {}).get('successful_assets', 0),
                'portfolio_sentiment': analysis_result.get('portfolio_summary', {}).get('portfolio_sentiment', 0.0),
                'portfolio_confidence': analysis_result.get('portfolio_summary', {}).get('portfolio_confidence', 0.5),
                'execution_time': analysis_result.get('execution_time_seconds', 0.0)
            },
            
            'downstream_access': {
                'backend_model_a': {
                    'file_type': 'CSV',
                    'access_method': f'GET /pipeline-outputs/{job_id}/csv',
                    'file': os.path.basename(csv_file)
                },
                'backend_model_b': {
                    'file_type': 'Context JSON',
                    'access_method': f'GET /pipeline-outputs/{job_id}/context/{{asset}}',
                    'available_assets': ['NIFTY50', 'GOLD', 'BITCOIN', 'REIT']
                }
            }
        }
        
        # Add context file details
        for context_file in context_files:
            asset_name = os.path.basename(context_file).split('_context_')[0]
            manifest_data['files']['asset_contexts'].append({
                'asset': asset_name,
                'filename': os.path.basename(context_file),
                'full_path': context_file,
                'type': 'asset_context',
                'description': f'{asset_name} context data for Backend Model B'
            })
        
        # Save manifest
        with open(manifest_filename, 'w') as f:
            json.dump(manifest_data, f, indent=2, default=str)
        
        print(f"📋 Generated job manifest: {manifest_filename}")
        
        return manifest_filename
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _extract_key_metrics(self, metadata: Dict) -> Dict:
        """Extract key numerical metrics from metadata"""
        key_metrics = {}
        for key, value in self._flatten_dict(metadata).items():
            if isinstance(value, (int, float)) and not key.endswith('_time'):
                key_metrics[key] = value
        return key_metrics
    
    def _get_sentiment_interpretation(self, sentiment: float) -> str:
        """Convert sentiment score to interpretation"""
        if sentiment > 0.6:
            return "STRONG_BULLISH"
        elif sentiment > 0.3:
            return "BULLISH"
        elif sentiment > 0.1:
            return "MILD_BULLISH"
        elif sentiment > -0.1:
            return "NEUTRAL"
        elif sentiment > -0.3:
            return "MILD_BEARISH"
        elif sentiment > -0.6:
            return "BEARISH"
        else:
            return "STRONG_BEARISH"
    
    def _get_confidence_grade(self, confidence: float) -> str:
        """Convert confidence score to grade"""
        if confidence > 0.8:
            return "A"
        elif confidence > 0.6:
            return "B"
        elif confidence > 0.4:
            return "C"
        else:
            return "D"

# Main job-based pipeline function
async def generate_job_based_outputs(job_id: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate job-based pipeline outputs with tracking"""
    
    generator = JobBasedPipelineGenerator()
    pipeline_outputs = generator.generate_job_outputs(job_id, analysis_result)
    
    print(f"\n✅ JOB-BASED PIPELINE OUTPUTS GENERATED")
    print(f"📋 Job ID: {job_id}")
    print(f"📊 Files Generated: {pipeline_outputs['files_generated']}")
    print(f"📁 Location: {pipeline_outputs['job_directory']}")
    print(f"🔗 Ready for downstream processing: {pipeline_outputs['ready_for_downstream']}")
    
    return pipeline_outputs

# Test the job-based pipeline
if __name__ == "__main__":
    import asyncio
    
    # Example usage
    sample_job_id = str(uuid.uuid4())
    sample_result = {
        'analysis_results': {
            'BITCOIN': {
                'asset_type': 'BITCOIN',
                'sentiment': -0.014,
                'confidence': 0.52,
                'status': 'success',
                'execution_time': 9.6,
                'component_details': {
                    'micro_momentum': {
                        'sentiment': 0.031,
                        'confidence': 0.72,
                        'framework_weight': 0.3,
                        'metadata': {'roc_1h': 0.005, 'rsi': 50.77}
                    }
                }
            },
            'REIT': {
                'asset_type': 'REIT',
                'sentiment': -0.599,
                'confidence': 0.8,
                'status': 'success'
            }
        },
        'portfolio_summary': {
            'portfolio_sentiment': -0.204,
            'portfolio_confidence': 0.607,
            'successful_assets': 2
        },
        'timestamp': datetime.now().isoformat()
    }
    
    async def test_job_pipeline():
        result = await generate_job_based_outputs(sample_job_id, sample_result)
        print(f"\n🧪 Test completed:")
        print(f"📋 Job ID: {result['job_id']}")
        print(f"📊 CSV: {result['consolidated_csv']}")
        print(f"📁 Context files: {len(result['asset_context_files'])}")
    
    asyncio.run(test_job_pipeline())
"""
Unified Pipeline API

Provides a single entry point for the complete data pipeline:
1. The Sniff - CSV scanning and column analysis
2. The Mapping - Column identity confirmation
3. The Pipeline - Matrix generation
4. The Training - Async model training

Path Structure (Three-level decoupling):
    - Shared matrices: workspace/{user_id}/{dataset_name}/
    - Model outputs:   result/{user_id}/{dataset_name}/{model_name}/{timestamp}/

Usage:
    from data_pipeline import PipelineAPI
    
    # Initialize with user_id
    api = PipelineAPI(user_id='researcher_001')
    
    # Phase 1: Scan CSV
    scan_result = api.scan_csv('/path/to/data.csv')
    
    # Phase 2: Confirm mapping
    mapping = {
        'text_column': 'content',
        'time_column': 'year',
        'covariate_columns': ['category', 'source']
    }
    validation = api.set_mapping(mapping)
    
    # Phase 3: Generate matrices (to workspace/{user_id}/{dataset_name}/)
    pipeline_result = api.run_pipeline()
    
    # Phase 4: Start training (outputs to result/{user_id}/{dataset_name}/{model}/)
    session = api.start_training(models=['lda', 'etm', 'dtm'])
    
    # Check progress
    status = api.get_training_status()
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_WORKSPACE, BASE_RESULT, LOGS_DIR,
    get_workspace_path, get_result_path, ensure_dir
)

from .csv_scanner import CSVScanner
from .column_mapper import ColumnMapper, MappingConfig
from .matrix_pipeline import MatrixPipeline
from .async_trainer import AsyncTrainer, TrainingSession


class PipelineAPI:
    """
    Unified API for the complete data pipeline.
    
    Implements three-level path decoupling:
        user_id -> dataset_name -> model_name
    """
    
    def __init__(
        self,
        user_id: str = "default_user",
        vocab_size: int = 5000,
        num_topics: int = 20,
        epochs: int = 100,
        gpu: int = 0,
        language: str = 'auto',
        force: bool = False,
        # Legacy parameter for backward compatibility
        output_base_dir: str = None,
    ):
        """
        Initialize Pipeline API.
        
        Args:
            user_id: User identifier for path isolation
            vocab_size: Maximum vocabulary size
            num_topics: Number of topics for training
            epochs: Training epochs for neural models
            gpu: GPU device ID
            language: Language for text processing and visualization
            force: Force overwrite existing matrices
            output_base_dir: (Legacy) Base directory for outputs
        """
        self.user_id = user_id
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.epochs = epochs
        self.gpu = gpu
        self.language = language
        self.force = force
        
        # Legacy support
        self.output_base_dir = Path(output_base_dir) if output_base_dir else None
        
        # Pipeline state
        self.csv_path: Optional[str] = None
        self.dataset_name: Optional[str] = None
        self.scanner: Optional[CSVScanner] = None
        self.mapper: Optional[ColumnMapper] = None
        self.pipeline: Optional[MatrixPipeline] = None
        self.trainer: Optional[AsyncTrainer] = None
        
        # Results
        self.scan_result: Optional[Dict] = None
        self.pipeline_result: Optional[Dict] = None
        self.session_id: Optional[str] = None
    
    # ==========================================================================
    # Path helpers
    # ==========================================================================
    
    def get_workspace_dir(self) -> Path:
        """Get workspace directory for shared matrices."""
        if self.output_base_dir:
            # Legacy mode: use output_base_dir
            return self.output_base_dir / self.dataset_name / 'data'
        return get_workspace_path(self.user_id, self.dataset_name or "")
    
    def get_result_dir(self, model_name: str, timestamp: str = "") -> Path:
        """Get result directory for model outputs."""
        if self.output_base_dir:
            # Legacy mode
            return self.output_base_dir / self.dataset_name / 'models' / model_name / timestamp
        return get_result_path(self.user_id, self.dataset_name or "", model_name, timestamp)
    
    # ========== Phase 1: The Sniff ==========
    
    def scan_csv(self, file_path: str, sample_rows: int = 100) -> Dict[str, Any]:
        """
        Phase 1: Scan CSV file and analyze columns.
        
        Args:
            file_path: Path to CSV file
            sample_rows: Number of rows to sample for analysis
            
        Returns:
            Scan result with column analysis and recommendations
        """
        self.csv_path = file_path
        self.dataset_name = Path(file_path).stem
        
        self.scanner = CSVScanner(sample_rows=sample_rows)
        self.scan_result = self.scanner.scan(file_path)
        
        # Initialize mapper with scan result
        self.mapper = ColumnMapper(self.scan_result)
        
        return self.scan_result
    
    def get_scan_result(self) -> Optional[Dict]:
        """Get the last scan result."""
        return self.scan_result
    
    # ========== Phase 2: The Mapping ==========
    
    def get_mapping_table(self) -> List[Dict]:
        """
        Get mapping table for frontend display.
        
        Returns:
            List of column info with suggested and current mappings
        """
        if self.mapper is None:
            raise RuntimeError("No CSV scanned. Call scan_csv() first.")
        return self.mapper.get_mapping_table()
    
    def set_mapping(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set column mapping from user selection.
        
        Args:
            mapping: {
                'text_column': str (required),
                'time_column': str or None (optional, enables DTM),
                'covariate_columns': List[str] (optional, enables STM),
                'id_column': str or None (optional),
                'label_column': str or None (optional),
            }
            
        Returns:
            Validation result with available models
        """
        if self.mapper is None:
            raise RuntimeError("No CSV scanned. Call scan_csv() first.")
        
        return self.mapper.update_mapping(mapping)
    
    def get_mapping(self) -> Optional[Dict]:
        """Get current mapping configuration."""
        if self.mapper is None:
            return None
        config = self.mapper.get_config()
        return config.to_dict() if config else None
    
    def validate_mapping(self) -> Dict[str, Any]:
        """Validate current mapping."""
        if self.mapper is None:
            return {'valid': False, 'errors': ['No CSV scanned']}
        return self.mapper.validate()
    
    # ========== Phase 3: The Pipeline ==========
    
    def run_pipeline(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 3: Generate training matrices from mapped data.
        
        Outputs to: workspace/{user_id}/{dataset_name}/
        
        Args:
            output_dir: Output directory (auto-generated if None)
            
        Returns:
            Pipeline result with file paths and statistics
        """
        if self.mapper is None:
            raise RuntimeError("No CSV scanned. Call scan_csv() first.")
        
        validation = self.mapper.validate()
        if not validation['valid']:
            raise ValueError(f"Invalid mapping: {validation['errors']}")
        
        # Generate session ID
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine output directory using new path structure
        if output_dir is None:
            output_dir = self.get_workspace_dir()
        
        output_dir = Path(output_dir)
        
        # Check if matrices already exist
        bow_exists = (output_dir / 'bow_matrix.npz').exists() or (output_dir / 'bow_matrix.npy').exists()
        
        if bow_exists and not self.force:
            print(f"[INFO] Matrices already exist in {output_dir}")
            print(f"[INFO] Use force=True to overwrite")
            # Load existing config
            config_path = output_dir / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                self.pipeline_result = {
                    'output_dir': str(output_dir),
                    'files': {
                        'bow_matrix': str(output_dir / 'bow_matrix.npz'),
                        'vocab': str(output_dir / 'vocab.json'),
                        'config': str(config_path),
                    },
                    'stats': existing_config.get('stats', {}),
                    'skipped': True,
                }
                return self.pipeline_result
        
        # Create output directory
        ensure_dir(output_dir)
        
        # Create and run pipeline
        mapping_config = self.mapper.get_config()
        
        self.pipeline = MatrixPipeline(
            mapping_config=mapping_config,
            output_dir=str(output_dir),
            vocab_size=self.vocab_size,
            language=self.language,
        )
        
        self.pipeline_result = self.pipeline.run(self.csv_path)
        
        return self.pipeline_result
    
    def get_pipeline_result(self) -> Optional[Dict]:
        """Get the last pipeline result."""
        return self.pipeline_result
    
    # ========== Phase 4: The Training ==========
    
    def get_available_models(self) -> Dict[str, bool]:
        """
        Get which models are available based on current data.
        
        Returns:
            Dict mapping model name to availability
        """
        if self.pipeline_result is None:
            # Use mapping config to determine availability
            if self.mapper:
                return self.mapper.get_config().get_available_models()
            return {}
        
        return self.pipeline_result.get('available_models', {})
    
    def start_training(
        self,
        models: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        on_progress: Optional[Callable[[Dict], None]] = None,
        on_complete: Optional[Callable[[Dict], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ) -> TrainingSession:
        """
        Phase 4: Start asynchronous model training.
        
        Args:
            models: List of models to train (None for all available)
            output_dir: Output directory for models
            on_progress: Callback for progress updates
            on_complete: Callback when training completes
            on_error: Callback for errors
            
        Returns:
            Training session
        """
        if self.pipeline_result is None:
            raise RuntimeError("Pipeline not run. Call run_pipeline() first.")
        
        data_dir = self.pipeline_result['output_dir']
        
        if output_dir is None:
            output_dir = self.output_base_dir / self.dataset_name / 'models'
        
        self.trainer = AsyncTrainer(
            data_dir=data_dir,
            output_dir=str(output_dir),
            dataset_name=self.dataset_name,
            num_topics=self.num_topics,
            epochs=self.epochs,
            gpu=self.gpu,
            language=self.language,
        )
        
        # Set callbacks
        self.trainer.on_progress = on_progress
        self.trainer.on_complete = on_complete
        self.trainer.on_error = on_error
        
        return self.trainer.start_training(models)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        if self.trainer is None:
            return {'status': 'not_started'}
        return self.trainer.get_status()
    
    def stop_training(self):
        """Stop training gracefully."""
        if self.trainer:
            self.trainer.stop_training()
    
    # ========== Utility Methods ==========
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get complete pipeline status."""
        return {
            'phase': self._get_current_phase(),
            'csv_path': self.csv_path,
            'dataset_name': self.dataset_name,
            'scan_result': self.scan_result is not None,
            'mapping': self.get_mapping(),
            'mapping_valid': self.validate_mapping() if self.mapper else None,
            'pipeline_result': self.pipeline_result is not None,
            'available_models': self.get_available_models(),
            'training_status': self.get_training_status(),
        }
    
    def _get_current_phase(self) -> str:
        """Get current pipeline phase."""
        if self.trainer and self.trainer.session:
            return 'training'
        if self.pipeline_result:
            return 'ready_to_train'
        if self.mapper:
            return 'mapping'
        if self.scan_result:
            return 'scanned'
        return 'initial'
    
    def reset(self):
        """Reset pipeline state."""
        self.csv_path = None
        self.dataset_name = None
        self.scanner = None
        self.mapper = None
        self.pipeline = None
        self.trainer = None
        self.scan_result = None
        self.pipeline_result = None
        self.session_id = None
    
    def to_json(self) -> str:
        """Export current state as JSON."""
        return json.dumps(self.get_full_status(), ensure_ascii=False, indent=2)


# Convenience functions

def create_pipeline(**kwargs) -> PipelineAPI:
    """Create a PipelineAPI instance."""
    return PipelineAPI(**kwargs)


def quick_run(
    csv_path: str,
    text_column: str,
    time_column: Optional[str] = None,
    covariate_columns: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick run: scan, map, pipeline, and train in one call.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        time_column: Name of time column (optional)
        covariate_columns: List of covariate columns (optional)
        models: List of models to train (optional)
        **kwargs: Additional arguments for PipelineAPI
        
    Returns:
        Training session result
    """
    api = PipelineAPI(**kwargs)
    
    # Phase 1: Scan
    api.scan_csv(csv_path)
    
    # Phase 2: Map
    mapping = {
        'text_column': text_column,
        'time_column': time_column,
        'covariate_columns': covariate_columns or [],
    }
    api.set_mapping(mapping)
    
    # Phase 3: Pipeline
    api.run_pipeline()
    
    # Phase 4: Train (blocking)
    session = api.start_training(models)
    
    # Wait for completion
    import time
    while api.get_training_status().get('status') == 'running':
        time.sleep(5)
    
    return api.get_training_status()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pipeline_api.py <csv_file> <text_column> [time_column] [covariate_columns...]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    text_column = sys.argv[2]
    time_column = sys.argv[3] if len(sys.argv) > 3 else None
    covariate_columns = sys.argv[4:] if len(sys.argv) > 4 else []
    
    result = quick_run(
        csv_path=csv_path,
        text_column=text_column,
        time_column=time_column,
        covariate_columns=covariate_columns,
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

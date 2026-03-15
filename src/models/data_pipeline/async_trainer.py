"""
Phase 4: The Training - Async Model Training

Manages asynchronous training of all 11 baseline models:
- Traditional: LDA, HDP, BTM, STM
- Neural: NVDM, GSM, ProdLDA, ETM, CTM, DTM
- Clustering: BERTopic

Provides progress tracking and status updates for frontend.
"""

import os
import json
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
import traceback


class TrainingStatus(Enum):
    """Training status types."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'


@dataclass
class ModelProgress:
    """Progress info for a single model."""
    model_name: str
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0  # 0-100
    message: str = ''
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    output_dir: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'status': self.status.value,
            'progress': self.progress,
            'message': self.message,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'elapsed_time': (self.end_time or time.time()) - self.start_time if self.start_time else 0,
            'metrics': self.metrics,
            'error': self.error,
            'output_dir': self.output_dir,
        }


@dataclass
class TrainingSession:
    """Complete training session info."""
    session_id: str
    dataset_name: str
    data_dir: str
    output_dir: str
    models: List[str]
    status: TrainingStatus = TrainingStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    model_progress: Dict[str, ModelProgress] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'dataset_name': self.dataset_name,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'models': self.models,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'elapsed_time': (self.end_time or time.time()) - self.start_time if self.start_time else 0,
            'model_progress': {k: v.to_dict() for k, v in self.model_progress.items()},
            'overall_progress': self.get_overall_progress(),
            'config': self.config,
        }
    
    def get_overall_progress(self) -> float:
        """Calculate overall progress percentage."""
        if not self.model_progress:
            return 0.0
        total = sum(p.progress for p in self.model_progress.values())
        return total / len(self.model_progress)


class AsyncTrainer:
    """Manages asynchronous training of multiple models."""
    
    # All available models
    ALL_MODELS = [
        'lda', 'hdp', 'btm', 'stm',  # Traditional
        'nvdm', 'gsm', 'prodlda', 'etm', 'ctm', 'dtm',  # Neural
        'bertopic'  # Clustering
    ]
    
    # Models requiring special data
    MODELS_REQUIRE_TIME = ['dtm']
    MODELS_REQUIRE_COVARIATES = ['stm']
    MODELS_REQUIRE_SBERT = ['ctm', 'bertopic']
    MODELS_REQUIRE_WORD2VEC = ['etm', 'dtm']
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        dataset_name: str = 'custom',
        num_topics: int = 20,
        epochs: int = 100,
        batch_size: int = 64,
        gpu: int = 0,
        language: str = 'auto'
    ):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory containing prepared matrices
            output_dir: Output directory for model results
            dataset_name: Name of the dataset
            num_topics: Number of topics
            epochs: Training epochs for neural models
            batch_size: Batch size
            gpu: GPU device ID
            language: Language for visualization
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.num_topics = num_topics
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu
        self.language = language
        
        # Session management
        self.session: Optional[TrainingSession] = None
        self.training_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.progress_queue = queue.Queue()
        
        # Callbacks
        self.on_progress: Optional[Callable[[Dict], None]] = None
        self.on_complete: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None
    
    def get_available_models(self) -> Dict[str, bool]:
        """Check which models can run based on available data."""
        available = {}
        
        # Check for required files
        has_bow = (self.data_dir / 'bow_matrix.npz').exists()
        has_vocab = (self.data_dir / 'vocab.json').exists()
        has_time = (self.data_dir / 'time_slices.json').exists()
        has_covariates = (self.data_dir / 'covariates.npy').exists()
        has_sbert = (self.data_dir / 'sbert_embeddings.npy').exists()
        has_word2vec = (self.data_dir / 'word2vec_embeddings.npy').exists()
        
        for model in self.ALL_MODELS:
            # Basic requirement: BOW and vocab
            can_run = has_bow and has_vocab
            
            # Special requirements
            if model in self.MODELS_REQUIRE_TIME:
                can_run = can_run and has_time
            if model in self.MODELS_REQUIRE_COVARIATES:
                can_run = can_run and has_covariates
            if model in self.MODELS_REQUIRE_SBERT:
                can_run = can_run and has_sbert
            if model in self.MODELS_REQUIRE_WORD2VEC:
                can_run = can_run and has_word2vec
            
            available[model] = can_run
        
        return available
    
    def create_session(self, models: Optional[List[str]] = None) -> TrainingSession:
        """
        Create a new training session.
        
        Args:
            models: List of models to train (None for all available)
            
        Returns:
            Training session
        """
        available = self.get_available_models()
        
        if models is None:
            models = [m for m, can_run in available.items() if can_run]
        else:
            # Filter to only available models
            models = [m for m in models if available.get(m, False)]
        
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.session = TrainingSession(
            session_id=session_id,
            dataset_name=self.dataset_name,
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            models=models,
            config={
                'num_topics': self.num_topics,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'gpu': self.gpu,
                'language': self.language,
            }
        )
        
        # Initialize model progress
        for model in models:
            self.session.model_progress[model] = ModelProgress(model_name=model)
        
        return self.session
    
    def start_training(self, models: Optional[List[str]] = None) -> TrainingSession:
        """
        Start asynchronous training.
        
        Args:
            models: List of models to train
            
        Returns:
            Training session
        """
        if self.training_thread and self.training_thread.is_alive():
            raise RuntimeError("Training already in progress")
        
        self.create_session(models)
        self.stop_flag.clear()
        
        self.training_thread = threading.Thread(
            target=self._training_worker,
            daemon=True
        )
        self.training_thread.start()
        
        return self.session
    
    def stop_training(self):
        """Stop training gracefully."""
        self.stop_flag.set()
        if self.training_thread:
            self.training_thread.join(timeout=30)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        if self.session is None:
            return {'status': 'no_session'}
        return self.session.to_dict()
    
    def _training_worker(self):
        """Worker thread for training."""
        self.session.status = TrainingStatus.RUNNING
        self.session.start_time = time.time()
        
        try:
            for model_name in self.session.models:
                if self.stop_flag.is_set():
                    break
                
                self._train_single_model(model_name)
            
            self.session.status = TrainingStatus.COMPLETED
            
        except Exception as e:
            self.session.status = TrainingStatus.FAILED
            if self.on_error:
                self.on_error('session', str(e))
        
        finally:
            self.session.end_time = time.time()
            if self.on_complete:
                self.on_complete(self.session.to_dict())
    
    def _train_single_model(self, model_name: str):
        """Train a single model."""
        progress = self.session.model_progress[model_name]
        progress.status = TrainingStatus.RUNNING
        progress.start_time = time.time()
        progress.message = f"Training {model_name.upper()}..."
        
        self._emit_progress()
        
        try:
            # Import trainer
            from model.baseline_trainer import BaselineTrainer
            
            # Create model output directory
            model_output_dir = self.output_dir / model_name / f"exp_{self.session.session_id}"
            model_output_dir.mkdir(parents=True, exist_ok=True)
            progress.output_dir = str(model_output_dir)
            
            # Initialize trainer
            trainer = BaselineTrainer(
                dataset=self.dataset_name,
                num_topics=self.num_topics,
                output_dir=str(model_output_dir),
                data_exp_dir=str(self.data_dir),
            )
            
            # Load data
            progress.progress = 10
            progress.message = "Loading data..."
            self._emit_progress()
            
            trainer.load_preprocessed_data()
            
            # Train model
            progress.progress = 20
            progress.message = f"Training {model_name.upper()}..."
            self._emit_progress()
            
            train_result = self._call_train_method(trainer, model_name)
            
            # Evaluate
            progress.progress = 70
            progress.message = "Evaluating..."
            self._emit_progress()
            
            if train_result:
                metrics = self._evaluate_model(trainer, model_name, model_output_dir)
                progress.metrics = metrics
            
            # Visualize
            progress.progress = 85
            progress.message = "Generating visualizations..."
            self._emit_progress()
            
            self._visualize_model(model_name, model_output_dir)
            
            # Complete
            progress.progress = 100
            progress.status = TrainingStatus.COMPLETED
            progress.message = "Completed"
            progress.end_time = time.time()
            
        except Exception as e:
            progress.status = TrainingStatus.FAILED
            progress.error = str(e)
            progress.message = f"Failed: {str(e)}"
            progress.end_time = time.time()
            
            if self.on_error:
                self.on_error(model_name, str(e))
            
            traceback.print_exc()
        
        self._emit_progress()
    
    def _call_train_method(self, trainer, model_name: str) -> Dict:
        """Call appropriate training method for model."""
        import numpy as np
        
        if model_name == 'lda':
            return trainer.train_lda(max_iter=100)
        elif model_name == 'hdp':
            return trainer.train_hdp(max_topics=150)
        elif model_name == 'btm':
            return trainer.train_btm(n_iter=100)
        elif model_name == 'stm':
            covariates = None
            cov_path = self.data_dir / 'covariates.npy'
            if cov_path.exists():
                covariates = np.load(cov_path)
            return trainer.train_stm(max_iter=100, covariates=covariates)
        elif model_name == 'nvdm':
            return trainer.train_nvdm(epochs=self.epochs, batch_size=self.batch_size)
        elif model_name == 'gsm':
            return trainer.train_gsm(epochs=self.epochs, batch_size=self.batch_size)
        elif model_name == 'prodlda':
            return trainer.train_prodlda(epochs=self.epochs, batch_size=self.batch_size)
        elif model_name == 'etm':
            return trainer.train_etm(epochs=self.epochs, batch_size=self.batch_size)
        elif model_name == 'ctm':
            return trainer.train_ctm(epochs=self.epochs, batch_size=self.batch_size)
        elif model_name == 'dtm':
            return trainer.train_dtm(epochs=self.epochs, batch_size=self.batch_size)
        elif model_name == 'bertopic':
            return trainer.train_bertopic()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _evaluate_model(self, trainer, model_name: str, output_dir: Path) -> Dict:
        """Evaluate trained model."""
        try:
            from evaluation.unified_evaluator import UnifiedEvaluator
            
            evaluator = UnifiedEvaluator(
                theta=trainer.theta if hasattr(trainer, 'theta') else None,
                beta=trainer.beta if hasattr(trainer, 'beta') else None,
                vocab=trainer.vocab,
                bow_matrix=trainer.bow_matrix,
            )
            
            metrics = evaluator.compute_all_metrics()
            
            # Save metrics
            metrics_path = output_dir / f'metrics_k{self.num_topics}.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return {
                'td': metrics.get('topic_diversity_td', 0),
                'npmi': metrics.get('topic_coherence_npmi_avg', 0),
            }
        except Exception as e:
            print(f"Evaluation failed for {model_name}: {e}")
            return {}
    
    def _visualize_model(self, model_name: str, output_dir: Path):
        """Generate visualizations for model."""
        try:
            from visualization.run_visualization import run_baseline_visualization
            
            run_baseline_visualization(
                result_dir=str(output_dir),
                dataset=self.dataset_name,
                model=model_name,
                num_topics=self.num_topics,
                language=self.language,
            )
        except Exception as e:
            print(f"Visualization failed for {model_name}: {e}")
    
    def _emit_progress(self):
        """Emit progress update."""
        if self.on_progress and self.session:
            self.on_progress(self.session.to_dict())


def create_trainer(
    data_dir: str,
    output_dir: str,
    **kwargs
) -> AsyncTrainer:
    """Create an AsyncTrainer instance."""
    return AsyncTrainer(data_dir=data_dir, output_dir=output_dir, **kwargs)

"""
ETM Configuration Management

Centralized configuration for all ETM pipeline components.
"""

import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


# Base paths - support environment variables for flexible deployment
# Environment variables:
#   ETM_BASE_DIR: Base directory (default: /root/autodl-tmp)
#   ETM_DATA_DIR: Data directory (default: {BASE_DIR}/data)
#   ETM_RESULT_DIR: Result directory (default: {BASE_DIR}/result)
#   ETM_EMBEDDING_MODELS_DIR: Embedding models directory (default: {BASE_DIR}/embedding_models)
BASE_DIR = Path(os.environ.get("ETM_BASE_DIR", "/root/autodl-tmp"))
DATA_DIR = Path(os.environ.get("ETM_DATA_DIR", str(BASE_DIR / "data")))
EMBEDDING_DIR = Path(os.environ.get("ETM_EMBEDDING_DIR", str(BASE_DIR / "embedding")))
ETM_DIR = Path(os.environ.get("ETM_CODE_DIR", str(BASE_DIR / "ETM")))
# Unified embedding model storage path
EMBEDDING_MODELS_DIR = Path(os.environ.get("ETM_EMBEDDING_MODELS_DIR", str(BASE_DIR / "embedding_models")))
QWEN_MODEL_PATH = EMBEDDING_MODELS_DIR / "qwen3_embedding_0.6B"  # Default: 0.6B
RESULT_DIR = Path(os.environ.get("ETM_RESULT_DIR", str(BASE_DIR / "result")))

# Embedding model paths and dimension mapping
QWEN_MODEL_PATHS = {
    '0.6B': str(EMBEDDING_MODELS_DIR / "qwen3_embedding_0.6B"),
    '4B': str(EMBEDDING_MODELS_DIR / "qwen3_embedding_4B"),
    '8B': str(EMBEDDING_MODELS_DIR / "qwen3_embedding_8B"),
}

EMBEDDING_DIMS = {
    '0.6B': 1024,
    '4B': 2560,
    '8B': 4096,
}


def get_qwen_model_path(model_size: str) -> str:
    """Get Qwen model path based on model_size"""
    if model_size not in QWEN_MODEL_PATHS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(QWEN_MODEL_PATHS.keys())}")
    return QWEN_MODEL_PATHS[model_size]


def get_embedding_dim(model_size: str) -> int:
    """Get embedding dimension based on model_size"""
    return EMBEDDING_DIMS.get(model_size, 1024)

# Dataset-specific configurations
# vocab_size should be proportional to dataset size but not too large
# Rule of thumb: vocab_size ~ min(sqrt(n_docs) * 50, 10000)
DATASET_CONFIGS = {
    "socialTwitter": {
        "vocab_size": 5000,      # ~40K docs
        "num_topics": 20,
        "min_doc_freq": 5,
        "language": "multi",     # Spanish + some English
    },
    "hatespeech": {
        "vocab_size": 8000,      # ~437K docs
        "num_topics": 20,
        "min_doc_freq": 10,
        "language": "english",
    },
    "mental_health": {
        "vocab_size": 10000,     # ~1M docs
        "num_topics": 30,
        "min_doc_freq": 20,
        "language": "english",
    },
    "FCPB": {
        "vocab_size": 8000,      # ~854K docs
        "num_topics": 25,
        "min_doc_freq": 15,
        "language": "english",
    },
    "germanCoal": {
        "vocab_size": 3000,      # ~9K docs (smaller dataset)
        "num_topics": 15,
        "min_doc_freq": 3,
        "language": "german",
    },
    "edu_data": {
        "vocab_size": 5000,      # ~857 docs (education policy documents)
        "num_topics": 20,
        "min_doc_freq": 3,
        "language": "chinese",
        "has_timestamp": True,   # DTM specific
    },
}


@dataclass
class DataConfig:
    """Data configuration"""
    dataset: str = "socialTwitter"
    data_dir: str = str(DATA_DIR)
    text_column: str = "clean_text"
    label_column: str = "label"
    timestamp_column: Optional[str] = None
    
    @property
    def raw_data_path(self) -> str:
        """Get the raw data path, handling different naming conventions"""
        dataset_dir = os.path.join(self.data_dir, self.dataset)
        # Try different file naming patterns
        patterns = [
            f"{self.dataset}_text_only.csv",
            f"{self.dataset}_cleaned.csv",
            "complaints_text_only.csv",      # FCPB
            "german_coal_text_only.csv",     # germanCoal
        ]
        for pattern in patterns:
            path = os.path.join(dataset_dir, pattern)
            if os.path.exists(path):
                return path
        # Default fallback
        return os.path.join(dataset_dir, f"{self.dataset}_text_only.csv")
    
    @property
    def cleaned_data_path(self) -> str:
        return os.path.join(self.data_dir, self.dataset, f"{self.dataset}_cleaned.csv")


@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    mode: str = "zero_shot"  # zero_shot, supervised, unsupervised
    embedding_dim: int = 1024
    model_path: str = str(QWEN_MODEL_PATH)
    output_dir: str = str(EMBEDDING_DIR / "outputs")
    batch_size: int = 64
    max_length: int = 512
    
    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.output_dir, self.mode, f"{{dataset}}_{self.mode}_embeddings.npy")
    
    @property
    def labels_path(self) -> str:
        return os.path.join(self.output_dir, self.mode, f"{{dataset}}_{self.mode}_labels.npy")


@dataclass
class BOWConfig:
    """BOW generation configuration"""
    vocab_size: int = 8000
    min_doc_freq: int = 10
    max_doc_freq_ratio: float = 0.5
    use_tfidf: bool = False
    language: str = "english"  # english, chinese, german
    
    # Tokenization
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False
    remove_numbers: bool = True
    lowercase: bool = True
    min_word_length: int = 3  # Minimum word length to filter short noise


@dataclass
class ModelConfig:
    """ETM model configuration"""
    # Architecture
    num_topics: int = 20
    hidden_dim: int = 512
    doc_embedding_dim: int = 1024
    word_embedding_dim: int = 1024
    encoder_dropout: float = 0.2
    encoder_activation: str = "relu"
    train_word_embeddings: bool = True  # Default: train word embeddings from scratch
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 1e-4
    
    # KL Annealing - gradual warmup for stable training
    kl_start: float = 0.0
    kl_end: float = 1.0
    kl_warmup_epochs: int = 30
    
    # Early stopping - need enough epochs for KL warmup
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 0.001
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # DataLoader优化参数
    num_workers: int = 4              # 多进程加载数据
    pin_memory: bool = True           # 加速CPU到GPU的数据传输
    persistent_workers: bool = True   # 保持worker进程存活
    prefetch_factor: int = 2          # 每个worker预取的batch数量


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    top_k_coherence: int = 10
    top_k_diversity: int = 25
    compute_stability: bool = False
    stability_runs: int = 5
    
    # pyLDAvis
    use_pyldavis: bool = True
    
    # External coherence (using gensim)
    use_external_coherence: bool = True
    coherence_measures: List[str] = field(default_factory=lambda: ["c_npmi", "c_v", "u_mass"])


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    output_dir: str = ""
    figsize: tuple = (12, 8)
    dpi: int = 150
    language: str = "en"  # en, zh
    
    # Word cloud
    use_wordcloud: bool = True
    wordcloud_max_words: int = 50
    
    # Topic visualization
    num_topics_to_show: int = 20
    num_words_per_topic: int = 10
    
    # Temporal analysis
    enable_temporal: bool = False
    time_bins: int = 10


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    bow: BOWConfig = field(default_factory=BOWConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Global settings
    device: str = "cuda"
    gpu_id: int = 1  # Use GPU 1, avoid GPU 0
    seed: int = 42
    dev_mode: bool = False
    
    # Model size for directory structure (e.g., '0.6B', '1.5B')
    model_size: str = "0.6B"
    
    # Experiment management
    # data_exp: data experiment ID (e.g. "exp_20260207_140000_vocab3500_theta_0.6B_zero_shot")
    # train_exp: training experiment ID (auto-generated if empty)
    data_exp: str = ""
    train_exp: str = ""
    
    # Output - all results go to /root/autodl-tmp/result/{model_size}/{dataset}/
    output_base_dir: str = str(RESULT_DIR)
    
    @property
    def dataset_base_dir(self) -> str:
        """Base directory for this dataset: result/{model_size}/{dataset}/"""
        if self.model_size:
            return os.path.join(self.output_base_dir, self.model_size, self.data.dataset)
        return os.path.join(self.output_base_dir, self.data.dataset)
    
    @property
    def data_exp_dir(self) -> str:
        """Data experiment directory: result/{model_size}/{dataset}/data/{data_exp}/"""
        if self.data_exp:
            return os.path.join(self.dataset_base_dir, "data", self.data_exp)
        # Fallback to legacy flat structure for backward compatibility
        return self.dataset_base_dir
    
    @property
    def result_dir(self) -> str:
        """Training result directory"""
        if self.train_exp:
            # New structure: result/{model_size}/{dataset}/models/{train_exp}/
            return os.path.join(self.dataset_base_dir, "models", self.train_exp)
        # Fallback to legacy structure
        return os.path.join(self.dataset_base_dir, self.embedding.mode)
    
    @property
    def embeddings_dir(self) -> str:
        """Directory for document embeddings"""
        if self.data_exp:
            return os.path.join(self.data_exp_dir, "embeddings")
        return os.path.join(self.dataset_base_dir, self.embedding.mode, "embeddings")
    
    @property
    def bow_dir(self) -> str:
        """Directory for BOW matrices and vocabulary"""
        if self.data_exp:
            return os.path.join(self.data_exp_dir, "bow")
        # Legacy: BOW shared across all modes
        return os.path.join(self.dataset_base_dir, "bow")
    
    @property
    def model_dir(self) -> str:
        """Directory for trained model and matrices (theta, beta)"""
        return os.path.join(self.result_dir, "model")
    
    @property
    def evaluation_dir(self) -> str:
        """Directory for evaluation results"""
        return os.path.join(self.result_dir, "evaluation")
    
    @property
    def visualization_dir(self) -> str:
        """Directory for visualizations"""
        return os.path.join(self.result_dir, "visualization")
    
    @property
    def log_dir(self) -> str:
        """Directory for training logs - always use RESULT_DIR to avoid read-only code directory"""
        # Use global RESULT_DIR directly to ensure logs are never written to code directory
        return os.path.join(str(RESULT_DIR), "logs")
    
    # Legacy properties for backward compatibility
    @property
    def output_dir(self) -> str:
        return self.model_dir
    
    @property
    def analysis_dir(self) -> str:
        return self.visualization_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "data": asdict(self.data),
            "embedding": asdict(self.embedding),
            "bow": asdict(self.bow),
            "model": asdict(self.model),
            "evaluation": asdict(self.evaluation),
            "visualization": asdict(self.visualization),
            "device": self.device,
            "gpu_id": self.gpu_id,
            "seed": self.seed,
            "dev_mode": self.dev_mode,
            "output_base_dir": self.output_base_dir
        }
    
    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load config from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        config.data = DataConfig(**data.get("data", {}))
        config.embedding = EmbeddingConfig(**data.get("embedding", {}))
        config.bow = BOWConfig(**data.get("bow", {}))
        config.model = ModelConfig(**data.get("model", {}))
        config.evaluation = EvaluationConfig(**data.get("evaluation", {}))
        config.visualization = VisualizationConfig(**data.get("visualization", {}))
        config.device = data.get("device", "cuda")
        config.gpu_id = data.get("gpu_id", 1)
        config.seed = data.get("seed", 42)
        config.dev_mode = data.get("dev_mode", False)
        config.output_base_dir = data.get("output_base_dir", str(ETM_DIR / "outputs"))
        
        return config


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for ETM pipeline"""
    parser = argparse.ArgumentParser(
        description="ETM Topic Model Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Command
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ETM model")
    _add_common_args(train_parser)
    _add_training_args(train_parser)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--timestamp", type=str, default=None, help="Model timestamp to evaluate")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    _add_common_args(viz_parser)
    viz_parser.add_argument("--timestamp", type=str, default=None, help="Model timestamp to visualize")
    viz_parser.add_argument("--no_wordcloud", action="store_true", help="Disable word clouds")
    
    # Pipeline command (full pipeline)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    _add_common_args(pipeline_parser)
    _add_training_args(pipeline_parser)
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean data")
    clean_parser.add_argument("--input", type=str, required=True, help="Input file/directory")
    clean_parser.add_argument("--output", type=str, required=True, help="Output file/directory")
    clean_parser.add_argument("--language", type=str, default="english", choices=["english", "chinese", "german"])
    
    return parser


def _add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments"""
    parser.add_argument("--dataset", type=str, default="socialTwitter",
                        help="Dataset name")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "supervised", "unsupervised"],
                        help="Embedding mode")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--dev", action="store_true",
                        help="Enable development mode with extra logging")


def _add_training_args(parser: argparse.ArgumentParser):
    """Add training arguments"""
    # Model architecture
    parser.add_argument("--num_topics", type=int, default=20,
                        help="Number of topics (5-100)")
    parser.add_argument("--vocab_size", type=int, default=5000,
                        help="Vocabulary size (1000-20000)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension (256-1024)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.002,
                        help="Learning rate")
    
    # KL annealing
    parser.add_argument("--kl_start", type=float, default=0.0,
                        help="KL weight start value")
    parser.add_argument("--kl_end", type=float, default=1.0,
                        help="KL weight end value")
    parser.add_argument("--kl_warmup", type=int, default=50,
                        help="KL warmup epochs")
    
    # Early stopping
    parser.add_argument("--no_early_stopping", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    # Word embeddings
    parser.add_argument("--train_word_embeddings", action="store_true", default=True,
                        help="Train word embeddings from scratch (default: True)")
    parser.add_argument("--no_train_word_embeddings", action="store_true",
                        help="Use pretrained Qwen word embeddings (frozen)")
    
    # Temporal analysis
    parser.add_argument("--enable_temporal", action="store_true",
                        help="Enable temporal analysis (requires timestamp column)")
    parser.add_argument("--timestamp_column", type=str, default=None,
                        help="Column name for timestamps in data")
    
    # Model size
    parser.add_argument("--model_size", type=str, default="0.6B",
                        choices=["0.6B", "4B", "8B"],
                        help="Qwen model size (default: 0.6B)")
    
    # Pipeline control
    parser.add_argument("--skip_viz", action="store_true",
                        help="Skip visualization step")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation step")
    
    # Experiment management
    parser.add_argument("--data_exp", type=str, default="",
                        help="Data experiment ID (for THETA exp management)")
    parser.add_argument("--train_exp", type=str, default="",
                        help="Training experiment ID (for THETA exp management)")
    
    # Visualization language
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "zh"],
                        help="Visualization language (chart titles, labels)")
    
    # DataLoader优化参数
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Disable pin_memory for DataLoader")
    parser.add_argument("--no_persistent_workers", action="store_true",
                        help="Disable persistent workers")


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Create config from parsed arguments"""
    # Load base config if provided
    if args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig()
    
    # Override with command line arguments
    if hasattr(args, "dataset"):
        config.data.dataset = args.dataset
        
        # Apply dataset-specific defaults if available
        if args.dataset in DATASET_CONFIGS:
            ds_config = DATASET_CONFIGS[args.dataset]
            config.bow.vocab_size = ds_config.get("vocab_size", config.bow.vocab_size)
            config.model.num_topics = ds_config.get("num_topics", config.model.num_topics)
            config.bow.min_doc_freq = ds_config.get("min_doc_freq", config.bow.min_doc_freq)
            config.bow.language = ds_config.get("language", config.bow.language)
    
    if hasattr(args, "mode"):
        config.embedding.mode = args.mode
    if hasattr(args, "dev"):
        config.dev_mode = args.dev
    
    # Training args - command line ALWAYS overrides dataset defaults
    if hasattr(args, "num_topics"):
        config.model.num_topics = args.num_topics
    if hasattr(args, "vocab_size"):
        config.bow.vocab_size = args.vocab_size
    if hasattr(args, "hidden_dim"):
        config.model.hidden_dim = args.hidden_dim
    if hasattr(args, "epochs"):
        config.model.epochs = args.epochs
    if hasattr(args, "batch_size"):
        config.model.batch_size = args.batch_size
    if hasattr(args, "learning_rate"):
        config.model.learning_rate = args.learning_rate
    if hasattr(args, "kl_start"):
        config.model.kl_start = args.kl_start
    if hasattr(args, "kl_end"):
        config.model.kl_end = args.kl_end
    if hasattr(args, "kl_warmup"):
        config.model.kl_warmup_epochs = args.kl_warmup
    if hasattr(args, "no_early_stopping"):
        config.model.early_stopping = not args.no_early_stopping
    if hasattr(args, "patience"):
        config.model.patience = args.patience
    
    # Model size
    if hasattr(args, "model_size") and args.model_size:
        config.model_size = args.model_size
    
    # Word embeddings - default is True (train from scratch)
    if hasattr(args, "no_train_word_embeddings") and args.no_train_word_embeddings:
        config.model.train_word_embeddings = False
    else:
        config.model.train_word_embeddings = True
    
    # Temporal analysis
    if hasattr(args, "enable_temporal"):
        config.visualization.enable_temporal = args.enable_temporal
    if hasattr(args, "timestamp_column") and args.timestamp_column:
        config.data.timestamp_column = args.timestamp_column
    
    # DataLoader优化参数
    if hasattr(args, "num_workers"):
        config.model.num_workers = args.num_workers
    if hasattr(args, "no_pin_memory") and args.no_pin_memory:
        config.model.pin_memory = False
    if hasattr(args, "no_persistent_workers") and args.no_persistent_workers:
        config.model.persistent_workers = False
    
    # Visualization language
    if hasattr(args, "language") and args.language:
        config.visualization.language = args.language
    
    # Experiment management
    if hasattr(args, "data_exp") and args.data_exp:
        config.data_exp = args.data_exp
    if hasattr(args, "train_exp") and args.train_exp:
        config.train_exp = args.train_exp
    
    return config


# ============================================================================
# Parameter Constraints - Parameter constraint definitions for validating frontend inputs
# ============================================================================

PARAM_CONSTRAINTS = {
    "num_topics": {
        "type": "int",
        "min": 5,
        "max": 100,
        "default": 20,
        "options": [5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
        "description": "Number of topics, recommended to choose based on dataset size"
    },
    "vocab_size": {
        "type": "int",
        "min": 1000,
        "max": 20000,
        "default": 5000,
        "options": [1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000],
        "description": "Vocabulary size, recommended: sqrt(num_docs) * 50"
    },
    "hidden_dim": {
        "type": "int",
        "min": 128,
        "max": 1024,
        "default": 512,
        "options": [256, 512, 768, 1024],
        "description": "Encoder hidden layer dimension"
    },
    "epochs": {
        "type": "int",
        "min": 10,
        "max": 500,
        "default": 50,
        "options": [20, 30, 50, 100, 150, 200],
        "description": "Number of training epochs"
    },
    "batch_size": {
        "type": "int",
        "min": 8,
        "max": 512,
        "default": 64,
        "options": [16, 32, 64, 128, 256],
        "description": "Batch size"
    },
    "learning_rate": {
        "type": "float",
        "min": 0.00001,
        "max": 0.1,
        "default": 0.002,
        "options": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01],
        "description": "Learning rate"
    },
}


def validate_params(params: dict) -> tuple:
    """
    Validate if parameters are valid
    
    Args:
        params: Parameter dictionary, e.g., {"num_topics": 20, "vocab_size": 5000}
        
    Returns:
        (is_valid: bool, error_message: str)
        
    Example:
        is_valid, msg = validate_params({"num_topics": 200})
        # is_valid = False, msg = "num_topics=200 exceeds maximum 100"
    """
    for param_name, value in params.items():
        if param_name not in PARAM_CONSTRAINTS:
            continue
        
        constraints = PARAM_CONSTRAINTS[param_name]
        
        # Type check
        expected_type = constraints.get("type", "int")
        if expected_type == "int" and not isinstance(value, int):
            return False, f"{param_name} must be an integer, got {type(value).__name__}"
        if expected_type == "float" and not isinstance(value, (int, float)):
            return False, f"{param_name} must be a number, got {type(value).__name__}"
        
        # Range check
        if "min" in constraints and value < constraints["min"]:
            return False, f"{param_name}={value} is below minimum {constraints['min']}"
        if "max" in constraints and value > constraints["max"]:
            return False, f"{param_name}={value} exceeds maximum {constraints['max']}"
    
    return True, "OK"


def get_param_options() -> dict:
    """
    Get all parameter options - for frontend dropdown menus
    
    Returns:
        {
            "num_topics": {"options": [5, 10, ...], "default": 20, "description": "..."},
            ...
        }
    """
    return PARAM_CONSTRAINTS


# ============================================================================
# Predefined configurations for common use cases
# ============================================================================

PRESET_CONFIGS = {
    "small": {
        "num_topics": 10,
        "vocab_size": 2000,
        "epochs": 30,
        "hidden_dim": 256
    },
    "medium": {
        "num_topics": 20,
        "vocab_size": 5000,
        "epochs": 50,
        "hidden_dim": 512
    },
    "large": {
        "num_topics": 50,
        "vocab_size": 10000,
        "epochs": 100,
        "hidden_dim": 1024
    }
}


def get_preset_config(preset: str) -> PipelineConfig:
    """Get a preset configuration"""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PipelineConfig()
    preset_values = PRESET_CONFIGS[preset]
    
    config.model.num_topics = preset_values["num_topics"]
    config.bow.vocab_size = preset_values["vocab_size"]
    config.model.epochs = preset_values["epochs"]
    config.model.hidden_dim = preset_values["hidden_dim"]
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = PipelineConfig()
    config.data.dataset = "socialTwitter"
    config.embedding.mode = "zero_shot"
    
    print("Default configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save and load test
    config.save("/tmp/test_config.json")
    loaded_config = PipelineConfig.load("/tmp/test_config.json")
    print("\nLoaded configuration matches:", config.to_dict() == loaded_config.to_dict())

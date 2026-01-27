"""
Topic Models Module

Supports multiple topic models:
- ETM: Embedded Topic Model (using Qwen embedding)
- CTM: Contextualized Topic Model
- LDA: Latent Dirichlet Allocation (classic LDA)
- DTM: Dynamic Topic Model (reserved)

All models implement a unified interface for easy comparison experiments.
"""

# ETM
from .etm import ETM
from .encoder import ETMEncoder
from .decoder import ETMDecoder

# CTM
from .ctm import CTM, ZeroShotTM, CombinedTM, create_ctm

# LDA
from .lda import LDA, SklearnLDA, NeuralLDA, create_lda

# DTM
from .dtm import DTM, create_dtm

# Base classes
from .base import BaseTopicModel, NeuralTopicModel, TraditionalTopicModel

# Registry
from .registry import (
    get_topic_model_options,
    get_model_info,
    get_model_class,
    get_default_params,
    list_available_models,
    register_model
)

# Trainer
from .trainer import TopicModelTrainer, train_baseline_models

# Baseline (independent of Qwen embedding)
from .baseline_data import BaselineDataProcessor, prepare_baseline_data
from .baseline_trainer import BaselineTrainer
from .etm_original import OriginalETM, create_original_etm, train_word2vec_embeddings
from .baseline_evaluator import BaselineEvaluator, compare_all_models, print_comparison_table

__all__ = [
    # ETM
    'ETM', 'ETMEncoder', 'ETMDecoder',
    # CTM
    'CTM', 'ZeroShotTM', 'CombinedTM', 'create_ctm',
    # LDA
    'LDA', 'SklearnLDA', 'NeuralLDA', 'create_lda',
    # DTM
    'DTM', 'create_dtm',
    # Base
    'BaseTopicModel', 'NeuralTopicModel', 'TraditionalTopicModel',
    # Registry
    'get_topic_model_options', 'get_model_info', 'get_model_class',
    'get_default_params', 'list_available_models', 'register_model',
    # Trainer
    'TopicModelTrainer', 'train_baseline_models',
    # Baseline
    'BaselineDataProcessor', 'prepare_baseline_data', 'BaselineTrainer',
    # Original ETM (Baseline)
    'OriginalETM', 'create_original_etm', 'train_word2vec_embeddings',
    # Evaluator
    'BaselineEvaluator', 'compare_all_models', 'print_comparison_table',
]

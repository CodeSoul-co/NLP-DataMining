"""
Topic Models Module

Supports multiple topic models:

THETA (Main Model):
- ETM: Embedded Topic Model (using Qwen embedding)

Baselines:
- LDA: Latent Dirichlet Allocation (classic LDA)
- CTM: Contextualized Topic Model (SBERT + VAE)
- ETM: Original Embedded Topic Model (Word2Vec + VAE)
- DTM: Dynamic Topic Model (time-aware)
- HDP: Hierarchical Dirichlet Process (auto topic number)
- STM: Structural Topic Model (with covariates)
- BTM: Biterm Topic Model (for short texts)
- NTM: Neural Topic Model (NVDM/GSM/ProdLDA variants)
- BERTopic: BERT-based topic modeling

All models implement a unified interface for easy comparison experiments.
"""

# THETA (Main Model)
from .theta import ETM, ETMEncoder, ETMDecoder

# Baselines
from .baselines import (
    # LDA
    LDA, SklearnLDA, NeuralLDA, create_lda,
    # HDP
    HDP, create_hdp,
    # STM
    STM, create_stm,
    # BTM
    BTM, create_btm,
    # CTM
    CTM, ZeroShotTM, CombinedTM, create_ctm,
    # DTM
    DTM, create_dtm,
    # Original ETM
    OriginalETM, create_original_etm, train_word2vec_embeddings,
    # NTM variants
    NVDM, GSM, ProdLDA, create_nvdm, create_gsm, create_prodlda,
    # BERTopic
    BERTopicModel, create_bertopic,
)

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

# Baseline utilities
from .baseline_data import BaselineDataProcessor, prepare_baseline_data
from .baseline_trainer import BaselineTrainer
from .baseline_evaluator import BaselineEvaluator, compare_all_models, print_comparison_table

__all__ = [
    # THETA (Main Model)
    'ETM', 'ETMEncoder', 'ETMDecoder',
    # Baselines - LDA
    'LDA', 'SklearnLDA', 'NeuralLDA', 'create_lda',
    # Baselines - HDP
    'HDP', 'create_hdp',
    # Baselines - STM
    'STM', 'create_stm',
    # Baselines - BTM
    'BTM', 'create_btm',
    # Baselines - CTM
    'CTM', 'ZeroShotTM', 'CombinedTM', 'create_ctm',
    # Baselines - DTM
    'DTM', 'create_dtm',
    # Baselines - Original ETM
    'OriginalETM', 'create_original_etm', 'train_word2vec_embeddings',
    # Baselines - NTM variants
    'NVDM', 'GSM', 'ProdLDA', 'create_nvdm', 'create_gsm', 'create_prodlda',
    # Baselines - BERTopic
    'BERTopicModel', 'create_bertopic',
    # Base
    'BaseTopicModel', 'NeuralTopicModel', 'TraditionalTopicModel',
    # Registry
    'get_topic_model_options', 'get_model_info', 'get_model_class',
    'get_default_params', 'list_available_models', 'register_model',
    # Trainer
    'TopicModelTrainer', 'train_baseline_models',
    # Baseline utilities
    'BaselineDataProcessor', 'prepare_baseline_data', 'BaselineTrainer',
    # Evaluator
    'BaselineEvaluator', 'compare_all_models', 'print_comparison_table',
]

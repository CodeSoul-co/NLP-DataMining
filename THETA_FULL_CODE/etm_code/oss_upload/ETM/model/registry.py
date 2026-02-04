"""
Topic Model Registry - Topic model registry

Convenient for frontend and backend personnel to select different topic models.
More models (such as LDA, NTM, etc.) can be added here in the future.

Usage:
    from model.registry import get_topic_model_options, get_model_class
    
    # Get all available models (for frontend dropdown)
    options = get_topic_model_options()
    
    # Get model class
    ModelClass = get_model_class("etm")
"""

from typing import Dict, Any, Optional, Type
from dataclasses import dataclass


@dataclass
class TopicModelInfo:
    """Topic model information"""
    name: str                           # Display name
    description: str                    # Model description
    module_path: str                    # Module path
    class_name: str                     # Class name
    supports_embeddings: bool           # Whether supports pre-trained embeddings
    supports_pretrained_words: bool     # Whether supports pre-trained word vectors
    default_params: Dict[str, Any]      # Default parameters
    param_options: Dict[str, list]      # Parameter options


# ============================================================================
# Model registry - add new models here
# ============================================================================

TOPIC_MODEL_REGISTRY: Dict[str, TopicModelInfo] = {
    "etm": TopicModelInfo(
        name="ETM (Embedded Topic Model)",
        description="VAE-based topic model, uses Qwen word vectors as semantic foundation, "
                   "maps document embeddings to topic distribution through encoder, "
                   "decoder uses word vectors to reconstruct BOW.",
        module_path="model.etm",
        class_name="ETM",
        supports_embeddings=True,
        supports_pretrained_words=True,
        default_params={
            "num_topics": 20,
            "hidden_dim": 512,
            "doc_embedding_dim": 1024,
            "word_embedding_dim": 1024,
            "encoder_dropout": 0.2,
            "train_word_embeddings": True,
        },
        param_options={
            "num_topics": [5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
            "hidden_dim": [256, 512, 768, 1024],
            "encoder_dropout": [0.1, 0.2, 0.3, 0.5],
        }
    ),
    
    # CTM - Contextualized Topic Model (Baseline)
    "ctm": TopicModelInfo(
        name="CTM (Contextualized Topic Model)",
        description="Contextualized topic model, combines pre-trained language model embeddings with topic models. "
                   "Supports ZeroShot and Combined inference modes, usually has better topic coherence.",
        module_path="model.ctm",
        class_name="CTM",
        supports_embeddings=True,
        supports_pretrained_words=False,
        default_params={
            "num_topics": 20,
            "doc_embedding_dim": 1024,
            "hidden_sizes": (100, 100),
            "activation": "softplus",
            "dropout": 0.2,
            "model_type": "prodLDA",
            "inference_type": "zeroshot",
        },
        param_options={
            "num_topics": [5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
            "hidden_sizes": [(100, 100), (256, 256), (512, 512)],
            "model_type": ["prodLDA", "LDA"],
            "inference_type": ["zeroshot", "combined"],
        }
    ),
    
    # LDA - Classic probabilistic topic model (Baseline)
    "lda": TopicModelInfo(
        name="LDA (Latent Dirichlet Allocation)",
        description="Classic probabilistic topic model, based on sklearn implementation. "
                   "Does not require pre-trained embeddings, directly uses BOW as input. "
                   "Suitable as Baseline for comparison.",
        module_path="model.lda",
        class_name="SklearnLDA",
        supports_embeddings=False,
        supports_pretrained_words=False,
        default_params={
            "num_topics": 20,
            "max_iter": 100,
            "learning_method": "batch",
        },
        param_options={
            "num_topics": [5, 10, 20, 50, 100],
            "max_iter": [50, 100, 200],
            "learning_method": ["batch", "online"],
        }
    ),
    
    # Neural LDA - Neural network LDA
    "neural_lda": TopicModelInfo(
        name="Neural LDA (VAE-based)",
        description="VAE-based neural network LDA implementation. "
                   "Does not require pre-trained embeddings, can be GPU accelerated.",
        module_path="model.lda",
        class_name="NeuralLDA",
        supports_embeddings=False,
        supports_pretrained_words=False,
        default_params={
            "num_topics": 20,
            "hidden_dim": 256,
            "encoder_dropout": 0.2,
            "kl_weight": 1.0,
        },
        param_options={
            "num_topics": [5, 10, 20, 50, 100],
            "hidden_dim": [128, 256, 512],
        }
    ),
    
    # DTM - Dynamic Topic Model (reserved)
    "dtm": TopicModelInfo(
        name="DTM (Dynamic Topic Model)",
        description="Dynamic topic model, supports time series analysis, can track topic evolution over time. "
                   "Suitable for document collections with timestamps.",
        module_path="model.dtm",
        class_name="DTM",
        supports_embeddings=True,
        supports_pretrained_words=True,
        default_params={
            "num_topics": 20,
            "time_slices": 10,
            "hidden_dim": 512,
            "doc_embedding_dim": 1024,
            "word_embedding_dim": 1024,
            "evolution_weight": 0.1,
        },
        param_options={
            "num_topics": [5, 10, 15, 20, 30, 50],
            "time_slices": [5, 10, 20, 50],
            "hidden_dim": [256, 512, 1024],
        }
    ),
}


# ============================================================================
# API functions
# ============================================================================

def get_topic_model_options() -> Dict[str, Dict[str, Any]]:
    """
    Get all available topic model options - for frontend dropdown
    
    Returns:
        {
            "etm": {
                "name": "ETM (Embedded Topic Model)",
                "description": "...",
                "supports_embeddings": True,
                "default_params": {...},
                "param_options": {...}
            },
            ...
        }
    """
    return {
        model_id: {
            "name": info.name,
            "description": info.description,
            "supports_embeddings": info.supports_embeddings,
            "supports_pretrained_words": info.supports_pretrained_words,
            "default_params": info.default_params,
            "param_options": info.param_options,
        }
        for model_id, info in TOPIC_MODEL_REGISTRY.items()
    }


def get_model_info(model_id: str) -> Optional[TopicModelInfo]:
    """
    Get detailed information for specified model
    
    Args:
        model_id: Model ID (e.g. "etm")
        
    Returns:
        TopicModelInfo or None
    """
    return TOPIC_MODEL_REGISTRY.get(model_id)


def get_model_class(model_id: str) -> Type:
    """
    Get model class
    
    Args:
        model_id: Model ID (e.g. "etm")
        
    Returns:
        Model class
        
    Raises:
        ValueError: If model does not exist
        ImportError: If module import fails
    """
    if model_id not in TOPIC_MODEL_REGISTRY:
        available = list(TOPIC_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_id}. Available: {available}")
    
    info = TOPIC_MODEL_REGISTRY[model_id]
    
    # Dynamically import module
    import importlib
    module = importlib.import_module(info.module_path)
    model_class = getattr(module, info.class_name)
    
    return model_class


def get_default_params(model_id: str) -> Dict[str, Any]:
    """
    Get default parameters for model
    
    Args:
        model_id: Model ID
        
    Returns:
        Default parameters dictionary
    """
    info = TOPIC_MODEL_REGISTRY.get(model_id)
    if info is None:
        return {}
    return info.default_params.copy()


def list_available_models() -> list:
    """
    List all available model IDs
    
    Returns:
        List of model IDs
    """
    return list(TOPIC_MODEL_REGISTRY.keys())


def register_model(
    model_id: str,
    name: str,
    description: str,
    module_path: str,
    class_name: str,
    supports_embeddings: bool = False,
    supports_pretrained_words: bool = False,
    default_params: Optional[Dict[str, Any]] = None,
    param_options: Optional[Dict[str, list]] = None
) -> None:
    """
    Register new model - for dynamically adding models
    
    Args:
        model_id: Unique model identifier
        name: Display name
        description: Model description
        module_path: Module path
        class_name: Class name
        supports_embeddings: Whether supports pre-trained embeddings
        supports_pretrained_words: Whether supports pre-trained word vectors
        default_params: Default parameters
        param_options: Parameter options
    """
    TOPIC_MODEL_REGISTRY[model_id] = TopicModelInfo(
        name=name,
        description=description,
        module_path=module_path,
        class_name=class_name,
        supports_embeddings=supports_embeddings,
        supports_pretrained_words=supports_pretrained_words,
        default_params=default_params or {},
        param_options=param_options or {}
    )


# ============================================================================
# CLI testing
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("Available Topic Models:")
    print("=" * 60)
    
    for model_id, info in TOPIC_MODEL_REGISTRY.items():
        print(f"\n[{model_id}] {info.name}")
        print(f"  Description: {info.description[:80]}...")
        print(f"  Supports Embeddings: {info.supports_embeddings}")
        print(f"  Default params: {info.default_params}")
    
    print("\n" + "=" * 60)
    print("\nAPI Output (get_topic_model_options):")
    print(json.dumps(get_topic_model_options(), indent=2))

"""
Embedding Model Registry

Conveniently select different embedding models for both frontend and backend usage.
More models can be added here in the future (e.g., BGE, M3E).

Usage:
    from registry import get_embedding_model_options, get_embedding_model_path
    
    # Get all available models (for frontend dropdown)
    options = get_embedding_model_options()
    
    # Get model path
    path = get_embedding_model_path("qwen3_0.6B")
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


# Base paths - use environment variables for DLC compatibility
BASE_DIR = Path(os.environ.get("THETA_BASE", "/root/autodl-tmp"))
EMBEDDING_MODELS_DIR = Path(os.environ.get("THETA_MODEL_DIR", str(BASE_DIR / "embedding_models")))


@dataclass
class EmbeddingModelInfo:
    """Embedding model metadata"""
    name: str                           # Display name
    path: str                           # Model path
    embedding_dim: int                  # Embedding dimension
    max_length: int                     # Max sequence length
    description: str                    # Model description
    languages: list                     # Supported languages
    model_size: str                     # Model size (parameter count)
    requires_gpu: bool                  # Whether GPU is required
    default_batch_size: int             # Recommended batch size


# ============================================================================
# Model registry - add new models here
# ============================================================================

EMBEDDING_MODEL_REGISTRY: Dict[str, EmbeddingModelInfo] = {
    "0.6B": EmbeddingModelInfo(
        name="Qwen3-Embedding-0.6B",
        path=str(EMBEDDING_MODELS_DIR / "qwen3_embedding_0.6B"),
        embedding_dim=1024,
        max_length=512,
        description="Alibaba Cloud Qwen3 lightweight embedding model, supports Chinese and English, "
                   "suitable for resource-limited environments, with fast inference.",
        languages=["chinese", "english", "multi"],
        model_size="0.6B",
        requires_gpu=True,
        default_batch_size=16
    ),
    
    "4B": EmbeddingModelInfo(
        name="Qwen3-Embedding-4B",
        path=str(EMBEDDING_MODELS_DIR / "qwen3_embedding_4B"),
        embedding_dim=2560,
        max_length=512,
        description="Alibaba Cloud Qwen3 mid-size embedding model, better quality but requires more VRAM.",
        languages=["chinese", "english", "multi"],
        model_size="4B",
        requires_gpu=True,
        default_batch_size=8
    ),
    
    "8B": EmbeddingModelInfo(
        name="Qwen3-Embedding-8B",
        path=str(EMBEDDING_MODELS_DIR / "qwen3_embedding_8B"),
        embedding_dim=4096,
        max_length=512,
        description="Alibaba Cloud Qwen3 large embedding model, best quality, requires large VRAM.",
        languages=["chinese", "english", "multi"],
        model_size="8B",
        requires_gpu=True,
        default_batch_size=4
    ),
    
    # =========== More models can be added in the future ===========
    # "bge_large_zh": EmbeddingModelInfo(
    #     name="BGE-Large-Chinese",
    #     path=str(EMBEDDING_MODELS_DIR / "bge-large-zh-v1.5"),
    #     embedding_dim=1024,
    #     max_length=512,
    #     description="BAAI BGE Chinese embedding model, excellent Chinese performance",
    #     languages=["chinese"],
    #     model_size="0.3B",
    #     requires_gpu=True,
    #     default_batch_size=32
    # ),
}


# ============================================================================
# API functions
# ============================================================================

def get_embedding_model_options() -> Dict[str, Dict[str, Any]]:
    """
    Get options for all available embedding models (for frontend dropdown)
    
    Returns:
        {
            "qwen3_0.6B": {
                "name": "Qwen3-Embedding-0.6B",
                "path": "/root/autodl-tmp/qwen3_embedding_0.6B",
                "embedding_dim": 1024,
                "description": "...",
                ...
            },
            ...
        }
    """
    result = {}
    for model_id, info in EMBEDDING_MODEL_REGISTRY.items():
        # Check if the model exists
        model_exists = os.path.exists(info.path)
        
        result[model_id] = {
            "name": info.name,
            "path": info.path,
            "embedding_dim": info.embedding_dim,
            "max_length": info.max_length,
            "description": info.description,
            "languages": info.languages,
            "model_size": info.model_size,
            "requires_gpu": info.requires_gpu,
            "default_batch_size": info.default_batch_size,
            "available": model_exists,  # Whether the model is already downloaded
        }
    return result


def get_embedding_model_info(model_id: str) -> Optional[EmbeddingModelInfo]:
    """
    Get detailed info for a specified model
    
    Args:
        model_id: Model ID (e.g., "qwen3_0.6B")
        
    Returns:
        EmbeddingModelInfo or None
    """
    return EMBEDDING_MODEL_REGISTRY.get(model_id)


def get_embedding_model_path(model_id: str) -> str:
    """
    Get model path
    
    Args:
        model_id: Model ID
        
    Returns:
        Model path string
        
    Raises:
        ValueError: If the model does not exist
    """
    if model_id not in EMBEDDING_MODEL_REGISTRY:
        available = list(EMBEDDING_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown embedding model: {model_id}. Available: {available}")
    
    return EMBEDDING_MODEL_REGISTRY[model_id].path


def get_embedding_dim(model_id: str) -> int:
    """
    Get embedding dimension for a model
    
    Args:
        model_id: Model ID
        
    Returns:
        Embedding dimension
    """
    info = EMBEDDING_MODEL_REGISTRY.get(model_id)
    if info is None:
        return 1024  # Default
    return info.embedding_dim


def list_available_models() -> list:
    """
    List all available model IDs
    
    Returns:
        Model ID list
    """
    return list(EMBEDDING_MODEL_REGISTRY.keys())


def list_downloaded_models() -> list:
    """
    List downloaded models
    
    Returns:
        Downloaded model ID list
    """
    return [
        model_id for model_id, info in EMBEDDING_MODEL_REGISTRY.items()
        if os.path.exists(info.path)
    ]


def register_model(
    model_id: str,
    name: str,
    path: str,
    embedding_dim: int,
    max_length: int = 512,
    description: str = "",
    languages: Optional[list] = None,
    model_size: str = "unknown",
    requires_gpu: bool = True,
    default_batch_size: int = 32
) -> None:
    """
    Register a new model (for dynamically adding models)
    
    Args:
        model_id: Unique model identifier
        name: Display name
        path: Model path
        embedding_dim: Embedding dimension
        max_length: Max sequence length
        description: Model description
        languages: Supported language list
        model_size: Model size
        requires_gpu: Whether GPU is required
        default_batch_size: Recommended batch size
    """
    EMBEDDING_MODEL_REGISTRY[model_id] = EmbeddingModelInfo(
        name=name,
        path=path,
        embedding_dim=embedding_dim,
        max_length=max_length,
        description=description,
        languages=languages or ["english"],
        model_size=model_size,
        requires_gpu=requires_gpu,
        default_batch_size=default_batch_size
    )


def get_recommended_model(language: str = "english") -> str:
    """
    Get recommended model by language
    
    Args:
        language: Language (english/chinese/german/multi)
        
    Returns:
        Recommended model ID
    """
    # Prefer downloaded models
    downloaded = list_downloaded_models()
    
    for model_id in downloaded:
        info = EMBEDDING_MODEL_REGISTRY[model_id]
        if language in info.languages or "multi" in info.languages:
            return model_id
    
    # If none are downloaded, return default
    return "qwen3_0.6B"


# ============================================================================
# CLI test
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("Available Embedding Models:")
    print("=" * 60)
    
    for model_id, info in EMBEDDING_MODEL_REGISTRY.items():
        exists = "✓" if os.path.exists(info.path) else "✗"
        print(f"\n[{exists}] {model_id}: {info.name}")
        print(f"    Path: {info.path}")
        print(f"    Dim: {info.embedding_dim}, Max Length: {info.max_length}")
        print(f"    Languages: {info.languages}")
        print(f"    Size: {info.model_size}, GPU: {info.requires_gpu}")
    
    print("\n" + "=" * 60)
    print("\nDownloaded models:", list_downloaded_models())
    
    print("\n" + "=" * 60)
    print("\nAPI Output (get_embedding_model_options):")
    print(json.dumps(get_embedding_model_options(), indent=2))

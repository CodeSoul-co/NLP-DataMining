"""
Embedding Model Registry - Embedding模型注册表

方便前后端人员选择不同的Embedding模型。
未来可以在这里添加更多模型（如BGE、M3E等）。

Usage:
    from registry import get_embedding_model_options, get_embedding_model_path
    
    # 获取所有可用模型（用于前端下拉框）
    options = get_embedding_model_options()
    
    # 获取模型路径
    path = get_embedding_model_path("qwen3_0.6B")
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


# 基础路径
BASE_DIR = Path("/root/autodl-tmp")


@dataclass
class EmbeddingModelInfo:
    """Embedding模型信息"""
    name: str                           # 显示名称
    path: str                           # 模型路径
    embedding_dim: int                  # 嵌入维度
    max_length: int                     # 最大序列长度
    description: str                    # 模型描述
    languages: list                     # 支持的语言
    model_size: str                     # 模型大小 (参数量)
    requires_gpu: bool                  # 是否需要GPU
    default_batch_size: int             # 推荐批次大小


# ============================================================================
# 模型注册表 - 在这里添加新模型
# ============================================================================

EMBEDDING_MODEL_REGISTRY: Dict[str, EmbeddingModelInfo] = {
    "qwen3_0.6B": EmbeddingModelInfo(
        name="Qwen3-Embedding-0.6B",
        path=str(BASE_DIR / "qwen3_embedding_0.6B"),
        embedding_dim=1024,
        max_length=512,
        description="阿里云Qwen3系列轻量级embedding模型，支持中英文，"
                   "适合资源有限的环境，推理速度快。",
        languages=["chinese", "english", "multi"],
        model_size="0.6B",
        requires_gpu=True,
        default_batch_size=16
    ),
    
    # =========== 未来可添加更多模型 ===========
    # "qwen3_1.5B": EmbeddingModelInfo(
    #     name="Qwen3-Embedding-1.5B",
    #     path=str(BASE_DIR / "qwen3_embedding_1.5B"),
    #     embedding_dim=1536,
    #     max_length=512,
    #     description="Qwen3中等规模embedding模型，效果更好但需要更多显存",
    #     languages=["chinese", "english", "multi"],
    #     model_size="1.5B",
    #     requires_gpu=True,
    #     default_batch_size=8
    # ),
    #
    # "bge_large_zh": EmbeddingModelInfo(
    #     name="BGE-Large-Chinese",
    #     path=str(BASE_DIR / "bge-large-zh-v1.5"),
    #     embedding_dim=1024,
    #     max_length=512,
    #     description="智源BGE中文embedding模型，中文效果优秀",
    #     languages=["chinese"],
    #     model_size="0.3B",
    #     requires_gpu=True,
    #     default_batch_size=32
    # ),
    #
    # "bge_large_en": EmbeddingModelInfo(
    #     name="BGE-Large-English",
    #     path=str(BASE_DIR / "bge-large-en-v1.5"),
    #     embedding_dim=1024,
    #     max_length=512,
    #     description="智源BGE英文embedding模型",
    #     languages=["english"],
    #     model_size="0.3B",
    #     requires_gpu=True,
    #     default_batch_size=32
    # ),
    #
    # "m3e_base": EmbeddingModelInfo(
    #     name="M3E-Base",
    #     path=str(BASE_DIR / "m3e-base"),
    #     embedding_dim=768,
    #     max_length=512,
    #     description="Moka开源中文embedding模型",
    #     languages=["chinese"],
    #     model_size="0.1B",
    #     requires_gpu=False,
    #     default_batch_size=64
    # ),
    #
    # "e5_large": EmbeddingModelInfo(
    #     name="E5-Large",
    #     path=str(BASE_DIR / "e5-large-v2"),
    #     embedding_dim=1024,
    #     max_length=512,
    #     description="微软E5多语言embedding模型",
    #     languages=["english", "multi"],
    #     model_size="0.3B",
    #     requires_gpu=True,
    #     default_batch_size=32
    # ),
}


# ============================================================================
# API函数
# ============================================================================

def get_embedding_model_options() -> Dict[str, Dict[str, Any]]:
    """
    获取所有可用的Embedding模型选项 - 供前端下拉框使用
    
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
        # 检查模型是否存在
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
            "available": model_exists,  # 标记模型是否已下载
        }
    return result


def get_embedding_model_info(model_id: str) -> Optional[EmbeddingModelInfo]:
    """
    获取指定模型的详细信息
    
    Args:
        model_id: 模型ID (如 "qwen3_0.6B")
        
    Returns:
        EmbeddingModelInfo 或 None
    """
    return EMBEDDING_MODEL_REGISTRY.get(model_id)


def get_embedding_model_path(model_id: str) -> str:
    """
    获取模型路径
    
    Args:
        model_id: 模型ID
        
    Returns:
        模型路径字符串
        
    Raises:
        ValueError: 如果模型不存在
    """
    if model_id not in EMBEDDING_MODEL_REGISTRY:
        available = list(EMBEDDING_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown embedding model: {model_id}. Available: {available}")
    
    return EMBEDDING_MODEL_REGISTRY[model_id].path


def get_embedding_dim(model_id: str) -> int:
    """
    获取模型的嵌入维度
    
    Args:
        model_id: 模型ID
        
    Returns:
        嵌入维度
    """
    info = EMBEDDING_MODEL_REGISTRY.get(model_id)
    if info is None:
        return 1024  # 默认值
    return info.embedding_dim


def list_available_models() -> list:
    """
    列出所有可用的模型ID
    
    Returns:
        模型ID列表
    """
    return list(EMBEDDING_MODEL_REGISTRY.keys())


def list_downloaded_models() -> list:
    """
    列出已下载的模型
    
    Returns:
        已下载的模型ID列表
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
    注册新模型 - 用于动态添加模型
    
    Args:
        model_id: 模型唯一标识
        name: 显示名称
        path: 模型路径
        embedding_dim: 嵌入维度
        max_length: 最大序列长度
        description: 模型描述
        languages: 支持的语言列表
        model_size: 模型大小
        requires_gpu: 是否需要GPU
        default_batch_size: 推荐批次大小
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
    根据语言获取推荐的模型
    
    Args:
        language: 语言 (english/chinese/german/multi)
        
    Returns:
        推荐的模型ID
    """
    # 优先选择已下载的模型
    downloaded = list_downloaded_models()
    
    for model_id in downloaded:
        info = EMBEDDING_MODEL_REGISTRY[model_id]
        if language in info.languages or "multi" in info.languages:
            return model_id
    
    # 如果没有已下载的，返回默认
    return "qwen3_0.6B"


# ============================================================================
# CLI测试
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

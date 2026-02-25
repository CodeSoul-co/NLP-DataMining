"""
THETA Model - Topic Hierarchical Embedding with Transformer Architecture

The main topic model using Qwen embeddings for document representation.
"""

from .etm import ETM
from .encoder import ETMEncoder
from .decoder import ETMDecoder

__all__ = ['ETM', 'ETMEncoder', 'ETMDecoder']

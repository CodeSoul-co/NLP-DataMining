"""
Engine A: Data Preprocessing and Global Vocabulary Building

This module handles:
- Text cleaning and normalization
- Global vocabulary construction across all datasets
- BOW (Bag-of-Words) matrix generation for ETM reconstruction target
"""

from .vocab_builder import VocabBuilder
from .bow_generator import BOWGenerator

__all__ = ['VocabBuilder', 'BOWGenerator']

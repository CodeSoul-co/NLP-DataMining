"""
Base Topic Model Interface

All topic models should inherit from this base class to ensure a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


class BaseTopicModel(ABC):
    """
    Base topic model interface
    
    All topic models must implement the following methods:
    - fit(): Train the model
    - get_theta(): Get document-topic distribution
    - get_beta(): Get topic-word distribution
    - get_topic_words(): Get topic words
    """
    
    @property
    @abstractmethod
    def num_topics(self) -> int:
        """Number of topics"""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size"""
        pass
    
    @abstractmethod
    def get_theta(self, **kwargs) -> np.ndarray:
        """
        Get document-topic distribution
        
        Returns:
            theta: (num_docs, num_topics) Document-topic distribution matrix
        """
        pass
    
    @abstractmethod
    def get_beta(self) -> np.ndarray:
        """
        Get topic-word distribution
        
        Returns:
            beta: (num_topics, vocab_size) Topic-word distribution matrix
        """
        pass
    
    @abstractmethod
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get top words for each topic
        
        Args:
            vocab: Vocabulary list
            top_k: Number of words to return per topic
            
        Returns:
            {topic_id: [word1, word2, ...]}
        """
        pass
    
    def get_topic_words_with_weights(
        self,
        vocab: List[str],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top words and their weights for each topic
        
        Args:
            vocab: Vocabulary list
            top_k: Number of words to return per topic
            
        Returns:
            {topic_id: [(word1, weight1), (word2, weight2), ...]}
        """
        beta = self.get_beta()
        topic_words = {}
        
        for k in range(self.num_topics):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [
                (vocab[i], float(beta[k, i])) for i in top_indices
            ]
        
        return topic_words


class NeuralTopicModel(BaseTopicModel, nn.Module):
    """
    Neural network topic model base class
    
    Inherits from BaseTopicModel and nn.Module, for neural network-based topic models.
    """
    
    def __init__(self):
        nn.Module.__init__(self)
    
    @abstractmethod
    def forward(
        self,
        doc_embeddings: torch.Tensor,
        bow: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            doc_embeddings: Document embeddings (batch, embedding_dim)
            bow: BOW matrix (batch, vocab_size)
            
        Returns:
            Dict containing at least:
                - loss: Total loss
                - theta: Topic distribution
        """
        pass
    
    def save_model(self, path: str) -> None:
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        pass
    
    @classmethod
    def load_model(cls, path: str, device: torch.device) -> 'NeuralTopicModel':
        """Load model"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


class TraditionalTopicModel(BaseTopicModel):
    """
    Traditional topic model base class
    
    For non-neural network traditional topic models (e.g., sklearn's LDA).
    """
    
    @abstractmethod
    def fit(
        self,
        bow_matrix: np.ndarray,
        **kwargs
    ) -> 'TraditionalTopicModel':
        """
        Train the model
        
        Args:
            bow_matrix: BOW matrix (num_docs, vocab_size)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        bow_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Infer document topic distribution
        
        Args:
            bow_matrix: BOW matrix (num_docs, vocab_size)
            
        Returns:
            theta: (num_docs, num_topics)
        """
        pass
    
    def fit_transform(
        self,
        bow_matrix: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Train and infer
        """
        self.fit(bow_matrix, **kwargs)
        return self.transform(bow_matrix)

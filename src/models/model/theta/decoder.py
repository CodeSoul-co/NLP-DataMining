"""
ETM Decoder: Generates word distributions from topic distributions.

Uses word embeddings (from Qwen) to compute topic-word distributions.
This ensures interpretability of topics through word associations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ETMDecoder(nn.Module):
    """
    Decoder for ETM that generates word distributions.
    
    Architecture:
        theta (K) -> beta (K x V) -> word_dist (V)
        
    Where beta is computed as:
        beta = softmax(topic_embeddings @ word_embeddings.T)
    """
    
    def __init__(
        self,
        vocab_size: int,          # Vocabulary size (V)
        num_topics: int,          # Number of topics (K)
        embedding_dim: int,       # Word embedding dimension
        word_embeddings: Optional[torch.Tensor] = None,  # Pre-trained word embeddings
        train_embeddings: bool = False  # Whether to train word embeddings
    ):
        """
        Initialize decoder.
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics
            embedding_dim: Dimension of word embeddings
            word_embeddings: Pre-trained word embeddings (V x E)
            train_embeddings: Whether to fine-tune word embeddings
        """
        super(ETMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.embedding_dim = embedding_dim
        self.train_embeddings = train_embeddings
        
        # Word embeddings (rho in original ETM)
        if word_embeddings is not None:
            # Use pre-trained embeddings
            self.word_embeddings = nn.Parameter(
                word_embeddings.clone(),
                requires_grad=train_embeddings
            )
        else:
            # Initialize random embeddings
            # Use std=0.1 to create meaningful logit differences after dot product
            self.word_embeddings = nn.Parameter(
                torch.randn(vocab_size, embedding_dim) * 0.1,
                requires_grad=True
            )
        
        # Topic embeddings (alpha in original ETM)
        # Use larger std for topic embeddings to create sharper initial beta
        self.topic_embeddings = nn.Parameter(
            torch.randn(num_topics, embedding_dim) * 0.5
        )
    
    def get_beta(self) -> torch.Tensor:
        """
        Compute topic-word distribution matrix.
        
        beta[k, v] = P(word v | topic k)
        
        Returns:
            beta: Topic-word distribution, shape (K, V)
        """
        # Compute similarity between topic embeddings and word embeddings
        # Shape: (K, E) @ (E, V) -> (K, V)
        logits = torch.mm(self.topic_embeddings, self.word_embeddings.t())
        
        # Softmax over vocabulary dimension
        beta = F.softmax(logits, dim=-1)
        
        return beta
    
    def forward(
        self,
        theta: torch.Tensor,
        beta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode topic distribution to word distribution.
        
        Args:
            theta: Topic distribution, shape (batch_size, K)
            beta: Pre-computed topic-word distribution (optional)
            
        Returns:
            word_dist: Log word distribution, shape (batch_size, V)
        """
        if beta is None:
            beta = self.get_beta()
        
        # Compute word distribution: theta @ beta
        # Shape: (batch_size, K) @ (K, V) -> (batch_size, V)
        word_dist = torch.mm(theta, beta)
        
        # Add small epsilon for numerical stability
        word_dist = word_dist + 1e-10
        
        # Return log probabilities
        log_word_dist = torch.log(word_dist)
        
        return log_word_dist
    
    def get_topic_words(
        self,
        top_k: int = 10,
        vocab: Optional[list] = None
    ) -> list:
        """
        Get top words for each topic.
        
        Args:
            top_k: Number of top words per topic
            vocab: Vocabulary list (idx -> word)
            
        Returns:
            List of (topic_idx, [(word, prob), ...]) tuples
        """
        beta = self.get_beta().detach().cpu()
        
        topics = []
        for k in range(self.num_topics):
            topic_dist = beta[k]
            top_indices = torch.argsort(topic_dist, descending=True)[:top_k]
            
            if vocab is not None:
                words = [(vocab[idx.item()], topic_dist[idx].item()) 
                         for idx in top_indices]
            else:
                words = [(idx.item(), topic_dist[idx].item()) 
                         for idx in top_indices]
            
            topics.append((k, words))
        
        return topics
    
    def get_topic_embeddings(self) -> torch.Tensor:
        """
        Get topic embedding vectors.
        
        Returns:
            Topic embeddings, shape (K, E)
        """
        return self.topic_embeddings.detach().clone()

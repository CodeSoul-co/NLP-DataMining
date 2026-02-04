"""
Original ETM (Embedded Topic Model) - Original ETM implementation

This is the implementation of the original ETM paper, used as Baseline.
Unlike your ETM, original ETM:
1. Uses BOW as encoder input (not Qwen doc embedding)
2. Uses Word2Vec/GloVe word vectors (not Qwen word embedding)

Reference: Dieng et al., "Topic Modeling in Embedding Spaces", TACL 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base import NeuralTopicModel


class OriginalETM(NeuralTopicModel):
    """
    Original ETM implementation - used as Baseline

    Architecture:
        BOW -> Encoder -> theta (topic distribution)
        theta * beta -> word_dist
        beta = softmax(topic_embeddings @ word_embeddings.T)

    Differences from your ETM:
    - Encoder input: BOW (not Qwen doc embedding)
    - Word vectors: Word2Vec/GloVe or trainable (not Qwen word embedding)
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        embedding_dim: int = 300,  # Word2Vec/GloVe typically 300-dim
        hidden_dim: int = 800,
        dropout: float = 0.5,
        activation: str = 'softplus',
        word_embeddings: Optional[np.ndarray] = None,
        train_embeddings: bool = True,
        kl_weight: float = 1.0,
        dev_mode: bool = False,
        # Following parameters for interface compatibility
        doc_embedding_dim: int = None,
        word_embedding_dim: int = None,
        **kwargs
    ):
        """
        Initialize original ETM

        Args:
            vocab_size: Vocabulary size
            num_topics: Number of topics
            embedding_dim: Word vector dimension (Word2Vec=300, GloVe=300)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            activation: Activation function
            word_embeddings: Pre-trained word vectors (vocab_size, embedding_dim)
            train_embeddings: Whether to train word vectors
            kl_weight: KL divergence weight
            dev_mode: Debug mode
        """
        super().__init__()
        
        self._vocab_size = vocab_size
        self._num_topics = num_topics
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.kl_weight = kl_weight
        self.dev_mode = dev_mode
        self.train_embeddings = train_embeddings
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Word embedding matrix rho (V x E)
        if word_embeddings is not None:
            # Use pre-trained word vectors
            self.rho = nn.Parameter(
                torch.tensor(word_embeddings, dtype=torch.float32),
                requires_grad=train_embeddings
            )
            self.embedding_dim = word_embeddings.shape[1]
        else:
            # Trainable word vectors
            self.rho = nn.Parameter(
                torch.randn(vocab_size, embedding_dim) * 0.01,
                requires_grad=True
            )
        
        # Topic embedding matrix alpha (K x E)
        self.alphas = nn.Parameter(
            torch.randn(num_topics, self.embedding_dim) * 0.01
        )
        
        # Encoder: BOW -> theta
        # Original ETM uses BOW as input
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # VAE parameters
        self.mu_layer = nn.Linear(hidden_dim, num_topics)
        self.logvar_layer = nn.Linear(hidden_dim, num_topics)
        
        if self.dev_mode:
            print(f"[DEV] OriginalETM initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   embedding_dim={self.embedding_dim}")
            print(f"[DEV]   hidden_dim={hidden_dim}")
            print(f"[DEV]   train_embeddings={train_embeddings}")
    
    def _get_activation(self, act: str) -> nn.Module:
        """Get activation function"""
        activations = {
            'softplus': nn.Softplus(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
        }
        return activations.get(act, nn.Softplus())
    
    @property
    def num_topics(self) -> int:
        return self._num_topics
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode(self, bow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode BOW to topic distribution
        
        Args:
            bow: BOW matrix (batch, vocab_size)
            
        Returns:
            theta: Topic distribution (batch, num_topics)
            mu: Mean
            logvar: Log variance
        """
        # Normalize BOW
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # Encode
        hidden = self.encoder(bow_norm)
        
        # VAE parameters
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        # Softmax to get topic distribution
        theta = F.softmax(z, dim=-1)
        
        return theta, mu, logvar
    
    def get_beta(self) -> torch.Tensor:
        """
        Compute topic-word distribution beta
        
        beta = softmax(alpha @ rho.T)
        
        Returns:
            beta: (num_topics, vocab_size)
        """
        # alpha: (K, E), rho: (V, E)
        # logits: (K, V)
        logits = torch.mm(self.alphas, self.rho.t())
        beta = F.softmax(logits, dim=-1)
        return beta
    
    def decode(self, theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Decode topic distribution to word distribution
        
        Args:
            theta: Topic distribution (batch, num_topics)
            beta: Topic-word distribution (num_topics, vocab_size)
            
        Returns:
            log_word_dist: Log word distribution (batch, vocab_size)
        """
        word_dist = torch.mm(theta, beta)
        log_word_dist = torch.log(word_dist + 1e-10)
        return log_word_dist
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,  # For interface compatibility, not used
        bow: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            doc_embeddings: Document embedding (not used, only for interface compatibility)
            bow: BOW matrix (batch, vocab_size)
            
        Returns:
            Dict containing:
                - loss / total_loss: Total loss
                - recon_loss: Reconstruction loss
                - kl_loss: KL divergence loss
                - theta: Topic distribution
        """
        # Encode
        theta, mu, logvar = self.encode(bow)
        
        # Get beta
        beta = self.get_beta()
        
        # Decode
        log_word_dist = self.decode(theta, beta)
        
        # Reconstruction loss
        recon_loss = -(bow * log_word_dist).sum(dim=1).mean()
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'theta': theta,
            'beta': beta
        }
    
    def get_theta(self, bow: torch.Tensor = None, **kwargs) -> np.ndarray:
        """Get document-topic distribution"""
        if bow is None:
            raise ValueError("bow is required for OriginalETM.get_theta()")
        
        self.eval()
        with torch.no_grad():
            theta, _, _ = self.encode(bow)
        return theta.cpu().numpy()
    
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get topic words
        
        Args:
            vocab: Vocabulary
            top_k: Number of words to return per topic
            
        Returns:
            Topic words dictionary {topic_id: [word1, word2, ...]}
        """
        with torch.no_grad():
            beta = self.get_beta().cpu().numpy()
        
        topic_words = {}
        for k in range(self._num_topics):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        
        return topic_words
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'vocab_size': self._vocab_size,
            'num_topics': self._num_topics,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'kl_weight': self.kl_weight,
            'train_embeddings': self.train_embeddings
        }


def train_word2vec_embeddings(
    texts: List[str],
    vocab: List[str],
    embedding_dim: int = 300,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4
) -> np.ndarray:
    """
    Train Word2Vec embeddings using gensim
    
    Args:
        texts: Text list
        vocab: Vocabulary
        embedding_dim: Word vector dimension
        window: Window size
        min_count: Minimum word frequency
        workers: Number of worker threads
        
    Returns:
        embeddings: (vocab_size, embedding_dim)
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("gensim not installed. Install with: pip install gensim")
    
    print(f"Training Word2Vec embeddings (dim={embedding_dim})...")
    
    # Tokenize
    tokenized_texts = [text.split() for text in texts]
    
    # Train Word2Vec
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10
    )
    
    # Extract word vectors
    embeddings = np.zeros((len(vocab), embedding_dim))
    oov_count = 0
    
    for i, word in enumerate(vocab):
        if word in model.wv:
            embeddings[i] = model.wv[word]
        else:
            # OOV words use random initialization
            embeddings[i] = np.random.randn(embedding_dim) * 0.01
            oov_count += 1
    
    print(f"Word2Vec training complete. OOV words: {oov_count}/{len(vocab)}")
    return embeddings


def create_original_etm(
    vocab_size: int,
    num_topics: int = 20,
    embedding_dim: int = 300,
    word_embeddings: Optional[np.ndarray] = None,
    **kwargs
) -> OriginalETM:
    """
    Factory function to create original ETM model
    
    Args:
        vocab_size: Vocabulary size
        num_topics: Number of topics
        embedding_dim: Word vector dimension
        word_embeddings: Pre-trained word vectors
        **kwargs: Other parameters
        
    Returns:
        OriginalETM model instance
    """
    return OriginalETM(
        vocab_size=vocab_size,
        num_topics=num_topics,
        embedding_dim=embedding_dim,
        word_embeddings=word_embeddings,
        **kwargs
    )

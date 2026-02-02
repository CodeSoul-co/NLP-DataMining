"""
LDA (Latent Dirichlet Allocation) - Classic Probabilistic Topic Model

Provides two implementations:
1. SklearnLDA: Classic LDA implementation based on sklearn (recommended for Baseline)
2. NeuralLDA: VAE-based neural network LDA implementation

Both implementations maintain consistent interface with ETM for unified calling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Any, Tuple
import numpy as np

from .base import TraditionalTopicModel, NeuralTopicModel


# ============================================================================
# Sklearn LDA (Recommended for Baseline)
# ============================================================================

class SklearnLDA(TraditionalTopicModel):
    """
    Classic LDA implementation based on sklearn
    
    Features:
    1. Uses Variational Bayes inference, stable and reliable
    2. Does not require GPU
    3. Suitable as baseline for comparison
    4. Supports online learning
    
    Compatible with ETM interface for unified calling.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        alpha: float = None,  # Document-topic prior, None means auto-learn
        eta: float = None,    # Topic-word prior, None means auto-learn
        max_iter: int = 100,
        learning_method: str = 'batch',  # 'batch' or 'online'
        random_state: int = 42,
        n_jobs: int = -1,
        dev_mode: bool = False,
        # The following parameters are for interface compatibility, LDA does not use them
        doc_embedding_dim: int = None,
        word_embedding_dim: int = None,
        word_embeddings: Any = None,
        train_word_embeddings: bool = False,
        **kwargs
    ):
        from sklearn.decomposition import LatentDirichletAllocation
        
        self._vocab_size = vocab_size
        self._num_topics = num_topics
        self.dev_mode = dev_mode
        
        # Set prior parameters
        doc_topic_prior = alpha if alpha is not None else 1.0 / num_topics
        topic_word_prior = eta if eta is not None else 1.0 / num_topics
        
        # Create sklearn LDA model
        self.model = LatentDirichletAllocation(
            n_components=num_topics,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            max_iter=max_iter,
            learning_method=learning_method,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=1 if dev_mode else 0
        )
        
        # Store trained theta
        self._theta = None
        self._is_fitted = False
        
        if self.dev_mode:
            print(f"[DEV] SklearnLDA initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   learning_method={learning_method}")
    
    @property
    def num_topics(self) -> int:
        return self._num_topics
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def fit(
        self,
        bow_matrix: np.ndarray,
        **kwargs
    ) -> 'SklearnLDA':
        """
        Train LDA model
        
        Args:
            bow_matrix: BOW matrix (num_docs, vocab_size), can be sparse matrix
            
        Returns:
            self
        """
        if self.dev_mode:
            print(f"[DEV] Fitting LDA on {bow_matrix.shape[0]} documents...")
        
        self._theta = self.model.fit_transform(bow_matrix)
        self._is_fitted = True
        
        if self.dev_mode:
            print(f"[DEV] LDA fitting completed.")
            print(f"[DEV]   Perplexity: {self.model.perplexity(bow_matrix):.2f}")
        
        return self
    
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
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.transform(bow_matrix)
    
    def get_theta(self, bow_matrix: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Get document-topic distribution
        
        Args:
            bow_matrix: If provided, infer topic distribution for new documents; otherwise return training set distribution
            
        Returns:
            theta: (num_docs, num_topics)
        """
        if bow_matrix is not None:
            return self.transform(bow_matrix)
        if self._theta is None:
            raise RuntimeError("No theta available. Call fit() first.")
        return self._theta
    
    def get_beta(self) -> np.ndarray:
        """
        Get topic-word distribution
        
        Returns:
            beta: (num_topics, vocab_size) Normalized topic-word distribution
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # sklearn's components_ is unnormalized, need to normalize
        beta = self.model.components_
        beta = beta / beta.sum(axis=1, keepdims=True)
        return beta
    
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get topic words
        
        Args:
            vocab: Vocabulary list
            top_k: Number of words to return per topic
            
        Returns:
            Topic words dictionary {topic_id: [word1, word2, ...]}
        """
        beta = self.get_beta()
        
        topic_words = {}
        for k in range(self._num_topics):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        
        return topic_words
    
    def get_perplexity(self, bow_matrix: np.ndarray) -> float:
        """Compute perplexity"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.perplexity(bow_matrix)
    
    def save_model(self, path: str) -> None:
        """Save model"""
        import joblib
        joblib.dump({
            'model': self.model,
            'vocab_size': self._vocab_size,
            'num_topics': self._num_topics,
            'theta': self._theta,
            'is_fitted': self._is_fitted
        }, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'SklearnLDA':
        """Load model"""
        import joblib
        data = joblib.load(path)
        
        instance = cls(
            vocab_size=data['vocab_size'],
            num_topics=data['num_topics']
        )
        instance.model = data['model']
        instance._theta = data['theta']
        instance._is_fitted = data['is_fitted']
        
        return instance


# ============================================================================
# Neural LDA (VAE-based)
# ============================================================================

class LDAEncoder(nn.Module):
    """
    LDA encoder - infers topic distribution from BOW
    
    Unlike ETM, LDA directly uses BOW as input (does not require pretrained embeddings)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_topics: int = 20,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_topics = num_topics
        
        # BOW encoder
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # VAE parameters
        self.mu = nn.Linear(hidden_dim, num_topics)
        self.logvar = nn.Linear(hidden_dim, num_topics)
    
    def forward(self, bow: torch.Tensor) -> tuple:
        """
        Forward pass
        
        Args:
            bow: BOW matrix (batch, vocab_size)
            
        Returns:
            theta: Topic distribution (batch, num_topics)
            mu: Mean
            logvar: Log variance
        """
        # Normalize BOW
        bow_normalized = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # Encode
        hidden = self.encoder(bow_normalized)
        
        # VAE parameters
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        
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


class LDADecoder(nn.Module):
    """
    LDA decoder - generates word distribution from topic distribution
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        
        # Topic-word distribution (learnable parameter)
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))
    
    def get_beta(self) -> torch.Tensor:
        """Get normalized topic-word distribution"""
        return F.softmax(self.beta, dim=-1)
    
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            theta: Topic distribution (batch, num_topics)
            
        Returns:
            word_dist: Word distribution (batch, vocab_size)
        """
        beta = self.get_beta()  # (num_topics, vocab_size)
        word_dist = torch.mm(theta, beta)  # (batch, vocab_size)
        return word_dist


class NeuralLDA(NeuralTopicModel):
    """
    Neural LDA - VAE-based neural network LDA implementation
    
    Main features:
    1. Does not require pretrained embeddings
    2. Directly uses BOW as input
    3. Uses VAE framework for inference
    4. Supports GPU acceleration
    
    Maintains consistent interface with ETM for unified calling
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        hidden_dim: int = 256,
        encoder_dropout: float = 0.2,
        alpha: float = 0.1,  # Dirichlet prior parameter
        kl_weight: float = 1.0,
        dev_mode: bool = False,
        # The following parameters are for interface compatibility, LDA does not use them
        doc_embedding_dim: int = None,
        word_embedding_dim: int = None,
        word_embeddings: torch.Tensor = None,
        train_word_embeddings: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self._vocab_size = vocab_size
        self._num_topics = num_topics
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.kl_weight = kl_weight
        self.dev_mode = dev_mode
        
        # Encoder
        self.encoder = LDAEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_topics=num_topics,
            dropout=encoder_dropout
        )
        
        # Decoder
        self.decoder = LDADecoder(
            vocab_size=vocab_size,
            num_topics=num_topics
        )
        
        if self.dev_mode:
            print(f"[DEV] NeuralLDA initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   Note: NeuralLDA does not use embeddings")
    
    @property
    def num_topics(self) -> int:
        return self._num_topics
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,  # For interface compatibility, but LDA does not use
        bow: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            doc_embeddings: Document embeddings (LDA does not use, only for interface compatibility)
            bow: BOW matrix (batch, vocab_size)
            
        Returns:
            Dict containing:
                - loss / total_loss: Total loss
                - recon_loss: Reconstruction loss
                - kl_loss: KL divergence loss
                - theta: Topic distribution
        """
        # LDA directly uses BOW, ignores doc_embeddings
        theta, mu, logvar = self.encoder(bow)
        
        # Decode
        word_dist = self.decoder(theta)
        
        # Reconstruction loss
        bow_normalized = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        recon_loss = -torch.sum(bow_normalized * torch.log(word_dist + 1e-10), dim=1).mean()
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'theta': theta
        }
    
    def get_theta(self, bow: torch.Tensor = None, **kwargs) -> np.ndarray:
        """Get document-topic distribution"""
        if bow is None:
            raise ValueError("bow is required for NeuralLDA.get_theta()")
        
        self.eval()
        with torch.no_grad():
            theta, _, _ = self.encoder(bow)
        return theta.cpu().numpy()
    
    def get_beta(self) -> np.ndarray:
        """Get topic-word distribution"""
        with torch.no_grad():
            beta = self.decoder.get_beta()
        return beta.cpu().numpy()
    
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """
        Get topic words
        
        Args:
            vocab: Vocabulary list
            top_k: Number of words to return per topic
            
        Returns:
            Topic words dictionary {topic_id: [word1, word2, ...]}
        """
        beta = self.get_beta()
        
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
            'hidden_dim': self.hidden_dim,
            'alpha': self.alpha,
            'kl_weight': self.kl_weight
        }


# ============================================================================
# Compatibility Alias - Maintain backward compatibility
# ============================================================================

# Default LDA uses sklearn implementation
LDA = SklearnLDA


# ============================================================================
# Factory Functions
# ============================================================================

def create_lda(
    vocab_size: int,
    num_topics: int = 20,
    use_neural: bool = False,
    **kwargs
) -> TraditionalTopicModel:
    """
    Factory function to create LDA model
    
    Args:
        vocab_size: Vocabulary size
        num_topics: Number of topics
        use_neural: Whether to use neural network version
        **kwargs: Other parameters
        
    Returns:
        LDA model instance
    """
    if use_neural:
        return NeuralLDA(
            vocab_size=vocab_size,
            num_topics=num_topics,
            **kwargs
        )
    else:
        return SklearnLDA(
            vocab_size=vocab_size,
            num_topics=num_topics,
            **kwargs
        )

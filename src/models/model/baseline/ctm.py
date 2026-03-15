"""
CTM (Contextualized Topic Model)

Based on contextualized-topic-models reference implementation, adapted to project interface.

CTM combines pretrained language model embeddings with traditional topic models:
- ZeroShotTM: Uses only contextual embedding for inference
- CombinedTM: Uses both BOW and contextual embedding

References:
- Cross-lingual Contextualized Topic Models with Zero-shot Learning (Bianchi et al., 2020)
- Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence (Bianchi et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Any, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..base import NeuralTopicModel


# ============================================================================
# Inference Networks
# ============================================================================

class ContextualInferenceNetwork(nn.Module):
    """
    Contextual inference network - uses only contextual embedding
    
    Used for ZeroShotTM
    """
    
    def __init__(
        self,
        input_size: int,
        bert_size: int,
        n_components: int,
        hidden_sizes: Tuple[int, ...] = (100, 100),
        activation: str = 'softplus',
        dropout: float = 0.2,
        label_size: int = 0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.bert_size = bert_size
        self.n_components = n_components
        
        # Activation function
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Softplus()
        
        # Build inference network - uses only BERT embedding
        self.adapt_bert = nn.Linear(bert_size, hidden_sizes[0])
        
        # Hidden layers
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Output layers
        last_hidden = hidden_sizes[-1] if hidden_sizes else bert_size
        self.f_mu = nn.Linear(last_hidden, n_components)
        self.f_sigma = nn.Linear(last_hidden, n_components)
        
        # Batch normalization
        self.bn_mu = nn.BatchNorm1d(n_components, affine=False)
        self.bn_sigma = nn.BatchNorm1d(n_components, affine=False)
        
        self.drop = nn.Dropout(dropout)
    
    def forward(
        self,
        x_bow: torch.Tensor,
        x_bert: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x_bow: BOW (batch, vocab_size) - not used
            x_bert: BERT embedding (batch, bert_size)
            labels: Labels (optional)
            
        Returns:
            mu: Mean (batch, n_components)
            log_sigma: Log standard deviation (batch, n_components)
        """
        # Use only BERT embedding
        h = self.activation(self.adapt_bert(x_bert))
        h = self.drop(h)
        h = self.hidden(h)
        
        mu = self.bn_mu(self.f_mu(h))
        log_sigma = self.bn_sigma(self.f_sigma(h))
        
        return mu, log_sigma


class CombinedInferenceNetwork(nn.Module):
    """
    Combined inference network - uses both BOW and contextual embedding
    
    Used for CombinedTM
    """
    
    def __init__(
        self,
        input_size: int,
        bert_size: int,
        n_components: int,
        hidden_sizes: Tuple[int, ...] = (100, 100),
        activation: str = 'softplus',
        dropout: float = 0.2,
        label_size: int = 0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.bert_size = bert_size
        self.n_components = n_components
        
        # Activation function
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Softplus()
        
        # Build inference network - uses BOW + BERT embedding
        self.adapt_bert = nn.Linear(bert_size, hidden_sizes[0])
        self.adapt_bow = nn.Linear(input_size, hidden_sizes[0])
        
        # Fusion layer
        self.combine = nn.Linear(hidden_sizes[0] * 2, hidden_sizes[0])
        
        # Hidden layers
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Output layers
        last_hidden = hidden_sizes[-1] if hidden_sizes else hidden_sizes[0]
        self.f_mu = nn.Linear(last_hidden, n_components)
        self.f_sigma = nn.Linear(last_hidden, n_components)
        
        # Batch normalization
        self.bn_mu = nn.BatchNorm1d(n_components, affine=False)
        self.bn_sigma = nn.BatchNorm1d(n_components, affine=False)
        
        self.drop = nn.Dropout(dropout)
    
    def forward(
        self,
        x_bow: torch.Tensor,
        x_bert: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x_bow: BOW (batch, vocab_size)
            x_bert: BERT embedding (batch, bert_size)
            labels: Labels (optional)
            
        Returns:
            mu: Mean (batch, n_components)
            log_sigma: Log standard deviation (batch, n_components)
        """
        # Process BOW and BERT embedding
        h_bert = self.activation(self.adapt_bert(x_bert))
        h_bow = self.activation(self.adapt_bow(x_bow))
        
        # Fusion
        h = torch.cat([h_bert, h_bow], dim=1)
        h = self.activation(self.combine(h))
        h = self.drop(h)
        h = self.hidden(h)
        
        mu = self.bn_mu(self.f_mu(h))
        log_sigma = self.bn_sigma(self.f_sigma(h))
        
        return mu, log_sigma


# ============================================================================
# Decoder Network
# ============================================================================

class CTMDecoderNetwork(nn.Module):
    """
    CTM decoder network
    
    Supports two modes:
    - prodLDA: Product of Experts LDA
    - LDA: Standard LDA
    """
    
    def __init__(
        self,
        input_size: int,
        bert_size: int,
        inference_type: str = 'zeroshot',
        n_components: int = 10,
        model_type: str = 'prodLDA',
        hidden_sizes: Tuple[int, ...] = (100, 100),
        activation: str = 'softplus',
        dropout: float = 0.2,
        learn_priors: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.learn_priors = learn_priors
        
        # Select inference network
        if inference_type == 'zeroshot':
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components,
                hidden_sizes, activation, dropout
            )
        elif inference_type == 'combined':
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components,
                hidden_sizes, activation, dropout
            )
        else:
            raise ValueError(f"Unknown inference_type: {inference_type}")
        
        # Prior parameters
        topic_prior_mean = 0.0
        self.prior_mean = nn.Parameter(
            torch.tensor([topic_prior_mean] * n_components),
            requires_grad=learn_priors
        )
        
        topic_prior_variance = 1.0 - (1.0 / n_components)
        self.prior_variance = nn.Parameter(
            torch.tensor([topic_prior_variance] * n_components),
            requires_grad=learn_priors
        )
        
        # Topic-word distribution
        self.beta = nn.Parameter(torch.empty(n_components, input_size))
        nn.init.xavier_uniform_(self.beta)
        
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        self.drop_theta = nn.Dropout(dropout)
        
        # Store topic-word matrix
        self.topic_word_matrix = None
    
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(
        self,
        x_bow: torch.Tensor,
        x_bert: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass
        
        Returns:
            prior_mean, prior_variance, posterior_mu, posterior_sigma,
            posterior_log_sigma, word_dist
        """
        # Infer posterior
        posterior_mu, posterior_log_sigma = self.inf_net(x_bow, x_bert, labels)
        posterior_sigma = torch.exp(posterior_log_sigma)
        
        # Sample theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma),
            dim=1
        )
        theta = self.drop_theta(theta)
        
        # Compute word distribution
        if self.model_type == 'prodLDA':
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)),
                dim=1
            )
            self.topic_word_matrix = self.beta
        elif self.model_type == 'LDA':
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            self.topic_word_matrix = beta
            word_dist = torch.matmul(theta, beta)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
        
        return (
            self.prior_mean,
            self.prior_variance,
            posterior_mu,
            posterior_sigma,
            posterior_log_sigma,
            word_dist
        )
    
    def get_posterior(
        self,
        x_bow: torch.Tensor,
        x_bert: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior distribution parameters"""
        return self.inf_net(x_bow, x_bert, labels)
    
    def get_theta(
        self,
        x_bow: torch.Tensor,
        x_bert: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        """Get topic distribution"""
        with torch.no_grad():
            mu, log_sigma = self.get_posterior(x_bow, x_bert, labels)
            theta = F.softmax(self.reparameterize(mu, log_sigma), dim=1)
        return theta
    
    def sample(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        n_samples: int = 20
    ) -> torch.Tensor:
        """Sample multiple times and average"""
        with torch.no_grad():
            mu = mu.unsqueeze(0).repeat(n_samples, 1, 1)
            log_sigma = log_sigma.unsqueeze(0).repeat(n_samples, 1, 1)
            theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)
            return theta.mean(dim=0)


# ============================================================================
# CTM Model
# ============================================================================

class CTM(NeuralTopicModel):
    """
    Contextualized Topic Model
    
    Neural topic model combining pretrained language model embeddings with topic models.
    
    Features:
    1. Uses pretrained embeddings (e.g., Qwen) to capture semantic information
    2. Supports zero-shot cross-lingual topic modeling
    3. Typically has better topic coherence than traditional LDA
    
    Compatible with ETM interface for unified calling.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        doc_embedding_dim: int = 1024,
        hidden_sizes: Tuple[int, ...] = (100, 100),
        activation: str = 'softplus',
        dropout: float = 0.2,
        model_type: str = 'prodLDA',
        inference_type: str = 'zeroshot',
        learn_priors: bool = True,
        kl_weight: float = 1.0,
        dev_mode: bool = False,
        # The following parameters are for interface compatibility
        word_embedding_dim: int = None,
        word_embeddings: torch.Tensor = None,
        train_word_embeddings: bool = False,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size, num_topics=num_topics)
        
        self._vocab_size = vocab_size
        self._num_topics = num_topics
        self.doc_embedding_dim = doc_embedding_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.model_type = model_type
        self.inference_type = inference_type
        self.learn_priors = learn_priors
        self.kl_weight = kl_weight
        self.dev_mode = dev_mode
        
        # Decoder network
        self.decoder = CTMDecoderNetwork(
            input_size=vocab_size,
            bert_size=doc_embedding_dim,
            inference_type=inference_type,
            n_components=num_topics,
            model_type=model_type,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            learn_priors=learn_priors
        )
        
        # Store trained theta
        self._training_theta = None
        
        if self.dev_mode:
            print(f"[DEV] CTM initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   doc_embedding_dim={doc_embedding_dim}")
            print(f"[DEV]   inference_type={inference_type}")
            print(f"[DEV]   model_type={model_type}")
    
    @property
    def num_topics(self) -> int:
        return self._num_topics
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def _compute_loss(
        self,
        x_bow: torch.Tensor,
        word_dist: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_variance: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_variance: torch.Tensor,
        posterior_log_variance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute KL divergence and reconstruction loss"""
        # KL divergence
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)
        logvar_det_division = (
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        )
        kl_loss = 0.5 * (
            var_division + diff_term - self._num_topics + logvar_det_division
        )
        
        # Reconstruction loss
        recon_loss = -torch.sum(x_bow * torch.log(word_dist + 1e-10), dim=1)
        
        return kl_loss, recon_loss
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,
        bow: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            doc_embeddings: Document embeddings (batch, doc_embedding_dim)
            bow: BOW matrix (batch, vocab_size)
            
        Returns:
            Dict containing:
                - loss / total_loss: Total loss
                - recon_loss: Reconstruction loss
                - kl_loss: KL divergence loss
                - theta: Topic distribution
        """
        # Normalize BOW
        bow_normalized = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # Forward pass
        (
            prior_mean,
            prior_variance,
            posterior_mu,
            posterior_sigma,
            posterior_log_sigma,
            word_dist
        ) = self.decoder(bow_normalized, doc_embeddings)
        
        # Compute loss
        kl_loss, recon_loss = self._compute_loss(
            bow_normalized,
            word_dist,
            prior_mean,
            prior_variance,
            posterior_mu,
            posterior_sigma,
            posterior_log_sigma
        )
        
        # Total loss
        total_loss = (self.kl_weight * kl_loss + recon_loss).mean()
        
        # Get theta
        theta = F.softmax(
            self.decoder.reparameterize(posterior_mu, posterior_log_sigma),
            dim=1
        )
        
        return {
            'loss': total_loss,
            'total_loss': total_loss,
            'recon_loss': recon_loss.mean(),
            'kl_loss': kl_loss.mean(),
            'theta': theta
        }
    
    def get_theta(
        self,
        doc_embeddings: torch.Tensor = None,
        bow: torch.Tensor = None,
        n_samples: int = 20,
        **kwargs
    ) -> np.ndarray:
        """
        Get document-topic distribution
        
        Args:
            doc_embeddings: Document embeddings
            bow: BOW matrix
            n_samples: Number of samples
            
        Returns:
            theta: (num_docs, num_topics)
        """
        if doc_embeddings is None or bow is None:
            if self._training_theta is not None:
                return self._training_theta
            raise ValueError("doc_embeddings and bow are required")
        
        self.eval()
        with torch.no_grad():
            bow_normalized = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
            mu, log_sigma = self.decoder.get_posterior(bow_normalized, doc_embeddings)
            theta = self.decoder.sample(mu, log_sigma, n_samples)
        
        return theta.cpu().numpy()
    
    def get_beta(self) -> np.ndarray:
        """
        Get topic-word distribution
        
        Returns:
            beta: (num_topics, vocab_size)
        """
        with torch.no_grad():
            if self.model_type == 'prodLDA':
                beta = F.softmax(self.decoder.beta, dim=1)
            else:
                beta = F.softmax(
                    self.decoder.beta_batchnorm(self.decoder.beta),
                    dim=1
                )
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
            'doc_embedding_dim': self.doc_embedding_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'dropout': self.dropout,
            'model_type': self.model_type,
            'inference_type': self.inference_type,
            'learn_priors': self.learn_priors,
            'kl_weight': self.kl_weight
        }


# ============================================================================
# CTM Subclasses - For convenience
# ============================================================================

class ZeroShotTM(CTM):
    """
    ZeroShotTM - Zero-shot topic model
    
    Uses only contextual embedding for inference, does not depend on BOW.
    Suitable for cross-lingual topic modeling.
    """
    
    def __init__(self, **kwargs):
        kwargs['inference_type'] = 'zeroshot'
        super().__init__(**kwargs)


class CombinedTM(CTM):
    """
    CombinedTM - Combined topic model
    
    Uses both BOW and contextual embedding for inference.
    Typically has better performance than ZeroShotTM.
    """
    
    def __init__(self, **kwargs):
        kwargs['inference_type'] = 'combined'
        super().__init__(**kwargs)


# ============================================================================
# Factory Functions
# ============================================================================

def create_ctm(
    vocab_size: int,
    num_topics: int = 20,
    doc_embedding_dim: int = 1024,
    inference_type: str = 'zeroshot',
    **kwargs
) -> CTM:
    """
    Factory function to create CTM model
    
    Args:
        vocab_size: Vocabulary size
        num_topics: Number of topics
        doc_embedding_dim: Document embedding dimension
        inference_type: Inference type ('zeroshot' or 'combined')
        **kwargs: Other parameters
        
    Returns:
        CTM model instance
    """
    if inference_type == 'zeroshot':
        return ZeroShotTM(
            vocab_size=vocab_size,
            num_topics=num_topics,
            doc_embedding_dim=doc_embedding_dim,
            **kwargs
        )
    else:
        return CombinedTM(
            vocab_size=vocab_size,
            num_topics=num_topics,
            doc_embedding_dim=doc_embedding_dim,
            **kwargs
        )

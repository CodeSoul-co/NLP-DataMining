"""
Neural Topic Models (NTM)

Includes three variants:
1. NVDM (Neural Variational Document Model) - Basic VAE topic model
2. GSM (Gaussian Softmax Model) - NVDM with softmax constraint
3. ProdLDA (Product of Experts LDA) - Uses Logistic-Normal prior

Reference:
- NVDM: Miao et al., "Neural Variational Inference for Text Processing", ICML 2016
- ProdLDA: Srivastava & Sutton, "Autoencoding Variational Inference For Topic Models", ICLR 2017
- GSM: Miao et al., "Discovering Discrete Latent Topics with Neural Variational Inference", ICML 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Dict, Optional, List, Any, Tuple
import numpy as np

from ..base import NeuralTopicModel


def kld_normal(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """KL divergence to standard normal distribution.
    
    Args:
        mu: Mean, shape (batch_size, dim)
        log_sigma: Log standard deviation, shape (batch_size, dim)
    
    Returns:
        KL divergence, shape (batch_size,)
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


class NormalParameter(nn.Module):
    """Parameterizes a normal distribution with mean and log_sigma."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mu = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        self._reset_parameters()
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mu(h), self.log_sigma(h)
    
    def _reset_parameters(self):
        init.zeros_(self.log_sigma.weight)
        init.zeros_(self.log_sigma.bias)


class TopicDecoder(nn.Module):
    """Decodes topic distribution to word distribution."""
    
    def __init__(self, num_topics: int, vocab_size: int, bias: bool = True):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.topic_word = nn.Linear(num_topics, vocab_size, bias=bias)
    
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: Topic distribution, shape (batch_size, num_topics)
        
        Returns:
            Log probability over vocabulary, shape (batch_size, vocab_size)
        """
        return F.log_softmax(self.topic_word(theta), dim=-1)
    
    def get_beta(self) -> torch.Tensor:
        """Get topic-word distribution (beta matrix)."""
        return F.softmax(self.topic_word.weight.data.T, dim=-1)


class NVDM(NeuralTopicModel):
    """
    Neural Variational Document Model (NVDM)
    
    The first VAE-based topic model. Uses standard normal prior.
    Topic representation is unconstrained (can be positive or negative).
    
    Architecture:
        BOW -> Encoder -> (mu, log_sigma) -> z -> Decoder -> word_dist
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size, num_topics=num_topics)
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Encoder: BOW -> hidden -> (mu, log_sigma)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        # Normal parameters
        self.normal = NormalParameter(hidden_dim, num_topics)
        
        # Decoder: z -> word_dist
        self.decoder = TopicDecoder(num_topics, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, bow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode BOW to latent distribution parameters."""
        h = self.encoder(bow)
        mu, log_sigma = self.normal(h)
        return mu, log_sigma
    
    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to word distribution."""
        z = self.dropout(z)
        return self.decoder(z)
    
    def forward(
        self,
        bow: torch.Tensor,
        doc_emb: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            bow: Bag-of-words, shape (batch_size, vocab_size)
            doc_emb: Not used in NVDM
        
        Returns:
            Dict with theta, recon_loss, kl_loss
        """
        # Normalize BOW
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # Encode
        mu, log_sigma = self.encode(bow_norm)
        
        # Sample z
        z = self.reparameterize(mu, log_sigma)
        
        # Decode
        log_prob = self.decode(z)
        
        # Losses
        recon_loss = -(bow_norm * log_prob).sum(dim=-1)
        kl_loss = kld_normal(mu, log_sigma)
        
        return {
            'theta': z,  # Note: z is not a probability distribution in NVDM
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': mu,
            'log_sigma': log_sigma
        }
    
    def get_theta(self, bow: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get topic distribution for documents."""
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        mu, _ = self.encode(bow_norm)
        return mu  # Use mean as point estimate
    
    def get_beta(self) -> torch.Tensor:
        """Get topic-word distribution."""
        return self.decoder.get_beta()
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'vocab_size': self._vocab_size,
            'num_topics': self._num_topics,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate
        }
    
    def get_topic_words(self, vocab: List[str], top_k: int = 10) -> Dict[str, List[str]]:
        """Get top words for each topic."""
        beta = self.get_beta().cpu().numpy()
        topic_words = {}
        for k in range(self._num_topics):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        return topic_words


class GSM(NeuralTopicModel):
    """
    Gaussian Softmax Model (GSM)
    
    Improvement over NVDM: applies softmax to z to get proper probability distribution.
    This makes topic proportions interpretable (non-negative, sum to 1).
    
    Architecture:
        BOW -> Encoder -> (mu, log_sigma) -> z -> softmax(z) -> Decoder -> word_dist
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size, num_topics=num_topics)
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout)
        )
        
        # Normal parameters
        self.normal = NormalParameter(hidden_dim, num_topics)
        
        # Decoder
        self.decoder = TopicDecoder(num_topics, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, bow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode BOW to latent distribution parameters."""
        h = self.encoder(bow)
        mu, log_sigma = self.normal(h)
        return mu, log_sigma
    
    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        bow: torch.Tensor,
        doc_emb: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Normalize BOW
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # Encode
        mu, log_sigma = self.encode(bow_norm)
        
        # Sample z and apply softmax to get theta
        z = self.reparameterize(mu, log_sigma)
        theta = F.softmax(z, dim=-1)
        theta = self.dropout(theta)
        
        # Decode
        log_prob = self.decoder(theta)
        
        # Losses
        recon_loss = -(bow_norm * log_prob).sum(dim=-1)
        kl_loss = kld_normal(mu, log_sigma)
        
        return {
            'theta': theta,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': mu,
            'log_sigma': log_sigma
        }
    
    def get_theta(self, bow: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get topic distribution for documents."""
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        mu, _ = self.encode(bow_norm)
        return F.softmax(mu, dim=-1)
    
    def get_beta(self) -> torch.Tensor:
        """Get topic-word distribution."""
        return self.decoder.get_beta()
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'vocab_size': self._vocab_size,
            'num_topics': self._num_topics,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate
        }
    
    def get_topic_words(self, vocab: List[str], top_k: int = 10) -> Dict[str, List[str]]:
        """Get top words for each topic."""
        beta = self.get_beta().cpu().numpy()
        topic_words = {}
        for k in range(self._num_topics):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        return topic_words


class ProdLDA(NeuralTopicModel):
    """
    Product of Experts LDA (ProdLDA)
    
    Uses Logistic-Normal distribution to approximate Dirichlet prior.
    Key innovation: uses product of experts instead of mixture model.
    
    Architecture:
        BOW -> Encoder -> (mu, log_sigma) -> z -> softmax(z) -> Decoder -> word_dist
        
    The decoder uses softmax over both topics and vocabulary.
    
    Reference: Srivastava & Sutton, ICLR 2017
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        variance: float = 0.995,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size, num_topics=num_topics)
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.variance = variance
        
        # Encoder
        self.en1_fc = nn.Linear(vocab_size, hidden_dim)
        self.en2_fc = nn.Linear(hidden_dim, hidden_dim)
        self.en2_drop = nn.Dropout(dropout)
        
        # Latent parameters with batch norm
        self.mean_fc = nn.Linear(hidden_dim, num_topics)
        self.mean_bn = nn.BatchNorm1d(num_topics, affine=False)
        self.logvar_fc = nn.Linear(hidden_dim, num_topics)
        self.logvar_bn = nn.BatchNorm1d(num_topics, affine=False)
        
        # Dropout for z
        self.z_drop = nn.Dropout(dropout)
        
        # Decoder with batch norm
        self.decoder_fc = nn.Linear(num_topics, vocab_size)
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=False)
        
        # Prior parameters (standard Logistic-Normal)
        self.register_buffer('prior_mean', torch.zeros(1, num_topics))
        self.register_buffer('prior_var', torch.ones(1, num_topics) * variance)
        self.register_buffer('prior_logvar', torch.log(torch.ones(1, num_topics) * variance))
    
    def encode(self, bow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode BOW to posterior parameters."""
        en1 = F.softplus(self.en1_fc(bow))
        en2 = F.softplus(self.en2_fc(en1))
        en2 = self.en2_drop(en2)
        
        posterior_mean = self.mean_bn(self.mean_fc(en2))
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))
        
        return posterior_mean, posterior_logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, theta: torch.Tensor) -> torch.Tensor:
        """Decode topic distribution to word distribution."""
        return F.softmax(self.decoder_bn(self.decoder_fc(theta)), dim=-1)
    
    def forward(
        self,
        bow: torch.Tensor,
        doc_emb: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Normalize BOW
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        
        # Encode
        posterior_mean, posterior_logvar = self.encode(bow_norm)
        posterior_var = torch.exp(posterior_logvar)
        
        # Sample z and apply softmax
        z = self.reparameterize(posterior_mean, posterior_logvar)
        theta = F.softmax(z, dim=-1)
        theta = self.z_drop(theta)
        
        # Decode
        recon = self.decode(theta)
        
        # Reconstruction loss (negative log likelihood)
        recon_loss = -(bow_norm * (recon + 1e-10).log()).sum(dim=-1)
        
        # KL divergence with Logistic-Normal prior
        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_var)
        prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        
        kl_loss = 0.5 * ((var_division + diff_term + logvar_division).sum(dim=-1) - self.num_topics)
        
        return {
            'theta': theta,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'mu': posterior_mean,
            'logvar': posterior_logvar
        }
    
    def get_theta(self, bow: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get topic distribution for documents."""
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        mu, _ = self.encode(bow_norm)
        return F.softmax(mu, dim=-1)
    
    def get_beta(self) -> torch.Tensor:
        """Get topic-word distribution."""
        return F.softmax(self.decoder_fc.weight.data.T, dim=-1)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'vocab_size': self._vocab_size,
            'num_topics': self._num_topics,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'variance': self.variance
        }
    
    def get_topic_words(self, vocab: List[str], top_k: int = 10) -> Dict[str, List[str]]:
        """Get top words for each topic."""
        beta = self.get_beta().cpu().numpy()
        topic_words = {}
        for k in range(self._num_topics):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        return topic_words


# Factory functions
def create_nvdm(vocab_size: int, num_topics: int = 20, **kwargs) -> NVDM:
    """Create NVDM model."""
    return NVDM(vocab_size=vocab_size, num_topics=num_topics, **kwargs)


def create_gsm(vocab_size: int, num_topics: int = 20, **kwargs) -> GSM:
    """Create GSM model."""
    return GSM(vocab_size=vocab_size, num_topics=num_topics, **kwargs)


def create_prodlda(vocab_size: int, num_topics: int = 20, **kwargs) -> ProdLDA:
    """Create ProdLDA model."""
    return ProdLDA(vocab_size=vocab_size, num_topics=num_topics, **kwargs)

"""
ETM Encoder: Maps document embeddings to topic distributions.

Key modification from original ETM:
- Input: Qwen document embedding (1024-dim) instead of BOW
- Output: Topic distribution theta (K-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ETMEncoder(nn.Module):
    """
    Variational encoder for ETM.
    
    Maps document embeddings to topic distribution parameters (mu, logvar).
    Uses reparameterization trick for sampling.
    
    Architecture:
        doc_embedding (D) -> hidden -> hidden -> (mu, logvar) -> z -> theta
    """
    
    def __init__(
        self,
        input_dim: int,           # Qwen embedding dimension (1024)
        hidden_dim: int,          # Hidden layer dimension
        num_topics: int,          # Number of topics (K)
        dropout: float = 0.2,     # Dropout rate
        activation: str = 'relu'  # Activation function
    ):
        """
        Initialize encoder.
        
        Args:
            input_dim: Input dimension (Qwen embedding size)
            hidden_dim: Hidden layer dimension
            num_topics: Number of topics
            dropout: Dropout rate
            activation: Activation function name
        """
        super(ETMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_topics = num_topics
        self.dropout_rate = dropout
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Inference network: doc_embedding -> hidden -> (mu, logvar)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers for mu and logvar
        self.mu_layer = nn.Linear(hidden_dim, num_topics)
        self.logvar_layer = nn.Linear(hidden_dim, num_topics)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'softplus': nn.Softplus(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,
        compute_kl: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder.
        
        Args:
            doc_embeddings: Document embeddings, shape (batch_size, input_dim)
            compute_kl: Whether to compute KL divergence
            
        Returns:
            theta: Topic distribution, shape (batch_size, num_topics)
            z: Latent representation, shape (batch_size, num_topics)
            kl_loss: KL divergence loss (scalar) or None
        """
        # First hidden layer
        h = self.fc1(doc_embeddings)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Second hidden layer
        h = self.fc2(h)
        h = self.bn2(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Get mu and logvar
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        # Reparameterization trick
        z = self._reparameterize(mu, logvar)
        
        # Convert to topic distribution
        theta = F.softmax(z, dim=-1)
        
        # Compute KL divergence
        kl_loss = None
        if compute_kl:
            kl_loss = self._compute_kl(mu, logvar)
        
        return theta, z, kl_loss
    
    def _reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def _compute_kl(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from standard normal.
        
        KL(q(z|x) || p(z)) where p(z) = N(0, I)
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            KL divergence (scalar, averaged over batch)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()
    
    def get_theta(
        self,
        doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Get topic distribution without computing KL.
        
        Args:
            doc_embeddings: Document embeddings
            
        Returns:
            theta: Topic distribution
        """
        theta, _, _ = self.forward(doc_embeddings, compute_kl=False)
        return theta

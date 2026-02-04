"""
DTM (Dynamic Topic Model) - Dynamic topic model

Topic model supporting time series, can track topic evolution over time.

TODO: This is a pseudocode framework, needs further implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np


class DTMEncoder(nn.Module):
    """
    DTM Encoder - Maps document embedding to topic distribution
    
    Similar to ETM encoder, but with added temporal information processing
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_topics: int = 20,
        time_slices: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_topics = num_topics
        self.time_slices = time_slices
        
        # Document encoder
        self.doc_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal encoding (learnable time embedding)
        self.time_embedding = nn.Embedding(time_slices, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output layers (VAE style)
        self.mu = nn.Linear(hidden_dim, num_topics)
        self.logvar = nn.Linear(hidden_dim, num_topics)
    
    def forward(
        self, 
        doc_embedding: torch.Tensor,
        time_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            doc_embedding: Document embedding (batch, input_dim)
            time_index: Time index (batch,)
            
        Returns:
            theta: Topic distribution (batch, num_topics)
            mu: Mean
            logvar: Log variance
        """
        # Encode document
        doc_hidden = self.doc_encoder(doc_embedding)
        
        # Get time embedding
        time_hidden = self.time_embedding(time_index)
        
        # Fusion
        combined = torch.cat([doc_hidden, time_hidden], dim=-1)
        hidden = F.relu(self.fusion(combined))
        
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


class DTMDecoder(nn.Module):
    """
    DTM Decoder - Reconstruct word distribution from topic distribution
    
    Topic-word distribution changes over time
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        time_slices: int = 10,
        embedding_dim: int = 1024,
        word_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.time_slices = time_slices
        self.embedding_dim = embedding_dim
        
        # Word embeddings
        if word_embeddings is not None:
            self.word_embeddings = nn.Parameter(word_embeddings, requires_grad=False)
        else:
            self.word_embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        
        # Time-dependent topic embeddings
        # Each time slice has its own topic vectors
        self.topic_embeddings = nn.Parameter(
            torch.randn(time_slices, num_topics, embedding_dim)
        )
        
        # Topic evolution network (optional: model smooth topic changes over time)
        self.topic_evolution = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
    
    def get_beta(self, time_index: int = None) -> torch.Tensor:
        """
        Get topic-word distribution
        
        Args:
            time_index: Time index, None returns beta for all time slices
            
        Returns:
            beta: (num_topics, vocab_size) or (time_slices, num_topics, vocab_size)
        """
        if time_index is not None:
            # Topic embedding for specific time
            topic_emb = self.topic_embeddings[time_index]  # (num_topics, embedding_dim)
            # Compute similarity with word embeddings
            beta = torch.mm(topic_emb, self.word_embeddings.t())  # (num_topics, vocab_size)
            beta = F.softmax(beta, dim=-1)
            return beta
        else:
            # Beta for all time slices
            betas = []
            for t in range(self.time_slices):
                topic_emb = self.topic_embeddings[t]
                beta = torch.mm(topic_emb, self.word_embeddings.t())
                beta = F.softmax(beta, dim=-1)
                betas.append(beta)
            return torch.stack(betas, dim=0)  # (time_slices, num_topics, vocab_size)
    
    def forward(
        self,
        theta: torch.Tensor,
        time_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            theta: Topic distribution (batch, num_topics)
            time_index: Time index (batch,)
            
        Returns:
            word_dist: Word distribution (batch, vocab_size)
        """
        batch_size = theta.size(0)
        word_dists = []
        
        for i in range(batch_size):
            t = time_index[i].item()
            beta = self.get_beta(t)  # (num_topics, vocab_size)
            word_dist = torch.mm(theta[i:i+1], beta)  # (1, vocab_size)
            word_dists.append(word_dist)
        
        return torch.cat(word_dists, dim=0)


class DTM(nn.Module):
    """
    Dynamic Topic Model - Dynamic topic model
    
    Main features:
    1. Supports time series data
    2. Topics evolve over time
    3. Can track topic change trends
    
    Maintains consistent interface with ETM for unified calling
    
    TODO: Improve the following features
    - [ ] Smooth constraints for topic evolution
    - [ ] Time series prediction
    - [ ] Topic lifecycle analysis
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        time_slices: int = 10,
        doc_embedding_dim: int = 1024,
        word_embedding_dim: int = 1024,
        hidden_dim: int = 512,
        encoder_dropout: float = 0.2,
        word_embeddings: Optional[torch.Tensor] = None,
        train_word_embeddings: bool = False,
        kl_weight: float = 0.5,
        evolution_weight: float = 0.1,  # Topic evolution smoothness constraint weight
        dev_mode: bool = False,
        **kwargs  # Accept additional parameters for interface compatibility
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.time_slices = time_slices
        self.doc_embedding_dim = doc_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.evolution_weight = evolution_weight
        self.dev_mode = dev_mode
        
        # Encoder
        self.encoder = DTMEncoder(
            input_dim=doc_embedding_dim,
            hidden_dim=hidden_dim,
            num_topics=num_topics,
            time_slices=time_slices,
            dropout=encoder_dropout
        )
        
        # Decoder
        self.decoder = DTMDecoder(
            vocab_size=vocab_size,
            num_topics=num_topics,
            time_slices=time_slices,
            embedding_dim=word_embedding_dim,
            word_embeddings=word_embeddings
        )
        
        if self.dev_mode:
            print(f"[DEV] DTM initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   time_slices={time_slices}")
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,
        bow: torch.Tensor,
        time_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            doc_embeddings: Document embedding (batch, doc_embedding_dim)
            bow: BOW matrix (batch, vocab_size)
            time_indices: Time indices (batch,), if None assumes all documents at same time
            
        Returns:
            Dict containing:
                - loss: Total loss
                - recon_loss: Reconstruction loss
                - kl_loss: KL divergence loss
                - evolution_loss: Topic evolution smoothness loss
                - theta: Topic distribution
        """
        batch_size = doc_embeddings.size(0)
        
        # If no time indices, default to 0
        if time_indices is None:
            time_indices = torch.zeros(batch_size, dtype=torch.long, device=doc_embeddings.device)
        
        # Encode
        theta, mu, logvar = self.encoder(doc_embeddings, time_indices)
        
        # Decode
        word_dist = self.decoder(theta, time_indices)
        
        # Reconstruction loss (negative log likelihood)
        bow_normalized = bow / (bow.sum(dim=1, keepdim=True) + 1e-10)
        recon_loss = -torch.sum(bow_normalized * torch.log(word_dist + 1e-10), dim=1).mean()
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Topic evolution smoothness loss (adjacent time slices should have similar topics)
        evolution_loss = self._compute_evolution_loss()
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss + self.evolution_weight * evolution_loss
        
        return {
            'total_loss': total_loss,
            'loss': total_loss,  # Compatibility
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'evolution_loss': evolution_loss,
            'theta': theta
        }
    
    def _compute_evolution_loss(self) -> torch.Tensor:
        """
        Compute topic evolution smoothness loss
        
        Encourage similar topic embeddings for adjacent time slices
        """
        topic_emb = self.decoder.topic_embeddings  # (time_slices, num_topics, embedding_dim)
        
        if self.time_slices < 2:
            return torch.tensor(0.0, device=topic_emb.device)
        
        # Compute difference between adjacent time slices
        diff = topic_emb[1:] - topic_emb[:-1]  # (time_slices-1, num_topics, embedding_dim)
        evolution_loss = torch.mean(diff.pow(2))
        
        return evolution_loss
    
    def get_beta(self, time_index: int = None) -> torch.Tensor:
        """Get topic-word distribution"""
        return self.decoder.get_beta(time_index)
    
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10,
        time_index: int = None
    ) -> Dict[str, List[str]]:
        """
        Get topic words
        
        Args:
            vocab: Vocabulary
            top_k: Number of words to return per topic
            time_index: Time index, None returns last time slice
            
        Returns:
            Topic words dictionary {topic_id: [word1, word2, ...]}
        """
        if time_index is None:
            time_index = self.time_slices - 1
        
        beta = self.get_beta(time_index)  # (num_topics, vocab_size)
        
        topic_words = {}
        for k in range(self.num_topics):
            top_indices = torch.topk(beta[k], top_k).indices.cpu().numpy()
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        
        return topic_words
    
    def get_topic_evolution(
        self,
        vocab: List[str],
        topic_id: int,
        top_k: int = 10
    ) -> Dict[int, List[str]]:
        """
        Get evolution of specific topic over time
        
        Args:
            vocab: Vocabulary
            topic_id: Topic ID
            top_k: Number of words to return per time slice
            
        Returns:
            {time_index: [word1, word2, ...]}
        """
        evolution = {}
        
        for t in range(self.time_slices):
            beta = self.get_beta(t)  # (num_topics, vocab_size)
            top_indices = torch.topk(beta[topic_id], top_k).indices.cpu().numpy()
            evolution[t] = [vocab[i] for i in top_indices]
        
        return evolution


# ============================================================================
# Factory functions - convenient for calling from registry
# ============================================================================

def create_dtm(
    vocab_size: int,
    num_topics: int = 20,
    word_embeddings: Optional[torch.Tensor] = None,
    **kwargs
) -> DTM:
    """
    Factory function to create DTM model
    
    Args:
        vocab_size: Vocabulary size
        num_topics: Number of topics
        word_embeddings: Pre-trained word embeddings
        **kwargs: Other parameters
        
    Returns:
        DTM model instance
    """
    return DTM(
        vocab_size=vocab_size,
        num_topics=num_topics,
        word_embeddings=word_embeddings,
        **kwargs
    )

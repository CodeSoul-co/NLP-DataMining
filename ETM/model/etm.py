"""
ETM (Embedded Topic Model) - Modified for Qwen Embedding Input

Key modifications from original ETM:
1. Encoder input: Qwen document embeddings (1024-dim) instead of BOW
2. Decoder: Uses Qwen word embeddings as semantic basis
3. Loss: BOW reconstruction + KL divergence

This separates:
- Semantic understanding (Qwen embeddings)
- Structure modeling (ETM topic distribution)
- Interpretability (word-based topic descriptions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

from .encoder import ETMEncoder
from .decoder import ETMDecoder


class ETM(nn.Module):
    """
    Embedded Topic Model with Qwen embedding input.
    
    Architecture:
        doc_embedding (Qwen) -> Encoder -> theta (topic dist)
        theta -> Decoder (with word embeddings) -> word_dist
        Loss = Reconstruction(word_dist, BOW) + KL(theta)
    
    Key outputs:
        - theta: Document-topic distribution (D x K)
        - beta: Topic-word distribution (K x V)
        - topic_embeddings: Topic vectors in embedding space (K x E)
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        doc_embedding_dim: int = 1024,      # Qwen embedding dimension
        word_embedding_dim: int = 1024,     # Word embedding dimension
        hidden_dim: int = 512,              # Encoder hidden dimension
        encoder_dropout: float = 0.2,
        encoder_activation: str = 'relu',
        word_embeddings: Optional[torch.Tensor] = None,
        train_word_embeddings: bool = False,
        kl_weight: float = 0.5,            # Weight for KL divergence loss
        dev_mode: bool = False
    ):
        """
        Initialize ETM model.
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics (K)
            doc_embedding_dim: Dimension of document embeddings from Qwen
            word_embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension for encoder
            encoder_dropout: Dropout rate for encoder
            encoder_activation: Activation function for encoder
            word_embeddings: Pre-trained word embeddings (V x E)
            train_word_embeddings: Whether to fine-tune word embeddings
            dev_mode: Print debug information
        """
        super(ETM, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.doc_embedding_dim = doc_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.dev_mode = dev_mode
        
        # Encoder: doc_embedding -> theta
        self.encoder = ETMEncoder(
            input_dim=doc_embedding_dim,
            hidden_dim=hidden_dim,
            num_topics=num_topics,
            dropout=encoder_dropout,
            activation=encoder_activation
        )
        
        # Decoder: theta -> word_dist
        self.decoder = ETMDecoder(
            vocab_size=vocab_size,
            num_topics=num_topics,
            embedding_dim=word_embedding_dim,
            word_embeddings=word_embeddings,
            train_embeddings=train_word_embeddings
        )
        
        if self.dev_mode:
            print(f"[DEV] ETM initialized:")
            print(f"[DEV]   vocab_size={vocab_size}")
            print(f"[DEV]   num_topics={num_topics}")
            print(f"[DEV]   doc_embedding_dim={doc_embedding_dim}")
            print(f"[DEV]   word_embedding_dim={word_embedding_dim}")
            print(f"[DEV]   hidden_dim={hidden_dim}")
    
    def forward(
        self,
        doc_embeddings: torch.Tensor,
        bow_targets: torch.Tensor,
        compute_loss: bool = True,
        kl_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ETM.
        
        Args:
            doc_embeddings: Document embeddings from Qwen, shape (batch, doc_emb_dim)
            bow_targets: BOW target for reconstruction, shape (batch, vocab_size)
            compute_loss: Whether to compute loss
            kl_weight: Weight for KL divergence loss
            
        Returns:
            Dictionary containing:
                - theta: Topic distribution (batch, K)
                - recon_loss: Reconstruction loss (scalar)
                - kl_loss: KL divergence loss (scalar)
                - total_loss: Total loss (scalar)
        """
        # Encode: doc_embedding -> theta
        theta, z, kl_theta_loss = self.encoder(doc_embeddings, compute_kl=True)
        
        # Get topic-word distribution
        beta = self.decoder.get_beta()
        
        # Decode: theta -> word distribution
        log_word_dist = self.decoder(theta, beta)
        word_dist = torch.exp(log_word_dist)
        
        output = {
            'theta': theta,
            'z': z,
            'beta': beta,
            'log_word_dist': log_word_dist,
            'word_dist': word_dist
        }
        
        if compute_loss:
            # BOW is already normalized in ETMDataset, use directly
            # bow_targets should sum to 1 for each document
            
            # 1. Reconstruction loss: negative log likelihood (cross-entropy)
            # recon_loss = -sum(p_true * log(p_pred)) where p_true is normalized BOW
            recon_loss = -torch.sum(bow_targets * log_word_dist, dim=-1).mean()
            
            # 2. KL divergence for variational inference
            # Use free bits strategy: ensure minimum KL per dimension to prevent posterior collapse
            # This encourages the model to use the latent space
            free_bits = 0.5  # Minimum KL per topic dimension
            kl_per_dim = kl_theta_loss  # Already averaged over batch
            kl_loss = kl_weight * torch.clamp(kl_per_dim, min=free_bits)
            
            output['recon_loss'] = recon_loss
            output['kl_theta_loss'] = kl_theta_loss
            output['kl_loss'] = kl_loss
            output['total_loss'] = recon_loss + kl_loss
        
        if self.dev_mode and not hasattr(self, '_forward_logged'):
            print(f"[DEV] Forward pass:")
            print(f"[DEV]   doc_embeddings shape: {doc_embeddings.shape}")
            print(f"[DEV]   bow_targets shape: {bow_targets.shape}")
            print(f"[DEV]   theta shape: {theta.shape}")
            print(f"[DEV]   beta shape: {beta.shape}")
            print(f"[DEV]   word_dist shape: {word_dist.shape}")
            self._forward_logged = True
        
        return output
    
    def get_theta(
        self,
        doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Get topic distribution for documents.
        
        Args:
            doc_embeddings: Document embeddings
            
        Returns:
            theta: Topic distribution (batch, K)
        """
        return self.encoder.get_theta(doc_embeddings)
    
    def get_beta(self) -> torch.Tensor:
        """
        Get topic-word distribution.
        
        Returns:
            beta: Topic-word distribution (K, V)
        """
        return self.decoder.get_beta()
    
    def get_topic_embeddings(self) -> torch.Tensor:
        """
        Get topic embedding vectors.
        
        Returns:
            Topic embeddings (K, E)
        """
        return self.decoder.get_topic_embeddings()
    
    def get_topic_words(
        self,
        top_k: int = 10,
        vocab: Optional[List[str]] = None
    ) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """
        Get top words for each topic.
        
        Args:
            top_k: Number of top words per topic
            vocab: Vocabulary list
            
        Returns:
            List of (topic_idx, [(word, prob), ...])
        """
        return self.decoder.get_topic_words(top_k, vocab)
    
    def get_all_outputs(
        self,
        doc_embeddings: torch.Tensor,
        bow_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get all important output matrices for analysis.
        
        Args:
            doc_embeddings: All document embeddings
            bow_targets: BOW targets (optional, for loss computation)
            
        Returns:
            Dictionary with numpy arrays:
                - theta: Document-topic distribution (D x K)
                - beta: Topic-word distribution (K x V)
                - topic_embeddings: Topic vectors (K x E)
        """
        self.eval()
        with torch.no_grad():
            theta = self.get_theta(doc_embeddings)
            beta = self.get_beta()
            topic_emb = self.get_topic_embeddings()
        
        return {
            'theta': theta.cpu().numpy(),
            'beta': beta.cpu().numpy(),
            'topic_embeddings': topic_emb.cpu().numpy()
        }
    
    def compute_perplexity(
        self,
        doc_embeddings: torch.Tensor,
        bow_targets: torch.Tensor
    ) -> float:
        """
        Compute perplexity on given data.
        
        Args:
            doc_embeddings: Document embeddings
            bow_targets: BOW targets
            
        Returns:
            Perplexity value
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(doc_embeddings, bow_targets, compute_loss=True)
            
            # Perplexity = exp(avg negative log likelihood per word)
            total_words = bow_targets.sum()
            avg_nll = output['recon_loss'] * bow_targets.shape[0] / total_words
            perplexity = torch.exp(avg_nll).item()
        
        return perplexity
        
    def compute_topic_coherence(
        self,
        bow_matrix: torch.Tensor,
        top_k: int = 10
    ) -> float:
        """
        Compute topic coherence using normalized pointwise mutual information (NPMI).
        
        Args:
            bow_matrix: BOW matrix for corpus, shape (num_docs, vocab_size)
            top_k: Number of top words per topic to consider
            
        Returns:
            Average topic coherence score
        """
        self.eval()
        with torch.no_grad():
            # Get topic-word distribution
            beta = self.get_beta().cpu().numpy()
            
            # Get top words for each topic
            top_words_indices = np.argsort(-beta, axis=1)[:, :top_k]
            
            # Convert BOW to document-term co-occurrence
            bow_np = bow_matrix.cpu().numpy()
            doc_term = (bow_np > 0).astype(np.float32)
            doc_count = doc_term.shape[0]
            
            # Compute coherence for each topic
            coherence_scores = []
            for topic_idx in range(self.num_topics):
                topic_words = top_words_indices[topic_idx]
                score = 0.0
                num_pairs = 0
                
                # Compute pairwise NPMI for top words
                for i in range(len(topic_words)):
                    for j in range(i+1, len(topic_words)):
                        word_i, word_j = topic_words[i], topic_words[j]
                        
                        # Count documents containing word i
                        count_i = doc_term[:, word_i].sum()
                        # Count documents containing word j
                        count_j = doc_term[:, word_j].sum()
                        # Count documents containing both words
                        count_ij = np.logical_and(doc_term[:, word_i], doc_term[:, word_j]).sum()
                        
                        # Avoid division by zero
                        if count_ij > 0:
                            # Compute PMI: log(p(i,j) / (p(i) * p(j)))
                            pmi = np.log((count_ij * doc_count) / (count_i * count_j))
                            # Normalize to [-1, 1]
                            npmi = pmi / (-np.log(count_ij / doc_count))
                            score += npmi
                            num_pairs += 1
                
                # Average coherence for this topic
                if num_pairs > 0:
                    coherence_scores.append(score / num_pairs)
            
            # Average coherence across all topics
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return avg_coherence
    
    def save_model(self, path: str) -> None:
        """Save model state dict"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'num_topics': self.num_topics,
                'doc_embedding_dim': self.doc_embedding_dim,
                'word_embedding_dim': self.word_embedding_dim,
                'hidden_dim': self.hidden_dim
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device) -> 'ETM':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            vocab_size=config['vocab_size'],
            num_topics=config['num_topics'],
            doc_embedding_dim=config['doc_embedding_dim'],
            word_embedding_dim=config['word_embedding_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model

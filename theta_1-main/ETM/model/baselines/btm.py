"""
BTM (Biterm Topic Model)

A topic model designed for short texts by modeling word co-occurrence patterns (biterms)
instead of document-word distributions. This addresses the sparsity problem in short texts.

Key Features:
- Models word pairs (biterms) instead of documents
- Better for short texts (tweets, titles, etc.)
- Uses Gibbs sampling for inference

Reference:
- Yan et al., "A Biterm Topic Model for Short Texts", WWW 2013

Note: This is a Python implementation based on the reference code in _reference/BTM/
"""

import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from collections import defaultdict
import random

from ..base import TraditionalTopicModel


class BTM(TraditionalTopicModel):
    """
    Biterm Topic Model for Short Texts
    
    Instead of modeling document-word distributions, BTM models biterms
    (unordered word pairs) that co-occur in the same context window.
    
    Attributes:
        num_topics: Number of topics
        alpha: Dirichlet prior for topic distribution
        beta: Dirichlet prior for word distribution
        n_iter: Number of Gibbs sampling iterations
        window_size: Context window size for extracting biterms
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int = 20,
        alpha: float = 1.0,
        beta: float = 0.01,
        n_iter: int = 100,
        window_size: int = 15,
        max_doc_words: int = 50,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize BTM model.
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics
            alpha: Dirichlet prior for topic distribution (symmetric)
            beta: Dirichlet prior for word distribution (symmetric)
            n_iter: Number of Gibbs sampling iterations
            window_size: Maximum distance between words in a biterm
            max_doc_words: Max words per document for biterm extraction
                          (prevents combinatorial explosion on long docs)
            random_state: Random seed
        """
        super().__init__(vocab_size=vocab_size, num_topics=num_topics)
        
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.window_size = window_size
        self.max_doc_words = max_doc_words
        self.random_state = random_state
        
        # Model parameters
        self._n_wz = None  # Word-topic count matrix (V x K)
        self._n_z = None   # Topic count vector (K,)
        self._theta = None  # Global topic distribution
        self._phi = None    # Topic-word distribution (K x V)
        
        # Biterms and their topic assignments
        self._biterms = []
        self._biterm_topics = []
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    def _extract_biterms(self, bow_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract biterms from documents.
        
        Args:
            bow_matrix: BOW matrix, shape (num_docs, vocab_size)
        
        Returns:
            List of biterms (word_i, word_j)
        """
        biterms = []
        
        for doc in bow_matrix:
            # Get words in document
            words = []
            for word_id, count in enumerate(doc):
                words.extend([word_id] * int(count))
            
            # Cap words per document to prevent combinatorial explosion
            if len(words) > self.max_doc_words:
                words = list(np.random.choice(words, self.max_doc_words, replace=False))
            
            # Extract biterms within window
            for i in range(len(words)):
                for j in range(i + 1, min(i + self.window_size + 1, len(words))):
                    biterms.append((words[i], words[j]))
        
        return biterms
    
    def _sample_topic(self, w1: int, w2: int) -> int:
        """
        Sample a topic for a biterm using Gibbs sampling (vectorized over topics).
        
        Args:
            w1: First word index
            w2: Second word index
        
        Returns:
            Sampled topic index
        """
        # Vectorized over all K topics at once
        n_z_sum = self._n_z.sum()
        denom = self._n_z + self.vocab_size * self.beta
        
        p = ((self._n_z + self.alpha) / (n_z_sum + self.num_topics * self.alpha)
             * (self._n_wz[w1] + self.beta) / denom
             * (self._n_wz[w2] + self.beta) / denom)
        
        p /= p.sum()
        return np.random.choice(self.num_topics, p=p)
    
    def fit(
        self,
        bow_matrix: np.ndarray,
        vocab: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs
    ) -> 'BTM':
        """
        Fit BTM model using Gibbs sampling.
        
        Args:
            bow_matrix: BOW matrix, shape (num_docs, vocab_size)
            vocab: Vocabulary list (optional)
            verbose: Print progress
        
        Returns:
            self
        """
        # Extract biterms
        if verbose:
            print("Extracting biterms...")
        self._biterms = self._extract_biterms(bow_matrix)
        n_biterms = len(self._biterms)
        
        if verbose:
            print(f"Extracted {n_biterms} biterms")
        
        # Initialize count matrices
        self._n_wz = np.zeros((self.vocab_size, self.num_topics))
        self._n_z = np.zeros(self.num_topics)
        
        # Convert biterms to numpy arrays for fast indexing
        biterm_arr = np.array(self._biterms, dtype=np.int32)  # (N, 2)
        w1_arr = biterm_arr[:, 0]
        w2_arr = biterm_arr[:, 1]
        
        # Random initialization (vectorized)
        self._biterm_topics = np.random.randint(0, self.num_topics, size=n_biterms)
        for b_idx in range(n_biterms):
            z = self._biterm_topics[b_idx]
            self._n_wz[w1_arr[b_idx], z] += 1
            self._n_wz[w2_arr[b_idx], z] += 1
            self._n_z[z] += 2
        
        # Gibbs sampling
        if verbose:
            print(f"Running Gibbs sampling for {self.n_iter} iterations...")
        
        for iteration in range(self.n_iter):
            for b_idx in range(n_biterms):
                w1 = w1_arr[b_idx]
                w2 = w2_arr[b_idx]
                old_z = self._biterm_topics[b_idx]
                
                # Remove current assignment
                self._n_wz[w1, old_z] -= 1
                self._n_wz[w2, old_z] -= 1
                self._n_z[old_z] -= 2
                
                # Sample new topic
                new_z = self._sample_topic(w1, w2)
                
                # Update assignment
                self._biterm_topics[b_idx] = new_z
                self._n_wz[w1, new_z] += 1
                self._n_wz[w2, new_z] += 1
                self._n_z[new_z] += 2
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}/{self.n_iter}")
        
        # Compute final distributions
        self._compute_distributions()
        
        # Compute document-topic distributions
        self._compute_doc_topics(bow_matrix)
        
        if verbose:
            print("BTM training completed.")
        
        return self
    
    def _compute_distributions(self):
        """Compute theta (global topic dist) and phi (topic-word dist)."""
        # Global topic distribution
        self._theta = (self._n_z + self.alpha) / (self._n_z.sum() + self.num_topics * self.alpha)
        
        # Topic-word distribution
        self._phi = np.zeros((self.num_topics, self.vocab_size))
        for k in range(self.num_topics):
            self._phi[k] = (self._n_wz[:, k] + self.beta) / (self._n_z[k] + self.vocab_size * self.beta)
    
    def _compute_doc_topics(self, bow_matrix: np.ndarray):
        """Compute document-topic distributions."""
        num_docs = bow_matrix.shape[0]
        self._doc_theta = np.zeros((num_docs, self.num_topics))
        
        for d, doc in enumerate(bow_matrix):
            # Get words in document
            words = []
            for word_id, count in enumerate(doc):
                words.extend([word_id] * int(count))
            
            if len(words) < 2:
                # For very short documents, use global distribution
                self._doc_theta[d] = self._theta
                continue
            
            # Compute P(z|d) by aggregating biterm probabilities
            p_zd = np.zeros(self.num_topics)
            
            for i in range(len(words)):
                for j in range(i + 1, min(i + self.window_size + 1, len(words))):
                    w1, w2 = words[i], words[j]
                    
                    # P(z|b) ∝ P(z) * P(w1|z) * P(w2|z)
                    for k in range(self.num_topics):
                        p_zd[k] += self._theta[k] * self._phi[k, w1] * self._phi[k, w2]
            
            # Normalize
            if p_zd.sum() > 0:
                self._doc_theta[d] = p_zd / p_zd.sum()
            else:
                self._doc_theta[d] = self._theta
    
    def transform(self, bow_matrix: np.ndarray) -> np.ndarray:
        """
        Transform documents to topic distributions.
        
        Args:
            bow_matrix: BOW matrix, shape (num_docs, vocab_size)
        
        Returns:
            theta: Document-topic distribution, shape (num_docs, num_topics)
        """
        if self._phi is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        num_docs = bow_matrix.shape[0]
        doc_theta = np.zeros((num_docs, self.num_topics))
        
        for d, doc in enumerate(bow_matrix):
            words = []
            for word_id, count in enumerate(doc):
                words.extend([word_id] * int(count))
            
            if len(words) < 2:
                doc_theta[d] = self._theta
                continue
            
            p_zd = np.zeros(self.num_topics)
            
            for i in range(len(words)):
                for j in range(i + 1, min(i + self.window_size + 1, len(words))):
                    w1, w2 = words[i], words[j]
                    
                    for k in range(self.num_topics):
                        p_zd[k] += self._theta[k] * self._phi[k, w1] * self._phi[k, w2]
            
            if p_zd.sum() > 0:
                doc_theta[d] = p_zd / p_zd.sum()
            else:
                doc_theta[d] = self._theta
        
        return doc_theta
    
    def get_theta(self, bow_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Get document-topic distribution."""
        if bow_matrix is not None:
            return self.transform(bow_matrix)
        return self._doc_theta
    
    def get_beta(self) -> np.ndarray:
        """Get topic-word distribution."""
        if self._phi is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._phi
    
    def get_global_topic_dist(self) -> np.ndarray:
        """Get global topic distribution (corpus-level)."""
        return self._theta
    
    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
        vocab: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top words for a topic.
        
        Args:
            topic_id: Topic index
            top_n: Number of top words
            vocab: Vocabulary list
        
        Returns:
            List of (word, probability) tuples
        """
        if self._phi is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topic_dist = self._phi[topic_id]
        top_indices = np.argsort(topic_dist)[::-1][:top_n]
        
        if vocab is not None:
            return [(vocab[i], topic_dist[i]) for i in top_indices]
        else:
            return [(str(i), topic_dist[i]) for i in top_indices]


def create_btm(vocab_size: int, num_topics: int = 20, **kwargs) -> BTM:
    """Create BTM model."""
    return BTM(vocab_size=vocab_size, num_topics=num_topics, **kwargs)

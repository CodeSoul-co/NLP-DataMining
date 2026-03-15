"""
HDP (Hierarchical Dirichlet Process) Topic Model

A non-parametric Bayesian topic model that automatically determines the number of topics.
Uses a two-level Dirichlet Process: global topics shared across documents, and document-specific
topic proportions.

Key Features:
- Automatic topic number inference
- Hierarchical structure for topic sharing
- Based on Gibbs sampling or variational inference

Reference:
- Teh et al., "Hierarchical Dirichlet Processes", JASA 2006
- Wang et al., "Online Variational Inference for the Hierarchical Dirichlet Process", AISTATS 2011

Note: This implementation uses gensim's HDP wrapper for simplicity.
For production use, consider the C++ implementation in _reference/hdp/
"""

import numpy as np
from typing import Dict, Optional, List, Any, Tuple
import warnings

from ..base import TraditionalTopicModel


class HDP(TraditionalTopicModel):
    """
    Hierarchical Dirichlet Process Topic Model
    
    Automatically infers the number of topics from data.
    Uses gensim's HdpModel implementation.
    
    Attributes:
        max_topics: Maximum number of topics (upper bound)
        alpha: Concentration parameter for document-level DP
        gamma: Concentration parameter for corpus-level DP
        kappa: Learning rate decay
        tau: Slow down parameter
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_topics: int = 150,
        alpha: float = 1.0,
        gamma: float = 1.0,
        kappa: float = 1.0,
        tau: float = 64.0,
        K: int = 15,  # Initial number of topics
        T: int = 150,  # Top level truncation
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize HDP model.
        
        Args:
            vocab_size: Size of vocabulary
            max_topics: Maximum number of topics (truncation level)
            alpha: Second level concentration (document-level)
            gamma: First level concentration (corpus-level)
            kappa: Learning rate decay
            tau: Slow down parameter
            K: Second level truncation (number of topics per document)
            T: Top level truncation (total number of topics)
            random_state: Random seed
        """
        # HDP doesn't have fixed num_topics, use max_topics as upper bound
        super().__init__(vocab_size=vocab_size, num_topics=max_topics)
        
        self.max_topics = max_topics
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.tau = tau
        self.K = min(K, max_topics)  # K should not exceed max_topics
        self.T = max_topics  # Use max_topics as top level truncation
        self.random_state = random_state
        
        self.model = None
        self.dictionary = None
        self.corpus = None
        self._actual_num_topics = None
        self._beta = None
        self._theta = None
    
    def _bow_to_corpus(self, bow_matrix: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Convert BOW matrix to gensim corpus format."""
        corpus = []
        for doc in bow_matrix:
            doc_bow = [(i, int(count)) for i, count in enumerate(doc) if count > 0]
            corpus.append(doc_bow)
        return corpus
    
    def fit(
        self,
        bow_matrix: np.ndarray,
        vocab: Optional[List[str]] = None,
        **kwargs
    ) -> 'HDP':
        """
        Fit HDP model.
        
        Args:
            bow_matrix: BOW matrix, shape (num_docs, vocab_size)
            vocab: Vocabulary list
        
        Returns:
            self
        """
        try:
            from gensim.models import HdpModel
            from gensim.corpora import Dictionary
        except ImportError:
            raise ImportError("gensim is required for HDP. Install with: pip install gensim")
        
        # Convert to gensim format
        self.corpus = self._bow_to_corpus(bow_matrix)
        
        # Create dictionary
        if vocab is not None:
            self.dictionary = Dictionary([vocab])
        else:
            # Create dummy dictionary
            self.dictionary = Dictionary([[str(i) for i in range(self.vocab_size)]])
        
        # Train HDP
        self.model = HdpModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            max_chunks=None,
            max_time=None,
            chunksize=256,
            kappa=self.kappa,
            tau=self.tau,
            K=self.K,
            T=self.T,
            alpha=self.alpha,
            gamma=self.gamma,
            eta=0.01,
            scale=1.0,
            var_converge=0.0001,
            random_state=self.random_state
        )
        
        # Get actual number of topics
        self._actual_num_topics = len(self.model.get_topics())
        
        # Cache beta matrix
        self._beta = self.model.get_topics()
        
        # Compute theta for training documents
        self._compute_theta(bow_matrix)
        
        return self
    
    def _compute_theta(self, bow_matrix: np.ndarray):
        """Compute document-topic distribution."""
        corpus = self._bow_to_corpus(bow_matrix)
        num_docs = len(corpus)
        num_topics = self._actual_num_topics
        
        self._theta = np.zeros((num_docs, num_topics))
        
        for i, doc in enumerate(corpus):
            topic_dist = self.model[doc]
            for topic_id, prob in topic_dist:
                if topic_id < num_topics:
                    self._theta[i, topic_id] = prob
        
        # Normalize
        row_sums = self._theta.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self._theta = self._theta / row_sums
    
    def transform(self, bow_matrix: np.ndarray) -> np.ndarray:
        """
        Transform documents to topic distributions.
        
        Args:
            bow_matrix: BOW matrix, shape (num_docs, vocab_size)
        
        Returns:
            theta: Document-topic distribution, shape (num_docs, num_topics)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        corpus = self._bow_to_corpus(bow_matrix)
        num_docs = len(corpus)
        num_topics = self._actual_num_topics
        
        theta = np.zeros((num_docs, num_topics))
        
        for i, doc in enumerate(corpus):
            topic_dist = self.model[doc]
            for topic_id, prob in topic_dist:
                if topic_id < num_topics:
                    theta[i, topic_id] = prob
        
        # Normalize
        row_sums = theta.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        theta = theta / row_sums
        
        return theta
    
    def get_theta(self, bow_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Get document-topic distribution."""
        if bow_matrix is not None:
            return self.transform(bow_matrix)
        return self._theta
    
    def get_beta(self) -> np.ndarray:
        """Get topic-word distribution."""
        if self._beta is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._beta
    
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
            topic_n: Number of top words
            vocab: Vocabulary list
        
        Returns:
            List of (word, probability) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topic_terms = self.model.show_topic(topic_id, topn=top_n)
        return topic_terms
    
    @property
    def actual_num_topics(self) -> int:
        """Get the actual number of topics inferred by HDP."""
        return self._actual_num_topics or 0
    
    def print_topics(self, num_topics: int = 10, num_words: int = 10):
        """Print top topics."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topics = self.model.print_topics(num_topics=num_topics, num_words=num_words)
        for topic_id, topic_str in topics:
            print(f"Topic {topic_id}: {topic_str}")


def create_hdp(vocab_size: int, max_topics: int = 150, **kwargs) -> HDP:
    """Create HDP model."""
    return HDP(vocab_size=vocab_size, max_topics=max_topics, **kwargs)

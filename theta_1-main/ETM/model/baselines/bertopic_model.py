"""
BERTopic - BERT-based Topic Modeling

A topic modeling technique that leverages BERT embeddings, UMAP dimensionality reduction,
and HDBSCAN clustering to create dense clusters of semantically similar documents.

Key Features:
- Uses pre-trained language model embeddings (SBERT)
- UMAP for dimensionality reduction
- HDBSCAN for clustering (automatic topic number)
- c-TF-IDF for topic representation
- Topic -1 represents outliers/noise

Reference:
- Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure", 2022

Note: This is a wrapper around the bertopic library.
"""

import numpy as np
from typing import Dict, Optional, List, Any, Tuple, Union
import warnings

from ..base import TraditionalTopicModel


class BERTopicModel(TraditionalTopicModel):
    """
    BERTopic Wrapper
    
    Uses BERT embeddings + UMAP + HDBSCAN for topic modeling.
    Automatically determines the number of topics.
    
    Attributes:
        embedding_model: Name or path of sentence transformer model
        n_neighbors: UMAP n_neighbors parameter
        n_components: UMAP dimensionality
        min_cluster_size: HDBSCAN minimum cluster size
        min_samples: HDBSCAN min_samples
        top_n_words: Number of words per topic
    """
    
    def __init__(
        self,
        vocab_size: int = None,  # Not used, kept for interface compatibility
        num_topics: int = None,  # Can be None for automatic detection
        embedding_model: str = "all-MiniLM-L6-v2",
        n_neighbors: int = 15,
        n_components: int = 5,
        min_cluster_size: int = 10,
        min_samples: int = 10,
        top_n_words: int = 10,
        language: str = "english",
        calculate_probabilities: bool = True,
        verbose: bool = True,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize BERTopic model.
        
        Args:
            vocab_size: Not used (kept for interface compatibility)
            num_topics: Number of topics (None for automatic detection)
            embedding_model: Sentence transformer model name
            n_neighbors: UMAP n_neighbors
            n_components: UMAP output dimensions
            min_cluster_size: HDBSCAN minimum cluster size
            min_samples: HDBSCAN min_samples
            top_n_words: Number of top words per topic
            language: Language for stopwords
            calculate_probabilities: Whether to calculate topic probabilities
            verbose: Print progress
            random_state: Random seed
        """
        # Use a placeholder for vocab_size since BERTopic doesn't need it
        super().__init__(vocab_size=vocab_size or 1, num_topics=num_topics or 0)
        
        self.embedding_model_name = embedding_model
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.top_n_words = top_n_words
        self.language = language
        self.calculate_probabilities = calculate_probabilities
        self.verbose = verbose
        self.random_state = random_state
        self._nr_topics = num_topics  # Target number of topics (can be None)
        
        self.model = None
        self._topics = None
        self._probs = None
        self._topic_info = None
        self._embeddings = None
    
    def _init_model(self):
        """Initialize BERTopic model with components."""
        import os
        from pathlib import Path
        
        try:
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "BERTopic requires additional packages. Install with:\n"
                "pip install bertopic umap-learn hdbscan sentence-transformers"
            ) from e
        
        # Initialize UMAP
        umap_model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=self.random_state
        )
        
        # Initialize HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Initialize embedding model - try local paths first to avoid network issues
        embedding_models_dir = os.environ.get('ETM_EMBEDDING_MODELS_DIR', '/mnt/embedding_models')
        local_sbert_paths = [
            '/mnt/sbert',  # DLC mount path
            f'{embedding_models_dir}/sbert/all-MiniLM-L6-v2',
            f'{embedding_models_dir}/all-MiniLM-L6-v2',
            f'{embedding_models_dir}/sbert',
            '/mnt/embedding_models/sbert/all-MiniLM-L6-v2',
            '/mnt/embedding_models/all-MiniLM-L6-v2',
            '/mnt/embedding_models/sbert',
            '/root/autodl-tmp/ETM/model/baselines/sbert/sentence-transformers/all-MiniLM-L6-v2',
            '/root/autodl-tmp/ETM/model/sbert/sentence-transformers/all-MiniLM-L6-v2',
        ]
        
        sbert_model_path = self.embedding_model_name
        print(f"  [BERTopic] Looking for local SBERT model...")
        for path in local_sbert_paths:
            if Path(path).exists():
                # Verify it has model files
                p = Path(path)
                has_config = (p / 'config.json').exists() or (p / 'config_sentence_transformers.json').exists()
                has_model = (p / 'pytorch_model.bin').exists() or (p / 'model.safetensors').exists()
                if has_config or has_model:
                    sbert_model_path = path
                    print(f"  [BERTopic] Using local SBERT model: {path}")
                    break
        
        if sbert_model_path == self.embedding_model_name:
            print(f"  [BERTopic] No local model found, will try online: {self.embedding_model_name}")
        
        embedding_model = SentenceTransformer(sbert_model_path)
        
        # Initialize BERTopic
        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            top_n_words=self.top_n_words,
            language=self.language,
            calculate_probabilities=self.calculate_probabilities,
            verbose=self.verbose,
            nr_topics=self._nr_topics  # None for automatic
        )
    
    def fit(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'BERTopicModel':
        """
        Fit BERTopic model.
        
        Args:
            texts: List of document texts
            embeddings: Pre-computed document embeddings (optional)
        
        Returns:
            self
        """
        if self.model is None:
            self._init_model()
        
        # Fit model
        self._topics, self._probs = self.model.fit_transform(
            texts, 
            embeddings=embeddings
        )
        
        # Store embeddings if provided
        self._embeddings = embeddings
        
        # Get topic info
        self._topic_info = self.model.get_topic_info()
        
        # Update actual number of topics (excluding -1)
        unique_topics = set(self._topics)
        unique_topics.discard(-1)
        self._num_topics = len(unique_topics)
        
        return self
    
    def fit_from_bow(
        self,
        bow_matrix: np.ndarray,
        vocab: List[str],
        **kwargs
    ) -> 'BERTopicModel':
        """
        Fit from BOW matrix by reconstructing texts.
        
        Note: This is a fallback method. BERTopic works best with raw texts.
        
        Args:
            bow_matrix: BOW matrix, shape (num_docs, vocab_size)
            vocab: Vocabulary list
        
        Returns:
            self
        """
        # Reconstruct texts from BOW (approximate)
        texts = []
        for doc in bow_matrix:
            words = []
            for word_id, count in enumerate(doc):
                if count > 0:
                    words.extend([vocab[word_id]] * int(min(count, 5)))  # Cap repetitions
            texts.append(' '.join(words))
        
        return self.fit(texts)
    
    def transform(self, texts: List[str], embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform documents to topic distributions.
        
        Args:
            texts: List of document texts
            embeddings: Pre-computed embeddings (optional)
        
        Returns:
            probs: Topic probabilities, shape (num_docs, num_topics)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topics, probs = self.model.transform(texts, embeddings=embeddings)
        
        if probs is None:
            # If probabilities not available, create one-hot encoding
            num_docs = len(texts)
            num_topics = self.num_topics
            probs = np.zeros((num_docs, num_topics))
            for i, t in enumerate(topics):
                if t >= 0 and t < num_topics:
                    probs[i, t] = 1.0
        
        return probs
    
    def get_theta(self, texts: Optional[List[str]] = None) -> np.ndarray:
        """Get document-topic distribution."""
        if texts is not None:
            return self.transform(texts)
        
        if self._probs is None:
            raise ValueError("No probabilities available. Fit model first.")
        
        return self._probs
    
    def get_beta(self) -> np.ndarray:
        """
        Get topic-word distribution.
        
        Note: BERTopic uses c-TF-IDF, not traditional beta matrix.
        This returns an approximation based on topic words.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get all topics (excluding -1)
        topics = self.model.get_topics()
        
        # Build vocabulary from all topic words
        all_words = set()
        for topic_id, words in topics.items():
            if topic_id != -1:
                for word, _ in words:
                    all_words.add(word)
        
        vocab = sorted(list(all_words))
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        
        # Build beta matrix
        num_topics = len([t for t in topics.keys() if t != -1])
        vocab_size = len(vocab)
        beta = np.zeros((num_topics, vocab_size))
        
        for topic_id, words in topics.items():
            if topic_id != -1 and topic_id < num_topics:
                for word, score in words:
                    if word in word_to_idx:
                        beta[topic_id, word_to_idx[word]] = max(0, score)
        
        # Normalize
        row_sums = beta.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        beta = beta / row_sums
        
        return beta
    
    def get_topics(self) -> List[int]:
        """Get topic assignments for fitted documents."""
        return self._topics
    
    def get_topic_info(self):
        """Get topic information DataFrame."""
        return self._topic_info
    
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
            vocab: Not used (BERTopic has its own vocabulary)
        
        Returns:
            List of (word, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        topic_words = self.model.get_topic(topic_id)
        return topic_words[:top_n] if topic_words else []
    
    def reduce_topics(self, num_topics: int):
        """
        Reduce the number of topics.
        
        Args:
            num_topics: Target number of topics
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.reduce_topics(self.model.original_topics_, num_topics)
        self._topics = self.model.topics_
        self._num_topics = num_topics
    
    def visualize_topics(self):
        """Visualize topics (returns plotly figure)."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.visualize_topics()
    
    def visualize_documents(self, texts: List[str], embeddings: Optional[np.ndarray] = None):
        """Visualize documents in 2D space."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.visualize_documents(texts, embeddings=embeddings)
    
    @property
    def outlier_count(self) -> int:
        """Get number of outlier documents (topic -1)."""
        if self._topics is None:
            return 0
        return sum(1 for t in self._topics if t == -1)


def create_bertopic(
    num_topics: int = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    **kwargs
) -> BERTopicModel:
    """Create BERTopic model."""
    return BERTopicModel(
        num_topics=num_topics,
        embedding_model=embedding_model,
        **kwargs
    )

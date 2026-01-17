"""
Unified Data Loader for ETM Visualization

Provides a single interface to load all ETM training results for visualization.
Handles different data formats and ensures consistent access to:
- theta (document-topic distribution)
- beta (topic-word distribution)
- topic_embeddings
- word_embeddings (vocab_embeddings)
- vocab
- topic_words
- doc_embeddings
- timestamps (if available in original data)
- training_history
- evaluation metrics
"""

import os
import json
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
import glob
import pandas as pd

logger = logging.getLogger(__name__)


class VisualizationDataLoader:
    """
    Unified data loader for ETM visualization.
    
    Loads all necessary data from the result directory structure:
    result/{dataset}/
        ├── bow/                    # Shared across modes
        │   ├── bow_matrix.npz
        │   ├── vocab.txt
        │   └── vocab_embeddings.npy
        └── {mode}/                 # zero_shot, supervised, unsupervised
            ├── embeddings/
            │   └── doc_embeddings.npy
            ├── model/
            │   ├── theta_{timestamp}.npy
            │   ├── beta_{timestamp}.npy
            │   ├── topic_embeddings_{timestamp}.npy
            │   ├── topic_words_{timestamp}.json
            │   └── training_history_{timestamp}.json
            └── evaluation/
                └── metrics_{timestamp}.json
    """
    
    def __init__(
        self,
        result_dir: str,
        dataset: str,
        mode: str,
        timestamp: Optional[str] = None,
        data_dir: Optional[str] = None
    ):
        """
        Initialize data loader.
        
        Args:
            result_dir: Base result directory (e.g., /root/autodl-tmp/result)
            dataset: Dataset name (e.g., hatespeech, socialTwitter)
            mode: Embedding mode (zero_shot, supervised, unsupervised)
            timestamp: Specific timestamp to load (if None, loads latest)
            data_dir: Original data directory for loading timestamps
        """
        self.result_dir = Path(result_dir)
        self.dataset = dataset
        self.mode = mode
        self.data_dir = Path(data_dir) if data_dir else Path(result_dir).parent / "data"
        
        # Paths
        self.dataset_dir = self.result_dir / dataset
        self.bow_dir = self.dataset_dir / "bow"
        self.mode_dir = self.dataset_dir / mode
        self.model_dir = self.mode_dir / "model"
        self.embeddings_dir = self.mode_dir / "embeddings"
        self.evaluation_dir = self.mode_dir / "evaluation"
        self.topic_words_dir = self.mode_dir / "topic_words"
        self.visualization_dir = self.mode_dir / "visualization"
        
        # Determine timestamp
        self.timestamp = timestamp or self._get_latest_timestamp()
        
        # Cache for loaded data
        self._cache: Dict[str, Any] = {}
        
        logger.info(f"VisualizationDataLoader initialized:")
        logger.info(f"  Dataset: {dataset}, Mode: {mode}")
        logger.info(f"  Timestamp: {self.timestamp}")
    
    def _get_latest_timestamp(self) -> Optional[str]:
        """Get the latest timestamp from model files."""
        pattern = str(self.model_dir / "theta_*.npy")
        files = glob.glob(pattern)
        
        if not files:
            # Try topic_words directory
            pattern = str(self.topic_words_dir / "topic_words_*.json")
            files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No model files found in {self.model_dir}")
            return None
        
        # Extract timestamps and get latest
        timestamps = []
        for f in files:
            name = os.path.basename(f)
            # Extract timestamp from filename like "theta_20260116_214418.npy"
            parts = name.replace('.npy', '').replace('.json', '').split('_')
            if len(parts) >= 3:
                ts = '_'.join(parts[-2:])  # e.g., "20260116_214418"
                timestamps.append(ts)
        
        if timestamps:
            return sorted(timestamps)[-1]
        return None
    
    def _load_with_cache(self, key: str, loader_func) -> Any:
        """Load data with caching."""
        if key not in self._cache:
            self._cache[key] = loader_func()
        return self._cache[key]
    
    # ==================== Core Data Loaders ====================
    
    def load_theta(self) -> Optional[np.ndarray]:
        """Load document-topic distribution (N x K)."""
        def loader():
            path = self.model_dir / f"theta_{self.timestamp}.npy"
            if path.exists():
                theta = np.load(path)
                logger.info(f"Loaded theta: {theta.shape}")
                return theta
            logger.warning(f"Theta not found: {path}")
            return None
        return self._load_with_cache('theta', loader)
    
    def load_beta(self) -> Optional[np.ndarray]:
        """Load topic-word distribution (K x V)."""
        def loader():
            path = self.model_dir / f"beta_{self.timestamp}.npy"
            if path.exists():
                beta = np.load(path)
                logger.info(f"Loaded beta: {beta.shape}")
                return beta
            logger.warning(f"Beta not found: {path}")
            return None
        return self._load_with_cache('beta', loader)
    
    def load_topic_embeddings(self) -> Optional[np.ndarray]:
        """Load topic embeddings (K x E)."""
        def loader():
            path = self.model_dir / f"topic_embeddings_{self.timestamp}.npy"
            if path.exists():
                emb = np.load(path)
                logger.info(f"Loaded topic_embeddings: {emb.shape}")
                return emb
            logger.warning(f"Topic embeddings not found: {path}")
            return None
        return self._load_with_cache('topic_embeddings', loader)
    
    def load_word_embeddings(self) -> Optional[np.ndarray]:
        """Load word/vocab embeddings (V x E) from shared BOW directory."""
        def loader():
            path = self.bow_dir / "vocab_embeddings.npy"
            if path.exists():
                emb = np.load(path)
                logger.info(f"Loaded word_embeddings: {emb.shape}")
                return emb
            logger.warning(f"Word embeddings not found: {path}")
            return None
        return self._load_with_cache('word_embeddings', loader)
    
    def load_vocab(self) -> Optional[List[str]]:
        """Load vocabulary list."""
        def loader():
            # Try vocab.txt first
            path = self.bow_dir / "vocab.txt"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    vocab = [line.strip() for line in f]
                logger.info(f"Loaded vocab: {len(vocab)} words")
                return vocab
            
            # Try vocab.json
            path = self.bow_dir / "vocab.json"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                logger.info(f"Loaded vocab: {len(vocab)} words")
                return vocab
            
            logger.warning(f"Vocab not found in {self.bow_dir}")
            return None
        return self._load_with_cache('vocab', loader)
    
    def load_topic_words(
        self,
        format: str = 'dict'
    ) -> Optional[Union[Dict[int, List[Tuple[str, float]]], List[Tuple[int, List[Tuple[str, float]]]]]]:
        """
        Load topic words.
        
        Args:
            format: Output format
                - 'dict': {topic_id: [(word, prob), ...]}
                - 'list': [(topic_id, [(word, prob), ...]), ...]
        
        Returns:
            Topic words in specified format
        """
        def loader():
            # Try topic_words directory first
            path = self.topic_words_dir / f"topic_words_{self.timestamp}.json"
            if not path.exists():
                # Try model directory
                path = self.model_dir / f"topic_words_{self.timestamp}.json"
            
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to dict format: {topic_id: [(word, prob), ...]}
                topic_words_dict = {}
                if isinstance(data, dict):
                    # Format: {"topic_id": [[word, prob], ...], ...}
                    for k, v in data.items():
                        topic_id = int(k.replace('topic_', '')) if isinstance(k, str) else int(k)
                        if isinstance(v, list) and len(v) > 0:
                            if isinstance(v[0], dict):
                                # Format: [{"word": "xxx", "prob": 0.1}, ...]
                                topic_words_dict[topic_id] = [(item['word'], item['prob']) for item in v]
                            elif isinstance(v[0], list):
                                # Format: [["word", 0.1], ...]
                                topic_words_dict[topic_id] = [(item[0], item[1]) for item in v]
                            elif isinstance(v[0], str):
                                # Format: ["word1", "word2", ...] - no probs
                                topic_words_dict[topic_id] = [(w, 1.0/(i+1)) for i, w in enumerate(v)]
                elif isinstance(data, list):
                    # Format: [[topic_id, [[word, prob], ...]], ...] or [[[word, prob], ...], ...]
                    for item in data:
                        if isinstance(item, list) and len(item) >= 2:
                            if isinstance(item[0], int) and isinstance(item[1], list):
                                # Format: [topic_id, [[word, prob], ...]]
                                topic_id = item[0]
                                words = item[1]
                                if words and isinstance(words[0], list):
                                    topic_words_dict[topic_id] = [(w, p) for w, p in words]
                                else:
                                    topic_words_dict[topic_id] = words
                            elif isinstance(item[0], str):
                                # Format: [[word, prob], ...] - use index as topic_id
                                topic_words_dict[len(topic_words_dict)] = [(w, p) for w, p in item] if isinstance(item[0], list) else item
                
                logger.info(f"Loaded topic_words: {len(topic_words_dict)} topics")
                return topic_words_dict
            
            logger.warning(f"Topic words not found: {path}")
            return None
        
        result = self._load_with_cache('topic_words', loader)
        
        if result is None:
            return None
        
        if format == 'list':
            return [(k, v) for k, v in sorted(result.items())]
        return result
    
    def load_doc_embeddings(self) -> Optional[np.ndarray]:
        """Load document embeddings (N x E)."""
        def loader():
            # Try different naming conventions
            patterns = [
                self.embeddings_dir / "doc_embeddings.npy",
                self.embeddings_dir / f"{self.dataset}_{self.mode}_embeddings.npy",
            ]
            
            for path in patterns:
                if path.exists():
                    emb = np.load(path)
                    logger.info(f"Loaded doc_embeddings: {emb.shape}")
                    return emb
            
            logger.warning(f"Doc embeddings not found in {self.embeddings_dir}")
            return None
        return self._load_with_cache('doc_embeddings', loader)
    
    def load_bow(self) -> Optional[sparse.csr_matrix]:
        """Load BOW matrix (N x V)."""
        def loader():
            path = self.bow_dir / "bow_matrix.npz"
            if path.exists():
                bow = sparse.load_npz(path)
                logger.info(f"Loaded BOW: {bow.shape}")
                return bow
            logger.warning(f"BOW not found: {path}")
            return None
        return self._load_with_cache('bow', loader)
    
    def load_training_history(self) -> Optional[Dict]:
        """Load training history."""
        def loader():
            path = self.model_dir / f"training_history_{self.timestamp}.json"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logger.info(f"Loaded training_history: {len(history.get('train_loss', []))} epochs")
                return history
            logger.warning(f"Training history not found: {path}")
            return None
        return self._load_with_cache('training_history', loader)
    
    def load_metrics(self) -> Optional[Dict]:
        """Load evaluation metrics."""
        def loader():
            path = self.evaluation_dir / f"metrics_{self.timestamp}.json"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                logger.info(f"Loaded metrics: {list(metrics.keys())}")
                return metrics
            logger.warning(f"Metrics not found: {path}")
            return None
        return self._load_with_cache('metrics', loader)
    
    def load_config(self) -> Optional[Dict]:
        """Load training config."""
        def loader():
            path = self.model_dir / f"config_{self.timestamp}.json"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded config")
                return config
            logger.warning(f"Config not found: {path}")
            return None
        return self._load_with_cache('config', loader)
    
    # ==================== Timestamp Loading ====================
    
    def load_timestamps(self) -> Optional[np.ndarray]:
        """
        Load timestamps from original data if available.
        
        Only loads real timestamps from the original dataset.
        Does NOT create synthetic timestamps.
        
        Returns:
            Array of timestamps if available, None otherwise
        """
        def loader():
            # Check if timestamps were saved during training
            ts_path = self.mode_dir / "timestamps.npy"
            if ts_path.exists():
                timestamps = np.load(ts_path, allow_pickle=True)
                logger.info(f"Loaded timestamps from saved file: {len(timestamps)}")
                return timestamps
            
            # Try to load from original data
            config = self.load_config()
            if config is None:
                return None
            
            timestamp_column = config.get('data', {}).get('timestamp_column')
            if timestamp_column is None:
                logger.info("No timestamp column configured - temporal analysis not available")
                return None
            
            # Load original data to get timestamps
            data_path = self.data_dir / self.dataset
            csv_files = list(data_path.glob("*.csv"))
            
            if not csv_files:
                logger.warning(f"No CSV files found in {data_path}")
                return None
            
            # Load the main CSV file
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, usecols=[timestamp_column])
                    if timestamp_column in df.columns:
                        timestamps = pd.to_datetime(df[timestamp_column], errors='coerce')
                        valid_count = timestamps.notna().sum()
                        if valid_count > 0:
                            logger.info(f"Loaded {valid_count} timestamps from {csv_file.name}")
                            return timestamps.values
                except Exception as e:
                    logger.debug(f"Could not load timestamps from {csv_file}: {e}")
                    continue
            
            logger.info("No valid timestamps found in original data")
            return None
        
        return self._load_with_cache('timestamps', loader)
    
    def has_timestamps(self) -> bool:
        """Check if timestamps are available."""
        return self.load_timestamps() is not None
    
    # ==================== Convenience Methods ====================
    
    def load_all(self) -> Dict[str, Any]:
        """
        Load all available data for visualization.
        
        Returns:
            Dictionary containing all loaded data
        """
        data = {
            'theta': self.load_theta(),
            'beta': self.load_beta(),
            'topic_embeddings': self.load_topic_embeddings(),
            'word_embeddings': self.load_word_embeddings(),
            'vocab': self.load_vocab(),
            'topic_words': self.load_topic_words(format='dict'),
            'topic_words_list': self.load_topic_words(format='list'),
            'doc_embeddings': self.load_doc_embeddings(),
            'bow': self.load_bow(),
            'timestamps': self.load_timestamps(),
            'training_history': self.load_training_history(),
            'metrics': self.load_metrics(),
            'config': self.load_config(),
        }
        
        # Add metadata
        data['metadata'] = {
            'dataset': self.dataset,
            'mode': self.mode,
            'timestamp': self.timestamp,
            'has_timestamps': data['timestamps'] is not None,
            'num_docs': data['theta'].shape[0] if data['theta'] is not None else None,
            'num_topics': data['theta'].shape[1] if data['theta'] is not None else None,
            'vocab_size': len(data['vocab']) if data['vocab'] is not None else None,
        }
        
        return data
    
    def get_visualization_ready_data(self) -> Dict[str, Any]:
        """
        Get data formatted for direct use with visualization classes.
        
        Returns:
            Dictionary with data ready for visualization
        """
        data = self.load_all()
        
        viz_data = {
            # For TopicVisualizer
            'topic_words': data['topic_words_list'],
            'beta': data['beta'],
            'theta': data['theta'],
            
            # For TopicEmbeddingSpaceVisualizer
            'topic_embeddings': data['topic_embeddings'],
            'word_embeddings': data['word_embeddings'],
            'vocab': data['vocab'],
            
            # For DocumentTopicUMAPVisualizer
            'doc_embeddings': data['doc_embeddings'],
            
            # For TemporalTopicAnalyzer (only if timestamps exist)
            'timestamps': data['timestamps'],
            'has_temporal': data['timestamps'] is not None,
            
            # Metadata
            'metadata': data['metadata'],
            'metrics': data['metrics'],
            'training_history': data['training_history'],
        }
        
        return viz_data
    
    def get_available_visualizations(self) -> List[str]:
        """
        Get list of available visualizations based on loaded data.
        
        Returns:
            List of visualization names that can be generated
        """
        data = self.load_all()
        available = []
        
        # Basic visualizations (always available if model trained)
        if data['theta'] is not None:
            available.extend([
                'document_topic_distribution',
                'document_topic_umap',
                'topic_proportions',
            ])
        
        if data['beta'] is not None:
            available.extend([
                'topic_word_bars',
                'topic_word_clouds',
                'topic_similarity_heatmap',
                'topic_network',
            ])
        
        if data['topic_words'] is not None:
            available.append('topic_word_table')
        
        if data['topic_embeddings'] is not None and data['word_embeddings'] is not None:
            available.extend([
                'topic_embedding_space',
                'topic_word_umap',
            ])
        
        if data['doc_embeddings'] is not None:
            available.extend([
                'document_similarity',
                'document_clustering',
            ])
        
        # Temporal visualizations (only if timestamps exist)
        if data['timestamps'] is not None:
            available.extend([
                'temporal_topic_evolution',
                'document_volume_trend',
                'topic_sankey_diagram',
                'dynamic_topic_evolution',
            ])
        
        if data['training_history'] is not None:
            available.append('training_loss_curve')
        
        if data['metrics'] is not None:
            available.append('evaluation_metrics')
        
        return available
    
    def summary(self) -> str:
        """Get a summary of loaded data."""
        data = self.load_all()
        
        lines = [
            f"=" * 60,
            f"ETM Visualization Data Summary",
            f"=" * 60,
            f"Dataset: {self.dataset}",
            f"Mode: {self.mode}",
            f"Timestamp: {self.timestamp}",
            f"-" * 60,
        ]
        
        # Data shapes
        if data['theta'] is not None:
            lines.append(f"theta (doc-topic): {data['theta'].shape}")
        if data['beta'] is not None:
            lines.append(f"beta (topic-word): {data['beta'].shape}")
        if data['topic_embeddings'] is not None:
            lines.append(f"topic_embeddings: {data['topic_embeddings'].shape}")
        if data['word_embeddings'] is not None:
            lines.append(f"word_embeddings: {data['word_embeddings'].shape}")
        if data['vocab'] is not None:
            lines.append(f"vocab: {len(data['vocab'])} words")
        if data['topic_words'] is not None:
            lines.append(f"topic_words: {len(data['topic_words'])} topics")
        if data['doc_embeddings'] is not None:
            lines.append(f"doc_embeddings: {data['doc_embeddings'].shape}")
        
        lines.append(f"-" * 60)
        lines.append(f"Timestamps available: {data['timestamps'] is not None}")
        
        lines.append(f"-" * 60)
        lines.append(f"Available visualizations:")
        for viz in self.get_available_visualizations():
            lines.append(f"  - {viz}")
        
        lines.append(f"=" * 60)
        
        return "\n".join(lines)


def load_etm_data(
    result_dir: str,
    dataset: str,
    mode: str,
    timestamp: Optional[str] = None
) -> VisualizationDataLoader:
    """
    Convenience function to create a data loader.
    
    Args:
        result_dir: Base result directory
        dataset: Dataset name
        mode: Embedding mode
        timestamp: Specific timestamp (optional)
    
    Returns:
        VisualizationDataLoader instance
    """
    return VisualizationDataLoader(result_dir, dataset, mode, timestamp)

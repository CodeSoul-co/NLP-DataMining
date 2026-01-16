"""
Topic Visualization Tools for ETM

This module provides visualization tools for ETM results:
- Topic word clouds
- Topic similarity heatmap
- Document-topic distribution visualization
- Topic evolution over time (if timestamps available)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

# Try to import wordcloud, but don't fail if it's not available
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logging.warning("WordCloud package not available. Install with 'pip install wordcloud'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopicVisualizer:
    """
    Visualization tools for ETM results.
    
    Provides methods to visualize:
    - Topic word clouds
    - Topic similarity heatmap
    - Document-topic distribution
    - Topic embeddings in 2D space
    """
    
    def __init__(
        self,
        output_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        cmap: str = "viridis",
        random_state: int = 42
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Default figure DPI
            cmap: Default colormap
            random_state: Random state for reproducibility
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        self.random_state = random_state
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def _save_or_show(self, fig, filename=None):
        """Save figure to file or show it"""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {filepath}")
            return filepath
        else:
            plt.show()
            return None
    
    def visualize_topic_words(
        self,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        num_topics: int = None,
        num_words: int = 10,
        as_wordcloud: bool = False,
        filename: str = None
    ) -> Union[plt.Figure, List[plt.Figure]]:
        """
        Visualize top words for each topic.
        
        Args:
            topic_words: List of (topic_idx, [(word, prob), ...])
            num_topics: Number of topics to visualize (None for all)
            num_words: Number of words per topic
            as_wordcloud: Whether to use word clouds
            filename: Filename to save visualization
            
        Returns:
            Figure or list of figures
        """
        if num_topics is None:
            num_topics = len(topic_words)
        else:
            num_topics = min(num_topics, len(topic_words))
        
        if as_wordcloud and not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud package not available, falling back to bar plots")
            as_wordcloud = False
        
        if as_wordcloud:
            # Create a word cloud for each topic
            figs = []
            for topic_idx, words in topic_words[:num_topics]:
                # Create word frequency dictionary
                word_freq = {word: prob for word, prob in words[:num_words*2]}
                
                # Create word cloud
                fig, ax = plt.subplots(figsize=(10, 6))
                wc = WordCloud(
                    background_color='white',
                    width=800,
                    height=400,
                    max_words=num_words,
                    random_state=self.random_state
                ).generate_from_frequencies(word_freq)
                
                ax.imshow(wc, interpolation='bilinear')
                ax.set_title(f'Topic {topic_idx}', fontsize=16)
                ax.axis('off')
                
                figs.append(fig)
                
                # Save or show
                if filename:
                    base, ext = os.path.splitext(filename)
                    topic_filename = f"{base}_topic{topic_idx}{ext}"
                    self._save_or_show(fig, topic_filename)
            
            return figs
        else:
            # Create bar plots for topics
            n_cols = min(3, num_topics)
            n_rows = (num_topics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(self.figsize[0] * n_cols / 2, self.figsize[1] * n_rows / 3),
                constrained_layout=True
            )
            
            # Flatten axes for easier indexing
            if n_rows * n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            for i, (topic_idx, words) in enumerate(topic_words[:num_topics]):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Extract words and probabilities
                top_words = [word for word, _ in words[:num_words]]
                top_probs = [prob for _, prob in words[:num_words]]
                
                # Create horizontal bar plot
                y_pos = np.arange(len(top_words))
                ax.barh(y_pos, top_probs, align='center', alpha=0.6, color=plt.cm.tab20(i % 20))
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_words)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_title(f'Topic {topic_idx}')
                ax.set_xlabel('Probability')
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            
            plt.suptitle('Top Words per Topic', fontsize=16)
            
            # Save or show
            return self._save_or_show(fig, filename)
    
    def visualize_topic_similarity(
        self,
        beta: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        metric: str = 'cosine',
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize topic similarity as a heatmap.
        
        Args:
            beta: Topic-word distribution matrix (K x V)
            topic_words: Optional list of topic words for labels
            metric: Similarity metric ('cosine', 'euclidean', 'correlation')
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        num_topics = beta.shape[0]
        
        # Compute similarity matrix
        if metric == 'cosine':
            sim_matrix = cosine_similarity(beta)
        elif metric == 'euclidean':
            # Convert distances to similarities
            dist_matrix = euclidean_distances(beta)
            sim_matrix = 1 / (1 + dist_matrix)
        elif metric == 'correlation':
            sim_matrix = np.corrcoef(beta)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Create labels if topic_words is provided
        if topic_words:
            labels = []
            for topic_idx, words in topic_words:
                top_words = [word for word, _ in words[:3]]
                label = f"{topic_idx}: {', '.join(top_words)}"
                labels.append(label)
            labels = labels[:num_topics]
        else:
            labels = [f"Topic {i}" for i in range(num_topics)]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            sim_matrix,
            annot=False,
            cmap='viridis',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        ax.set_title(f'Topic Similarity ({metric})', fontsize=16)
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_document_topics(
        self,
        theta: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = 'tsne',
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize document-topic distributions in 2D space.
        
        Args:
            theta: Document-topic distribution matrix (D x K)
            labels: Optional document labels for coloring
            method: Dimensionality reduction method ('tsne', 'pca')
            topic_words: Optional list of topic words for annotation
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=self.random_state,
                init='pca',
                learning_rate='auto'
            )
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensions
        theta_2d = reducer.fit_transform(theta)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot documents
        if labels is not None:
            # Convert labels to categorical if needed
            unique_labels = np.unique(labels)
            label_to_int = {label: i for i, label in enumerate(unique_labels)}
            
            # Color by label
            for label in unique_labels:
                mask = (labels == label)
                ax.scatter(
                    theta_2d[mask, 0],
                    theta_2d[mask, 1],
                    alpha=0.6,
                    label=str(label)
                )
            ax.legend(title='Document Label')
        else:
            # No labels, single color
            ax.scatter(theta_2d[:, 0], theta_2d[:, 1], alpha=0.6)
        
        ax.set_title(f'Document-Topic Distribution ({method.upper()})', fontsize=16)
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_topic_embeddings(
        self,
        topic_embeddings: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        method: str = 'tsne',
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize topic embeddings in 2D space.
        
        Args:
            topic_embeddings: Topic embedding matrix (K x E)
            topic_words: Optional list of topic words for annotation
            method: Dimensionality reduction method ('tsne', 'pca')
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=self.random_state,
                init='pca',
                learning_rate='auto'
            )
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensions
        embeddings_2d = reducer.fit_transform(topic_embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot topic embeddings
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.8,
            s=100,
            c=range(len(embeddings_2d)),
            cmap='tab20'
        )
        
        # Add annotations if topic_words is provided
        if topic_words:
            for i, (topic_idx, words) in enumerate(topic_words):
                if i >= len(embeddings_2d):
                    break
                
                # Get top words
                top_words = [word for word, _ in words[:2]]
                label = f"{topic_idx}: {', '.join(top_words)}"
                
                # Add annotation
                ax.annotate(
                    label,
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9,
                    alpha=0.8,
                    ha='center',
                    va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points'
                )
        
        ax.set_title(f'Topic Embeddings ({method.upper()})', fontsize=16)
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        
        # Save or show
        return self._save_or_show(fig, filename)
    
    def visualize_topic_proportions(
        self,
        theta: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        top_k: int = 10,
        filename: str = None
    ) -> plt.Figure:
        """
        Visualize average topic proportions across documents.
        
        Args:
            theta: Document-topic distribution matrix (D x K)
            topic_words: Optional list of topic words for labels
            top_k: Number of top topics to show
            filename: Filename to save visualization
            
        Returns:
            Figure
        """
        # Calculate average topic proportions
        topic_props = theta.mean(axis=0)
        
        # Get indices of top topics
        top_indices = np.argsort(-topic_props)[:top_k]
        top_props = topic_props[top_indices]
        
        # Create labels
        if topic_words:
            labels = []
            for idx in top_indices:
                for topic_idx, words in topic_words:
                    if topic_idx == idx:
                        top_words = [word for word, _ in words[:2]]
                        label = f"{topic_idx}: {', '.join(top_words)}"
                        labels.append(label)
                        break
            if len(labels) < len(top_indices):
                # Fill in missing labels
                for i, idx in enumerate(top_indices):
                    if i >= len(labels) or not labels[i].startswith(str(idx)):
                        labels.insert(i, f"Topic {idx}")
        else:
            labels = [f"Topic {idx}" for idx in top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_props))
        ax.barh(y_pos, top_props, align='center', alpha=0.6, color=plt.cm.tab20(np.arange(len(top_props)) % 20))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title('Average Topic Proportions', fontsize=16)
        ax.set_xlabel('Proportion')
        
        # Save or show
        return self._save_or_show(fig, filename)


def load_etm_results(results_dir: str, timestamp: str = None):
    """
    Load ETM results from files.
    
    Args:
        results_dir: Directory containing ETM results
        timestamp: Specific timestamp to load (None for latest)
        
    Returns:
        Dictionary with loaded results
    """
    # Find result files
    if timestamp:
        theta_path = os.path.join(results_dir, f"theta_{timestamp}.npy")
        beta_path = os.path.join(results_dir, f"beta_{timestamp}.npy")
        topic_words_path = os.path.join(results_dir, f"topic_words_{timestamp}.json")
        metrics_path = os.path.join(results_dir, f"metrics_{timestamp}.json")
    else:
        # Find latest files
        theta_files = sorted(Path(results_dir).glob("theta_*.npy"), reverse=True)
        beta_files = sorted(Path(results_dir).glob("beta_*.npy"), reverse=True)
        topic_words_files = sorted(Path(results_dir).glob("topic_words_*.json"), reverse=True)
        metrics_files = sorted(Path(results_dir).glob("metrics_*.json"), reverse=True)
        
        if not theta_files or not beta_files or not topic_words_files:
            raise FileNotFoundError(f"Could not find ETM result files in {results_dir}")
        
        theta_path = str(theta_files[0])
        beta_path = str(beta_files[0])
        topic_words_path = str(topic_words_files[0])
        metrics_path = str(metrics_files[0]) if metrics_files else None
    
    # Load files
    theta = np.load(theta_path)
    beta = np.load(beta_path)
    
    with open(topic_words_path, 'r') as f:
        topic_words = json.load(f)
    
    # Convert topic_words format
    topic_words = [(int(k), [(w, float(p)) for w, p in words]) for k, words in topic_words]
    
    # Load metrics if available
    metrics = None
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    return {
        'theta': theta,
        'beta': beta,
        'topic_words': topic_words,
        'metrics': metrics
    }


def visualize_etm_results(
    results_dir: str,
    output_dir: str = None,
    timestamp: str = None,
    show_wordcloud: bool = True
):
    """
    Visualize ETM results.
    
    Args:
        results_dir: Directory containing ETM results
        output_dir: Directory to save visualizations
        timestamp: Specific timestamp to load (None for latest)
        show_wordcloud: Whether to show word clouds
    """
    # Load results
    results = load_etm_results(results_dir, timestamp)
    
    # Create visualizer
    visualizer = TopicVisualizer(output_dir=output_dir)
    
    # Create visualizations
    logger.info("Generating topic word visualization...")
    visualizer.visualize_topic_words(
        results['topic_words'],
        num_topics=10,
        as_wordcloud=show_wordcloud and WORDCLOUD_AVAILABLE,
        filename="topic_words.png"
    )
    
    logger.info("Generating topic similarity visualization...")
    visualizer.visualize_topic_similarity(
        results['beta'],
        results['topic_words'],
        filename="topic_similarity.png"
    )
    
    logger.info("Generating topic proportions visualization...")
    visualizer.visualize_topic_proportions(
        results['theta'],
        results['topic_words'],
        filename="topic_proportions.png"
    )
    
    logger.info("Generating document-topic visualization...")
    visualizer.visualize_document_topics(
        results['theta'],
        method='tsne',
        filename="document_topics_tsne.png"
    )
    
    logger.info("Generating topic embeddings visualization...")
    topic_embeddings = results['beta'] @ results['beta'].T  # Approximate topic embeddings
    visualizer.visualize_topic_embeddings(
        topic_embeddings,
        results['topic_words'],
        filename="topic_embeddings.png"
    )
    
    logger.info("Visualizations complete!")


def generate_pyldavis_visualization(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix,
    vocab: List[str],
    output_path: str,
    mds: str = 'tsne',
    sort_topics: bool = True,
    R: int = 30
) -> Optional[str]:
    """
    Generate interactive pyLDAvis HTML visualization.
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V), can be sparse or dense
        vocab: Vocabulary list
        output_path: Path to save HTML file
        mds: Multidimensional scaling method ('tsne', 'mmds', 'pcoa')
        sort_topics: Whether to sort topics by prevalence
        R: Number of terms to display in barcharts
        
    Returns:
        Path to saved HTML file, or None if pyLDAvis not available
    """
    try:
        import pyLDAvis
    except ImportError:
        logger.warning("pyLDAvis not installed. Install with: pip install pyLDAvis")
        return None
    
    from scipy import sparse
    
    # Convert sparse matrix to dense if needed
    if sparse.issparse(bow_matrix):
        bow_dense = bow_matrix.toarray()
    else:
        bow_dense = np.asarray(bow_matrix)
    
    # Ensure arrays are float64
    theta = np.asarray(theta, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    
    # Normalize beta to ensure each row sums to 1
    beta_normalized = beta / beta.sum(axis=1, keepdims=True)
    
    # Document lengths
    doc_lengths = bow_dense.sum(axis=1).astype(np.int64)
    
    # Term frequency across corpus
    term_frequency = bow_dense.sum(axis=0).astype(np.int64)
    
    # Filter out zero-frequency terms
    nonzero_mask = term_frequency > 0
    if not nonzero_mask.all():
        logger.info(f"Filtering {(~nonzero_mask).sum()} zero-frequency terms")
        term_frequency = term_frequency[nonzero_mask]
        beta_normalized = beta_normalized[:, nonzero_mask]
        vocab = [v for v, m in zip(vocab, nonzero_mask) if m]
    
    try:
        # Create pyLDAvis visualization data
        vis_data = pyLDAvis.prepare(
            topic_term_dists=beta_normalized,
            doc_topic_dists=theta,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            mds=mds,
            sort_topics=sort_topics,
            R=R
        )
        
        # Save to HTML
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pyLDAvis.save_html(vis_data, output_path)
        logger.info(f"pyLDAvis visualization saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate pyLDAvis visualization: {e}")
        return None


def generate_pyldavis_notebook(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix,
    vocab: List[str],
    mds: str = 'tsne'
):
    """
    Generate pyLDAvis visualization for Jupyter notebook display.
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V)
        vocab: Vocabulary list
        mds: Multidimensional scaling method
        
    Returns:
        pyLDAvis prepared data object for notebook display
    """
    try:
        import pyLDAvis
        pyLDAvis.enable_notebook()
    except ImportError:
        logger.warning("pyLDAvis not installed")
        return None
    
    from scipy import sparse
    
    if sparse.issparse(bow_matrix):
        bow_dense = bow_matrix.toarray()
    else:
        bow_dense = np.asarray(bow_matrix)
    
    theta = np.asarray(theta, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    beta_normalized = beta / beta.sum(axis=1, keepdims=True)
    
    doc_lengths = bow_dense.sum(axis=1).astype(np.int64)
    term_frequency = bow_dense.sum(axis=0).astype(np.int64)
    
    nonzero_mask = term_frequency > 0
    if not nonzero_mask.all():
        term_frequency = term_frequency[nonzero_mask]
        beta_normalized = beta_normalized[:, nonzero_mask]
        vocab = [v for v, m in zip(vocab, nonzero_mask) if m]
    
    try:
        vis_data = pyLDAvis.prepare(
            topic_term_dists=beta_normalized,
            doc_topic_dists=theta,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency,
            mds=mds,
            sort_topics=True
        )
        return vis_data
    except Exception as e:
        logger.error(f"Failed to prepare pyLDAvis data: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ETM results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing ETM results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualizations")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Specific timestamp to load")
    parser.add_argument("--no_wordcloud", action="store_true",
                        help="Disable word cloud visualization")
    
    args = parser.parse_args()
    
    visualize_etm_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        show_wordcloud=not args.no_wordcloud
    )

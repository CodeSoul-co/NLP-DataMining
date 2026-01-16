"""
Document-Topic UMAP Visualization (BERTopic Style)

Creates 2D scatter plot of documents colored by their dominant topic,
with ellipse boundaries around topic clusters and topic labels.

This approximates BERTopic's UMAP visualization using ETM's theta matrix
(document-topic distribution) as document representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)

# Configure matplotlib for Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Try to import UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")

# Try to import t-SNE
from sklearn.manifold import TSNE


class DocumentTopicUMAPVisualizer:
    """
    Visualize documents in 2D space colored by their dominant topic.
    
    Similar to BERTopic's visualization where documents cluster by topic
    with ellipse boundaries and topic labels.
    """
    
    def __init__(
        self,
        theta: np.ndarray,
        topic_words: Dict[int, List] = None,
        vocab: List[str] = None,
        output_dir: Optional[str] = None,
        dpi: int = 150
    ):
        """
        Initialize document-topic UMAP visualizer.
        
        Args:
            theta: Document-topic distribution matrix (N x K)
            topic_words: Top words per topic {topic_id: [(word, prob), ...]}
            vocab: Vocabulary list
            output_dir: Directory to save visualizations
            dpi: Figure DPI
        """
        self.theta = theta
        self.topic_words = topic_words
        self.vocab = vocab
        self.output_dir = output_dir
        self.dpi = dpi
        
        self.n_docs = theta.shape[0]
        self.n_topics = theta.shape[1]
        
        # Assign each document to its dominant topic
        self.doc_topics = np.argmax(theta, axis=1)
        
        # Pre-compute 2D embeddings (lazy)
        self._coords_2d = None
        self._reduction_method = None
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def _get_topic_label(self, topic_idx: int, n_words: int = 3) -> str:
        """Get topic label with top words."""
        if self.topic_words and topic_idx in self.topic_words:
            words = self.topic_words[topic_idx]
            if isinstance(words, list):
                if len(words) > 0:
                    if isinstance(words[0], tuple):
                        word_list = [w[0] for w in words[:n_words]]
                    else:
                        word_list = words[:n_words]
                    return f"{topic_idx}_" + "_".join(word_list)
        return f"Topic_{topic_idx}"
    
    def compute_2d_embedding(
        self,
        method: str = 'umap',
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        sample_size: int = None
    ) -> np.ndarray:
        """
        Compute 2D embedding of documents using UMAP/t-SNE/PCA.
        
        Args:
            method: 'umap', 'tsne', or 'pca'
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random seed
            sample_size: If set, sample this many documents for faster computation
            
        Returns:
            2D coordinates (N x 2)
        """
        # Use theta as document representation
        X = self.theta
        
        # Sample if needed
        if sample_size and sample_size < self.n_docs:
            np.random.seed(random_state)
            self._sample_indices = np.random.choice(self.n_docs, sample_size, replace=False)
            X = X[self._sample_indices]
        else:
            self._sample_indices = None
        
        if method == 'umap' and HAS_UMAP:
            reducer = UMAP(
                n_components=2,
                n_neighbors=min(n_neighbors, len(X) - 1),
                min_dist=min_dist,
                random_state=random_state,
                metric='cosine'
            )
            coords_2d = reducer.fit_transform(X)
        elif method == 'tsne':
            perplexity = min(30, len(X) // 4)
            reducer = TSNE(
                n_components=2,
                perplexity=max(5, perplexity),
                random_state=random_state,
                init='pca'
            )
            coords_2d = reducer.fit_transform(X)
        else:  # PCA
            reducer = PCA(n_components=2, random_state=random_state)
            coords_2d = reducer.fit_transform(X)
        
        self._coords_2d = coords_2d
        self._reduction_method = method
        
        return coords_2d
    
    def _compute_topic_ellipse(
        self,
        coords: np.ndarray,
        n_std: float = 2.0
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Compute ellipse parameters for a set of points.
        
        Returns:
            (center_x, center_y, width, height, angle) or None
        """
        if len(coords) < 3:
            return None
        
        try:
            # Compute mean
            mean_x = float(np.mean(coords[:, 0]))
            mean_y = float(np.mean(coords[:, 1]))
            
            # Compute covariance matrix
            cov = np.cov(coords[:, 0], coords[:, 1])
            
            # Handle edge cases
            if cov.ndim == 0 or cov.shape != (2, 2):
                std_x = float(np.std(coords[:, 0]))
                std_y = float(np.std(coords[:, 1]))
                return (mean_x, mean_y, n_std * std_x * 2, n_std * std_y * 2, 0)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue (descending)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            
            # Compute ellipse parameters
            width = float(2 * n_std * np.sqrt(max(eigenvalues[0], 0.01)))
            height = float(2 * n_std * np.sqrt(max(eigenvalues[1], 0.01)))
            angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
            
            return (mean_x, mean_y, width, height, angle)
        except Exception as e:
            logger.debug(f"Ellipse computation failed: {e}")
            return None
    
    def plot_document_clusters(
        self,
        method: str = 'umap',
        sample_size: int = 5000,
        show_ellipses: bool = True,
        show_labels: bool = True,
        n_label_words: int = 3,
        figsize: Tuple[int, int] = (14, 12),
        point_size: int = 8,
        point_alpha: float = 0.5,
        ellipse_alpha: float = 0.15,
        colormap: str = 'tab20',
        filename: Optional[str] = None,
        title: str = None,
        random_state: int = 42
    ) -> plt.Figure:
        """
        Create BERTopic-style document cluster visualization.
        
        Args:
            method: Dimensionality reduction method ('umap', 'tsne', 'pca')
            sample_size: Number of documents to sample (for large datasets)
            show_ellipses: Draw ellipse boundaries around clusters
            show_labels: Show topic labels
            n_label_words: Number of words in topic labels
            figsize: Figure size
            point_size: Scatter point size
            point_alpha: Point transparency
            ellipse_alpha: Ellipse fill transparency
            colormap: Matplotlib colormap
            filename: Output filename
            title: Plot title
            random_state: Random seed
            
        Returns:
            Figure
        """
        # Compute 2D embedding
        coords_2d = self.compute_2d_embedding(
            method=method,
            sample_size=sample_size,
            random_state=random_state
        )
        
        # Get document topics for sampled documents
        if self._sample_indices is not None:
            doc_topics = self.doc_topics[self._sample_indices]
        else:
            doc_topics = self.doc_topics
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Get colormap
        cmap = plt.cm.get_cmap(colormap)
        
        # Get unique topics
        unique_topics = np.unique(doc_topics)
        n_unique = len(unique_topics)
        
        # Plot each topic's documents
        for i, topic_idx in enumerate(unique_topics):
            mask = doc_topics == topic_idx
            topic_coords = coords_2d[mask]
            
            if len(topic_coords) == 0:
                continue
            
            color = cmap(i / max(n_unique - 1, 1))
            
            # Plot points
            ax.scatter(
                topic_coords[:, 0], topic_coords[:, 1],
                color=color, s=point_size, alpha=point_alpha,
                label=f'Topic {topic_idx}', zorder=2
            )
            
            # Draw ellipse boundary
            if show_ellipses and len(topic_coords) >= 3:
                ellipse_params = self._compute_topic_ellipse(topic_coords, n_std=2.0)
                if ellipse_params:
                    cx, cy, width, height, angle = ellipse_params
                    ellipse = Ellipse(
                        (cx, cy), width, height, angle=angle,
                        facecolor=to_rgba(color, ellipse_alpha),
                        edgecolor=to_rgba(color, 0.6),
                        linewidth=1.5,
                        zorder=1
                    )
                    ax.add_patch(ellipse)
            
            # Add topic label
            if show_labels and len(topic_coords) >= 1:
                # Place label at cluster center
                center_x = np.mean(topic_coords[:, 0])
                center_y = np.mean(topic_coords[:, 1])
                
                label_text = self._get_topic_label(topic_idx, n_label_words)
                
                ax.annotate(
                    label_text,
                    (center_x, center_y),
                    fontsize=9,
                    fontweight='bold',
                    color=color,
                    ha='center',
                    va='center',
                    zorder=3,
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor=color,
                        alpha=0.85,
                        linewidth=1
                    )
                )
        
        # Add dimension labels
        ax.set_xlabel('D1', fontsize=12)
        ax.set_ylabel('D2', fontsize=12)
        
        # Clean up axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        else:
            method_name = {'umap': 'UMAP', 'tsne': 't-SNE', 'pca': 'PCA'}
            ax.set_title(
                f'文档主题二维空间分布图及聚类 ({method_name.get(method, method)})',
                fontsize=14, fontweight='bold', pad=15
            )
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def plot_topic_density(
        self,
        method: str = 'umap',
        sample_size: int = 5000,
        figsize: Tuple[int, int] = (12, 10),
        filename: Optional[str] = None,
        title: str = None,
        random_state: int = 42
    ) -> plt.Figure:
        """
        Create density-based visualization of document clusters.
        
        Args:
            method: Dimensionality reduction method
            sample_size: Number of documents to sample
            figsize: Figure size
            filename: Output filename
            title: Plot title
            random_state: Random seed
            
        Returns:
            Figure
        """
        # Compute 2D embedding
        coords_2d = self.compute_2d_embedding(
            method=method,
            sample_size=sample_size,
            random_state=random_state
        )
        
        # Get document topics
        if self._sample_indices is not None:
            doc_topics = self.doc_topics[self._sample_indices]
        else:
            doc_topics = self.doc_topics
        
        # Create figure with hexbin
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Create hexbin plot
        hb = ax.hexbin(
            coords_2d[:, 0], coords_2d[:, 1],
            C=doc_topics,
            reduce_C_function=lambda x: np.bincount(x.astype(int)).argmax() if len(x) > 0 else 0,
            gridsize=30,
            cmap='tab20',
            mincnt=1
        )
        
        # Add colorbar
        cb = plt.colorbar(hb, ax=ax, label='Dominant Topic')
        
        ax.set_xlabel('D1', fontsize=12)
        ax.set_ylabel('D2', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Document Topic Density Map', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def _save_or_show(self, fig, filename: Optional[str] = None):
        """Save figure or show it."""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Figure saved to {filepath}")
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return None


def plot_document_topic_umap(
    theta: np.ndarray,
    topic_words: Dict = None,
    method: str = 'umap',
    sample_size: int = 5000,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function to create document-topic UMAP visualization.
    
    Args:
        theta: Document-topic distribution (N x K)
        topic_words: Top words per topic
        method: Dimensionality reduction method
        sample_size: Number of documents to sample
        output_dir: Output directory
        filename: Output filename
        **kwargs: Additional arguments
        
    Returns:
        Figure
    """
    visualizer = DocumentTopicUMAPVisualizer(
        theta=theta,
        topic_words=topic_words,
        output_dir=output_dir
    )
    
    return visualizer.plot_document_clusters(
        method=method,
        sample_size=sample_size,
        filename=filename,
        **kwargs
    )

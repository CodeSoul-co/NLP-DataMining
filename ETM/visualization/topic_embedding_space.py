"""
Topic-Word Embedding Space Visualization (ETM Paper Style)

Creates ETM paper-style visualization showing topics and nearby words
in the same 2D embedding space.

Reference: "Topic Modeling in Embedding Spaces" (Dieng et al., 2020)
- Figure 2: A topic about Christianity found by ETM
- Figure 3: Topics about sports found by ETM
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)

# Try to import adjustText for label repulsion
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    logger.info("adjustText not available. Install with: pip install adjustText")

# Try to import UMAP (optional)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


class TopicEmbeddingSpaceVisualizer:
    """
    Visualize topics and words in the same embedding space.
    
    Shows topics as points in the word embedding space with
    nearby words, replicating ETM paper Figures 2 and 3.
    """
    
    def __init__(
        self,
        topic_embeddings: np.ndarray,
        word_embeddings: np.ndarray = None,
        beta: np.ndarray = None,
        vocab: List[str] = None,
        topic_words: Dict[int, List[Tuple[str, float]]] = None,
        output_dir: Optional[str] = None,
        dpi: int = 150
    ):
        """
        Initialize topic embedding space visualizer.
        
        Args:
            topic_embeddings: Topic embeddings (K x E), alpha in ETM
            word_embeddings: Word embeddings (V x E), rho in ETM (optional)
            beta: Topic-word distribution (K x V), used if word_embeddings not provided
            vocab: Vocabulary list
            topic_words: Pre-computed top words per topic
            output_dir: Directory to save visualizations
            dpi: Figure DPI
        """
        self.topic_embeddings = topic_embeddings
        self.word_embeddings = word_embeddings
        self.beta = beta
        self.vocab = vocab if isinstance(vocab, list) else list(vocab) if vocab else None
        self.topic_words = topic_words
        self.output_dir = output_dir
        self.dpi = dpi
        
        self.num_topics = topic_embeddings.shape[0]
        self.embedding_dim = topic_embeddings.shape[1]
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def get_topic_nearby_words(
        self,
        topic_idx: int,
        n_words: int = 30,
        method: str = 'beta'
    ) -> List[Tuple[str, int]]:
        """
        Get nearby words for a topic.
        
        Args:
            topic_idx: Topic index
            n_words: Number of words to retrieve
            method: 'beta' (use beta weights) or 'embedding' (use cosine similarity)
            
        Returns:
            List of (word, word_idx) tuples
        """
        if method == 'beta' and self.beta is not None:
            # Use beta weights to find top words
            top_indices = np.argsort(self.beta[topic_idx])[-n_words:][::-1]
            return [(self.vocab[idx], idx) for idx in top_indices]
        
        elif method == 'embedding' and self.word_embeddings is not None:
            # Use cosine similarity to topic embedding
            topic_emb = self.topic_embeddings[topic_idx]
            topic_emb = topic_emb / np.linalg.norm(topic_emb)
            
            word_norms = np.linalg.norm(self.word_embeddings, axis=1, keepdims=True)
            word_emb_normalized = self.word_embeddings / (word_norms + 1e-8)
            
            similarities = word_emb_normalized @ topic_emb
            top_indices = np.argsort(similarities)[-n_words:][::-1]
            return [(self.vocab[idx], idx) for idx in top_indices]
        
        elif self.topic_words is not None:
            # Use pre-computed topic words
            words = self.topic_words[topic_idx][:n_words]
            result = []
            for w in words:
                word = w[0] if isinstance(w, (tuple, list)) else w
                if self.vocab and word in self.vocab:
                    result.append((word, self.vocab.index(word)))
                else:
                    result.append((word, -1))
            return result
        
        return []
    
    def compute_2d_embedding(
        self,
        topic_indices: List[int],
        n_words_per_topic: int = 30,
        method: str = 'tsne',
        perplexity: int = 30,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """
        Compute 2D embedding for topics and their nearby words.
        
        Args:
            topic_indices: Topics to visualize
            n_words_per_topic: Words per topic
            method: 'tsne', 'umap', or 'pca'
            perplexity: t-SNE perplexity
            random_state: Random seed
            
        Returns:
            topic_coords: 2D coordinates for topics (n_topics x 2)
            word_coords: 2D coordinates for words (n_words x 2)
            word_labels: Word labels
            word_topic_assignments: Which topic each word belongs to
        """
        # Collect all words and their embeddings
        all_words = []
        word_indices = []
        word_topic_assignments = []
        
        for topic_idx in topic_indices:
            nearby_words = self.get_topic_nearby_words(topic_idx, n_words_per_topic)
            for word, word_idx in nearby_words:
                if word not in all_words and word_idx >= 0:
                    all_words.append(word)
                    word_indices.append(word_idx)
                    word_topic_assignments.append(topic_idx)
        
        # Build combined embedding matrix
        # Topics first, then words
        n_topics = len(topic_indices)
        n_words = len(all_words)
        
        if self.word_embeddings is not None:
            # Use actual word embeddings
            word_embs = self.word_embeddings[word_indices]
        else:
            # Approximate: use random embeddings based on topic similarity
            # This is a fallback when word embeddings are not available
            np.random.seed(random_state)
            word_embs = np.zeros((n_words, self.embedding_dim))
            for i, (word_idx, topic_idx) in enumerate(zip(word_indices, word_topic_assignments)):
                # Place word near its topic with some noise
                topic_emb = self.topic_embeddings[topic_idx]
                noise = np.random.randn(self.embedding_dim) * 0.3
                word_embs[i] = topic_emb + noise
        
        topic_embs = self.topic_embeddings[topic_indices]
        
        # Combine: [topics; words]
        combined = np.vstack([topic_embs, word_embs])
        
        # Dimensionality reduction
        if method == 'tsne':
            # Adjust perplexity for small datasets
            effective_perplexity = min(perplexity, max(5, (n_topics + n_words) // 4))
            reducer = TSNE(
                n_components=2,
                perplexity=effective_perplexity,
                random_state=random_state,
                init='pca'
            )
            coords_2d = reducer.fit_transform(combined)
        
        elif method == 'umap' and HAS_UMAP:
            n_neighbors = min(15, max(2, (n_topics + n_words) // 3))
            reducer = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords_2d = reducer.fit_transform(combined)
        
        else:  # PCA
            reducer = PCA(n_components=2, random_state=random_state)
            coords_2d = reducer.fit_transform(combined)
        
        topic_coords = coords_2d[:n_topics]
        word_coords = coords_2d[n_topics:]
        
        return topic_coords, word_coords, all_words, word_topic_assignments
    
    def plot_topic_embedding_space(
        self,
        topic_indices: List[int],
        n_words_per_topic: int = 30,
        n_labeled_words: int = 20,
        method: str = 'pca',
        show_convex_hull: bool = False,
        figsize: Tuple[int, int] = (10, 10),
        filename: Optional[str] = None,
        title: str = None,
        n_background_samples: int = 0,
        random_state: int = 42
    ) -> plt.Figure:
        """
        Create ETM paper-style topic embedding space visualization.
        
        TRUE LOCAL VISUALIZATION PIPELINE:
        1. Construct local point set S = {topic points} ∪ {topN nearby words}
        2. Perform 2D dimensionality reduction ONLY on S (not global)
        3. Force axis limits: xlim/ylim based on S with 8% padding
        4. Optional: sample few background points (not in axis calculation)
        
        Args:
            topic_indices: Topics to visualize (recommend 1-2)
            n_words_per_topic: Words per topic for local set S
            n_labeled_words: Words to label (should be <= n_words_per_topic)
            method: Dimensionality reduction ('pca', 'tsne', 'umap')
            show_convex_hull: Draw light convex hull
            figsize: Figure size
            filename: Output filename
            title: Plot title
            n_background_samples: Number of random background samples (0 = none)
            random_state: Random seed for reproducibility
            
        Returns:
            Figure
        """
        np.random.seed(random_state)
        
        # ========== STEP 1: Construct local point set S ==========
        # S = {topic embeddings} ∪ {topN nearby word embeddings for each topic}
        
        local_words = []  # (word, word_idx, topic_idx)
        for topic_idx in topic_indices:
            nearby = self.get_topic_nearby_words(topic_idx, n_words_per_topic)
            for word, word_idx in nearby:
                if word_idx >= 0 and word not in [w[0] for w in local_words]:
                    local_words.append((word, word_idx, topic_idx))
        
        # Get embeddings for local set S
        topic_embs = self.topic_embeddings[topic_indices]  # (n_topics, E)
        
        if self.word_embeddings is not None:
            word_embs = self.word_embeddings[[w[1] for w in local_words]]
        else:
            # Fallback: place words near their topic with noise
            word_embs = np.zeros((len(local_words), self.embedding_dim))
            for i, (word, word_idx, topic_idx) in enumerate(local_words):
                topic_emb = self.topic_embeddings[topic_idx]
                noise = np.random.randn(self.embedding_dim) * 0.3
                word_embs[i] = topic_emb + noise
        
        # Combine: S = [topics; words]
        S = np.vstack([topic_embs, word_embs])
        n_topics = len(topic_indices)
        n_words = len(local_words)
        
        # ========== STEP 2: Local 2D dimensionality reduction on S only ==========
        if method == 'umap' and HAS_UMAP:
            n_neighbors = min(15, max(2, len(S) // 3))
            reducer = UMAP(n_components=2, n_neighbors=n_neighbors, 
                          random_state=random_state, min_dist=0.1)
            S_2d = reducer.fit_transform(S)
        elif method == 'tsne':
            perplexity = min(30, max(5, len(S) // 4))
            reducer = TSNE(n_components=2, perplexity=perplexity, 
                          random_state=random_state, init='pca')
            S_2d = reducer.fit_transform(S)
        else:  # PCA
            reducer = PCA(n_components=2, random_state=random_state)
            S_2d = reducer.fit_transform(S)
        
        topic_coords = S_2d[:n_topics]
        word_coords = S_2d[n_topics:]
        word_labels = [w[0] for w in local_words]
        word_topic_assignments = [w[2] for w in local_words]
        
        # ========== STEP 3: Calculate axis limits from S only (5% padding) ==========
        x_min, y_min = S_2d.min(axis=0)
        x_max, y_max = S_2d.max(axis=0)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        pad = 0.05  # 5% padding (tighter)
        xlim = (x_min - x_range * pad, x_max + x_range * pad)
        ylim = (y_min - y_range * pad, y_max + y_range * pad)
        
        # ========== Create figure ==========
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Force axis limits (do NOT use set_aspect('equal'))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Colors for different topics
        topic_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
        
        # ========== STEP 4: Optional background samples (not in axis calc) ==========
        if n_background_samples > 0 and self.word_embeddings is not None:
            # Sample random words NOT in local set
            local_word_indices = set([w[1] for w in local_words])
            all_indices = set(range(len(self.vocab)))
            bg_candidates = list(all_indices - local_word_indices)
            n_bg = min(n_background_samples, len(bg_candidates))
            if n_bg > 0:
                bg_indices = np.random.choice(bg_candidates, n_bg, replace=False)
                bg_embs = self.word_embeddings[bg_indices]
                # Project background using same reducer (transform if possible)
                if hasattr(reducer, 'transform'):
                    bg_2d = reducer.transform(bg_embs)
                else:
                    # For t-SNE, just skip background
                    bg_2d = None
                if bg_2d is not None:
                    ax.scatter(bg_2d[:, 0], bg_2d[:, 1], 
                              c='#DDDDDD', s=2, alpha=0.1, zorder=0)
        
        # ========== Draw convex hulls (light decoration) ==========
        if show_convex_hull:
            for i, topic_idx in enumerate(topic_indices):
                topic_word_mask = [t == topic_idx for t in word_topic_assignments]
                topic_word_coords = word_coords[topic_word_mask]
                if len(topic_word_coords) >= 3:
                    try:
                        hull = ConvexHull(topic_word_coords)
                        hull_points = topic_word_coords[hull.vertices]
                        color = topic_colors[i % len(topic_colors)]
                        polygon = Polygon(
                            hull_points, alpha=0.06, facecolor=color,
                            edgecolor=color, linewidth=0.6, linestyle='--'
                        )
                        ax.add_patch(polygon)
                    except:
                        pass
        
        # ========== Plot words (small dots) ==========
        for j in range(n_words):
            topic_idx = word_topic_assignments[j]
            i = topic_indices.index(topic_idx)
            color = topic_colors[i % len(topic_colors)]
            ax.scatter(word_coords[j, 0], word_coords[j, 1], 
                      c=color, s=20, alpha=0.5, zorder=2, 
                      edgecolor='white', linewidth=0.3)
        
        # ========== Plot topics (large anchor points) ==========
        for i, topic_idx in enumerate(topic_indices):
            color = topic_colors[i % len(topic_colors)]
            # Outer glow
            ax.scatter(topic_coords[i, 0], topic_coords[i, 1],
                      c=color, s=500, alpha=0.12, zorder=3)
            # Main point
            ax.scatter(topic_coords[i, 0], topic_coords[i, 1],
                      c=color, s=250, zorder=5, edgecolor='white', linewidth=2)
            # Topic label
            ax.annotate(
                f'Topic {topic_idx}',
                (topic_coords[i, 0], topic_coords[i, 1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', color=color, zorder=7,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                         edgecolor=color, linewidth=1.2, alpha=0.95)
            )
        
        # ========== Label words with adjustText ==========
        # Only label n_labeled_words closest to each topic
        texts = []
        
        for i, topic_idx in enumerate(topic_indices):
            topic_pos = topic_coords[i]
            # Get words for this topic, sorted by distance
            word_dists = []
            for j in range(n_words):
                if word_topic_assignments[j] == topic_idx:
                    dist = np.linalg.norm(word_coords[j] - topic_pos)
                    word_dists.append((j, dist))
            word_dists.sort(key=lambda x: x[1])
            
            # Label fewer words to reduce clutter
            n_to_label = min(n_labeled_words, len(word_dists))
            for j, _ in word_dists[:n_to_label]:
                x, y = word_coords[j]
                word = word_labels[j]
                txt = ax.text(x, y, f'•{word}', fontsize=10, color='#2C3E50', 
                             alpha=0.9, ha='center', va='center', zorder=4)
                texts.append(txt)
        
        # Use adjustText for label repulsion (stronger force to spread out)
        if HAS_ADJUST_TEXT and texts:
            try:
                adjust_text(texts, ax=ax,
                           arrowprops=dict(arrowstyle='-', color='gray', alpha=0.2, lw=0.3),
                           expand_points=(2.0, 2.0),  # More expansion
                           force_text=(0.8, 0.8),     # Stronger text repulsion
                           force_points=(0.5, 0.5),   # Stronger point repulsion
                           lim=500)                   # More iterations
            except:
                pass
        
        # ========== Clean up axes ==========
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            topic_str = ', '.join([f'Topic {t}' for t in topic_indices])
            ax.set_title(f'{topic_str} in Word Embedding Space',
                        fontsize=14, fontweight='bold', pad=20)
        
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


def plot_topic_embedding_space(
    topic_embeddings: np.ndarray,
    topic_indices: List[int],
    word_embeddings: np.ndarray = None,
    beta: np.ndarray = None,
    vocab: List[str] = None,
    topic_words: Dict = None,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function to create topic embedding space visualization.
    
    Args:
        topic_embeddings: Topic embeddings (K x E)
        topic_indices: Topics to visualize
        word_embeddings: Word embeddings (V x E), optional
        beta: Topic-word distribution (K x V)
        vocab: Vocabulary
        topic_words: Pre-computed topic words
        output_dir: Output directory
        filename: Output filename
        **kwargs: Additional arguments
        
    Returns:
        Figure
    """
    visualizer = TopicEmbeddingSpaceVisualizer(
        topic_embeddings=topic_embeddings,
        word_embeddings=word_embeddings,
        beta=beta,
        vocab=vocab,
        topic_words=topic_words,
        output_dir=output_dir
    )
    
    return visualizer.plot_topic_embedding_space(
        topic_indices=topic_indices,
        filename=filename,
        **kwargs
    )


if __name__ == '__main__':
    # Example usage with simulated data
    np.random.seed(42)
    
    n_topics, n_vocab, embedding_dim = 20, 1000, 128
    
    # Simulate embeddings
    topic_embeddings = np.random.randn(n_topics, embedding_dim)
    word_embeddings = np.random.randn(n_vocab, embedding_dim)
    beta = np.random.dirichlet(np.ones(n_vocab) * 0.1, n_topics)
    vocab = [f"word_{i}" for i in range(n_vocab)]
    
    # Create visualization
    output_dir = './test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = TopicEmbeddingSpaceVisualizer(
        topic_embeddings=topic_embeddings,
        word_embeddings=word_embeddings,
        beta=beta,
        vocab=vocab,
        output_dir=output_dir
    )
    
    # Single topic
    visualizer.plot_topic_embedding_space(
        topic_indices=[0],
        n_words_per_topic=30,
        n_labeled_words=20,
        filename='topic_embedding_single.png'
    )
    
    # Multiple topics
    visualizer.plot_topic_embedding_space(
        topic_indices=[0, 1, 2],
        n_words_per_topic=30,
        n_labeled_words=15,
        show_convex_hull=True,
        filename='topic_embedding_multi.png'
    )
    
    print("Topic embedding space visualizations saved!")

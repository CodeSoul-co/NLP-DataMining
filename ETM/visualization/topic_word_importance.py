"""
Topic Word Importance Visualization (c-TF-IDF Style)

Creates horizontal bar charts showing top-N words per topic with importance scores,
similar to c-TF-IDF visualization style used in BERTopic and other topic models.

The importance score is computed as an approximation of c-TF-IDF using ETM's beta matrix:
- TF component: beta[k, w] (topic-word probability)
- IDF component: log(K / sum(beta[:, w] > threshold)) (inverse topic frequency)
- c-TF-IDF ≈ beta[k, w] * IDF[w]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)

# Configure matplotlib for Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TopicWordImportanceVisualizer:
    """
    Visualize topic word importance using c-TF-IDF style horizontal bar charts.
    """
    
    def __init__(
        self,
        beta: np.ndarray,
        vocab: List[str],
        topic_words: Dict[int, List] = None,
        output_dir: Optional[str] = None,
        dpi: int = 150
    ):
        """
        Initialize topic word importance visualizer.
        
        Args:
            beta: Topic-word distribution matrix (K x V)
            vocab: Vocabulary list
            topic_words: Pre-computed top words per topic (optional)
            output_dir: Directory to save visualizations
            dpi: Figure DPI
        """
        self.beta = beta
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.topic_words = topic_words
        self.output_dir = output_dir
        self.dpi = dpi
        
        self.num_topics = beta.shape[0]
        self.vocab_size = beta.shape[1]
        
        # Pre-compute c-TF-IDF scores
        self.ctfidf_scores = self._compute_ctfidf()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def _compute_ctfidf(self, threshold: float = 0.001) -> np.ndarray:
        """
        Compute approximate c-TF-IDF scores from ETM beta matrix.
        
        c-TF-IDF formula:
        - TF: beta[k, w] (word probability in topic k)
        - IDF: log(K / (1 + count of topics where beta[:, w] > threshold))
        - c-TF-IDF = TF * IDF
        
        Args:
            threshold: Minimum beta value to consider word present in topic
            
        Returns:
            c-TF-IDF scores matrix (K x V)
        """
        K = self.num_topics
        
        # Count how many topics each word appears in (above threshold)
        word_topic_count = np.sum(self.beta > threshold, axis=0) + 1  # +1 for smoothing
        
        # IDF component
        idf = np.log(K / word_topic_count)
        
        # c-TF-IDF = beta * IDF
        ctfidf = self.beta * idf
        
        return ctfidf
    
    def get_top_words(
        self,
        topic_idx: int,
        n_words: int = 10,
        use_ctfidf: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get top words for a topic with their importance scores.
        
        Args:
            topic_idx: Topic index
            n_words: Number of top words
            use_ctfidf: Use c-TF-IDF scores (True) or raw beta (False)
            
        Returns:
            List of (word, score) tuples
        """
        if use_ctfidf:
            scores = self.ctfidf_scores[topic_idx]
        else:
            scores = self.beta[topic_idx]
        
        # Get top word indices
        top_indices = np.argsort(scores)[::-1][:n_words]
        
        result = []
        for idx in top_indices:
            word = self.vocab[idx] if idx < len(self.vocab) else f"word_{idx}"
            result.append((word, scores[idx]))
        
        return result
    
    def plot_topic_word_bars(
        self,
        topic_indices: List[int] = None,
        n_words: int = 5,
        n_cols: int = 3,
        figsize: Tuple[int, int] = None,
        use_ctfidf: bool = True,
        bar_color: str = '#2C3E50',
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Create horizontal bar charts for multiple topics (grid layout).
        
        Args:
            topic_indices: Topics to visualize (default: all)
            n_words: Number of top words per topic
            n_cols: Number of columns in grid
            figsize: Figure size (auto-calculated if None)
            use_ctfidf: Use c-TF-IDF scores
            bar_color: Color for bars
            filename: Output filename
            title: Overall title
            
        Returns:
            Figure
        """
        if topic_indices is None:
            topic_indices = list(range(self.num_topics))
        
        n_topics = len(topic_indices)
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        # Auto-calculate figure size
        if figsize is None:
            figsize = (4 * n_cols, 2.5 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor='white')
        
        # Flatten axes for easy iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_flat = axes.flatten()
        
        for i, topic_idx in enumerate(topic_indices):
            ax = axes_flat[i]
            
            # Get top words
            top_words = self.get_top_words(topic_idx, n_words, use_ctfidf)
            words = [w for w, _ in top_words][::-1]  # Reverse for horizontal bar
            scores = [s for _, s in top_words][::-1]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(words))
            ax.barh(y_pos, scores, color=bar_color, edgecolor='none', height=0.7)
            
            # Set labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=10)
            ax.set_xlabel('权重', fontsize=9)
            
            # Title for each subplot
            ax.set_title(f'({chr(97 + i)}) Topic{topic_idx}', fontsize=11, pad=5)
            
            # Clean up
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', length=0)
            
            # Set x-axis limits with some padding
            max_score = max(scores) if scores else 1
            ax.set_xlim(0, max_score * 1.1)
        
        # Hide unused subplots
        for i in range(n_topics, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def plot_single_topic_bars(
        self,
        topic_idx: int,
        n_words: int = 10,
        figsize: Tuple[int, int] = (8, 6),
        use_ctfidf: bool = True,
        bar_color: str = '#2C3E50',
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Create horizontal bar chart for a single topic.
        
        Args:
            topic_idx: Topic index
            n_words: Number of top words
            figsize: Figure size
            use_ctfidf: Use c-TF-IDF scores
            bar_color: Color for bars
            filename: Output filename
            title: Plot title
            
        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Get top words
        top_words = self.get_top_words(topic_idx, n_words, use_ctfidf)
        words = [w for w, _ in top_words][::-1]
        scores = [s for _, s in top_words][::-1]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(words))
        bars = ax.barh(y_pos, scores, color=bar_color, edgecolor='none', height=0.6)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + max(scores) * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', va='center', fontsize=9, color='#666666')
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=11)
        ax.set_xlabel('权重 (c-TF-IDF Score)', fontsize=10)
        
        # Title
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        else:
            ax.set_title(f'Topic {topic_idx} - Top {n_words} Keywords', 
                        fontsize=13, fontweight='bold', pad=10)
        
        # Clean up
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        
        # Set x-axis limits
        max_score = max(scores) if scores else 1
        ax.set_xlim(0, max_score * 1.25)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def plot_comparative_bars(
        self,
        topic_indices: List[int],
        n_words: int = 5,
        figsize: Tuple[int, int] = None,
        colormap: str = 'tab10',
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Create comparative bar chart showing top words across multiple topics.
        
        Args:
            topic_indices: Topics to compare
            n_words: Number of top words per topic
            figsize: Figure size
            colormap: Matplotlib colormap name
            filename: Output filename
            title: Plot title
            
        Returns:
            Figure
        """
        n_topics = len(topic_indices)
        
        if figsize is None:
            figsize = (10, 3 * n_topics)
        
        fig, axes = plt.subplots(n_topics, 1, figsize=figsize, facecolor='white')
        
        if n_topics == 1:
            axes = [axes]
        
        cmap = plt.cm.get_cmap(colormap)
        
        for i, topic_idx in enumerate(topic_indices):
            ax = axes[i]
            color = cmap(i / max(n_topics - 1, 1))
            
            # Get top words
            top_words = self.get_top_words(topic_idx, n_words, use_ctfidf=True)
            words = [w for w, _ in top_words][::-1]
            scores = [s for _, s in top_words][::-1]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(words))
            ax.barh(y_pos, scores, color=color, edgecolor='none', height=0.6)
            
            # Set labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=10)
            ax.set_xlabel('权重', fontsize=9)
            ax.set_title(f'Topic {topic_idx}', fontsize=11, fontweight='bold', loc='left')
            
            # Clean up
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', length=0)
            
            max_score = max(scores) if scores else 1
            ax.set_xlim(0, max_score * 1.1)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
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


def plot_topic_word_importance(
    beta: np.ndarray,
    vocab: List[str],
    topic_indices: List[int] = None,
    n_words: int = 5,
    n_cols: int = 3,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function to create topic word importance bar charts.
    
    Args:
        beta: Topic-word distribution matrix (K x V)
        vocab: Vocabulary list
        topic_indices: Topics to visualize
        n_words: Number of top words per topic
        n_cols: Number of columns in grid
        output_dir: Output directory
        filename: Output filename
        **kwargs: Additional arguments
        
    Returns:
        Figure
    """
    visualizer = TopicWordImportanceVisualizer(
        beta=beta,
        vocab=vocab,
        output_dir=output_dir
    )
    
    return visualizer.plot_topic_word_bars(
        topic_indices=topic_indices,
        n_words=n_words,
        n_cols=n_cols,
        filename=filename,
        **kwargs
    )

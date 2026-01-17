"""
Dynamic Topic Evolution Visualization (DTM-style)

Creates comprehensive topic evolution plots approximating DTM:
- Top: Word evolution table (top words per time period)
- Middle: Topic proportion curves with smoothing
- Bottom: Event/annotation markers

Reference: Dynamic Topic Models (Blei & Lafferty, 2006)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter1d
import logging
import os

logger = logging.getLogger(__name__)


class DynamicTopicEvolutionVisualizer:
    """
    Visualize dynamic topic evolution approximating DTM.
    
    Uses ETM outputs aggregated by time slices with smoothing
    to approximate DTM's dynamic topic evolution.
    """
    
    def __init__(
        self,
        theta: np.ndarray,
        beta: np.ndarray,
        timestamps: np.ndarray,
        vocab: List[str],
        topic_words: Dict[int, List[Tuple[str, float]]] = None,
        output_dir: Optional[str] = None,
        dpi: int = 150
    ):
        """
        Initialize dynamic topic evolution visualizer.
        
        Args:
            theta: Document-topic distribution (N x K)
            beta: Topic-word distribution (K x V)
            timestamps: Document timestamps (N,) - years or dates
            vocab: Vocabulary list
            topic_words: Pre-computed top words per topic
            output_dir: Directory to save visualizations
            dpi: Figure DPI
        """
        self.theta = theta
        self.beta = beta
        # Convert timestamps to years if they are datetime objects
        timestamps = np.array(timestamps) if not isinstance(timestamps, np.ndarray) else timestamps
        if hasattr(timestamps[0], 'year'):
            self.timestamps = np.array([t.year for t in timestamps])
        else:
            self.timestamps = timestamps.astype(int) if timestamps.dtype != int else timestamps
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.topic_words = topic_words
        self.output_dir = output_dir
        self.dpi = dpi
        
        self.num_docs, self.num_topics = theta.shape
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def aggregate_by_time(
        self,
        time_bins: List[int] = None,
        bin_size: int = 10
    ) -> Dict:
        """
        Aggregate topic proportions by time bins.
        
        Args:
            time_bins: Explicit time bin edges (e.g., [1880, 1890, ...])
            bin_size: Size of each bin if time_bins not provided
            
        Returns:
            Dict with time_labels, topic_proportions, word_evolution
        """
        # Determine time bins
        min_time = int(self.timestamps.min())
        max_time = int(self.timestamps.max())
        
        if time_bins is None:
            time_bins = list(range(min_time, max_time + bin_size, bin_size))
        
        n_bins = len(time_bins) - 1
        time_labels = [f"{time_bins[i]}" for i in range(n_bins)]
        
        # Aggregate topic proportions per bin
        topic_proportions = np.zeros((n_bins, self.num_topics))
        doc_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (self.timestamps >= time_bins[i]) & (self.timestamps < time_bins[i+1])
            if mask.sum() > 0:
                topic_proportions[i] = self.theta[mask].mean(axis=0)
                doc_counts[i] = mask.sum()
        
        # Get top words per topic per time bin (approximate word evolution)
        word_evolution = {}
        for topic_idx in range(self.num_topics):
            word_evolution[topic_idx] = []
            for i in range(n_bins):
                # Use static beta (ETM doesn't have time-varying beta)
                # But we can weight by document proportions in that time
                mask = (self.timestamps >= time_bins[i]) & (self.timestamps < time_bins[i+1])
                if mask.sum() > 0:
                    # Get top words from beta
                    top_indices = np.argsort(self.beta[topic_idx])[-5:][::-1]
                    top_words = [self.vocab[idx] for idx in top_indices]
                    word_evolution[topic_idx].append(top_words)
                else:
                    word_evolution[topic_idx].append([])
        
        return {
            'time_bins': time_bins,
            'time_labels': time_labels,
            'topic_proportions': topic_proportions,
            'doc_counts': doc_counts,
            'word_evolution': word_evolution
        }
    
    def smooth_proportions(
        self,
        proportions: np.ndarray,
        sigma: float = 1.0
    ) -> np.ndarray:
        """
        Apply Gaussian smoothing to topic proportions.
        
        Args:
            proportions: Topic proportions (n_bins x n_topics)
            sigma: Gaussian smoothing parameter
            
        Returns:
            Smoothed proportions
        """
        smoothed = np.zeros_like(proportions)
        for topic_idx in range(proportions.shape[1]):
            smoothed[:, topic_idx] = gaussian_filter1d(
                proportions[:, topic_idx], sigma=sigma
            )
        return smoothed
    
    def plot_topic_evolution(
        self,
        topic_idx: int,
        time_bins: List[int] = None,
        bin_size: int = 10,
        smooth_sigma: float = 1.0,
        events: List[Dict] = None,
        n_words: int = 5,
        filename: Optional[str] = None,
        title: str = None,
        colors: List[str] = None
    ) -> plt.Figure:
        """
        Create DTM-style topic evolution plot for a single topic.
        
        Layout:
        - Top: Word evolution table
        - Middle: Topic proportion curve with smoothing
        - Bottom: Event annotations
        
        Args:
            topic_idx: Topic index to visualize
            time_bins: Time bin edges
            bin_size: Bin size if time_bins not provided
            smooth_sigma: Smoothing parameter
            events: List of event dicts with 'year', 'label', 'y_offset'
            n_words: Number of top words per time period
            filename: Output filename
            title: Plot title
            colors: Colors for different elements
            
        Returns:
            Figure
        """
        # Aggregate data
        data = self.aggregate_by_time(time_bins, bin_size)
        time_labels = data['time_labels']
        proportions = data['topic_proportions'][:, topic_idx]
        word_evolution = data['word_evolution'][topic_idx]
        
        # Smooth proportions
        smoothed = gaussian_filter1d(proportions, sigma=smooth_sigma)
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(14, 8), facecolor='#F5E6D3')  # Beige background
        gs = GridSpec(3, 1, height_ratios=[1.5, 2, 0.8], hspace=0.3)
        
        # ===== Top: Word Evolution Table =====
        ax_words = fig.add_subplot(gs[0])
        ax_words.set_facecolor('#F5E6D3')
        self._draw_word_evolution_table(ax_words, time_labels, word_evolution, n_words)
        
        # ===== Middle: Topic Proportion Curve =====
        ax_curve = fig.add_subplot(gs[1])
        ax_curve.set_facecolor('#F5E6D3')
        
        x = np.arange(len(time_labels))
        
        # Plot smoothed curve
        color = colors[0] if colors else '#E74C3C'  # Red default
        ax_curve.plot(x, smoothed, color=color, linewidth=2.5, label='Topic proportion')
        ax_curve.fill_between(x, 0, smoothed, alpha=0.2, color=color)
        
        # Add markers at data points
        ax_curve.scatter(x, proportions, color=color, s=30, zorder=5, edgecolor='white')
        
        ax_curve.set_xlim(-0.5, len(time_labels) - 0.5)
        ax_curve.set_ylim(0, max(smoothed.max() * 1.2, 0.1))
        ax_curve.set_xticks(x)
        ax_curve.set_xticklabels(time_labels, fontsize=8)
        ax_curve.set_ylabel('Proportion of Science', fontsize=10)
        ax_curve.tick_params(axis='y', labelsize=8)
        
        # Grid
        ax_curve.grid(True, axis='y', linestyle=':', alpha=0.5)
        ax_curve.set_axisbelow(True)
        
        # Spines
        ax_curve.spines['top'].set_visible(False)
        ax_curve.spines['right'].set_visible(False)
        
        # ===== Bottom: Event Annotations =====
        ax_events = fig.add_subplot(gs[2])
        ax_events.set_facecolor('#F5E6D3')
        
        if events:
            self._draw_event_annotations(ax_events, time_labels, events)
        else:
            ax_events.axis('off')
        
        # Title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        else:
            fig.suptitle(f'Topic {topic_idx} Evolution', fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return self._save_or_show(fig, filename)
    
    def plot_multi_topic_evolution(
        self,
        topic_indices: List[int],
        time_bins: List[int] = None,
        bin_size: int = 10,
        smooth_sigma: float = 1.0,
        events: List[Dict] = None,
        n_words: int = 5,
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Create DTM-style evolution plot for multiple topics.
        
        Args:
            topic_indices: List of topic indices to visualize
            time_bins: Time bin edges
            bin_size: Bin size
            smooth_sigma: Smoothing parameter
            events: Event annotations
            n_words: Number of top words
            filename: Output filename
            title: Plot title
            
        Returns:
            Figure
        """
        n_topics = len(topic_indices)
        
        # Aggregate data
        data = self.aggregate_by_time(time_bins, bin_size)
        time_labels = data['time_labels']
        
        # Colors for different topics
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
        
        # Create figure
        fig = plt.figure(figsize=(14, 4 + 3 * n_topics), facecolor='#F5E6D3')
        
        # Layout: word table + curve for each topic, then events at bottom
        n_rows = n_topics * 2 + 1  # 2 rows per topic + 1 for events
        height_ratios = []
        for _ in range(n_topics):
            height_ratios.extend([1, 1.5])  # words, curve
        height_ratios.append(0.5)  # events
        
        gs = GridSpec(n_rows, 1, height_ratios=height_ratios, hspace=0.2)
        
        x = np.arange(len(time_labels))
        
        for i, topic_idx in enumerate(topic_indices):
            proportions = data['topic_proportions'][:, topic_idx]
            word_evolution = data['word_evolution'][topic_idx]
            smoothed = gaussian_filter1d(proportions, sigma=smooth_sigma)
            color = colors[i % len(colors)]
            
            # Word evolution table
            ax_words = fig.add_subplot(gs[i * 2])
            ax_words.set_facecolor('#F5E6D3')
            self._draw_word_evolution_table(ax_words, time_labels, word_evolution, n_words)
            
            # Topic proportion curve
            ax_curve = fig.add_subplot(gs[i * 2 + 1])
            ax_curve.set_facecolor('#F5E6D3')
            
            ax_curve.plot(x, smoothed, color=color, linewidth=2.5)
            ax_curve.fill_between(x, 0, smoothed, alpha=0.2, color=color)
            ax_curve.scatter(x, proportions, color=color, s=20, zorder=5, edgecolor='white')
            
            ax_curve.set_xlim(-0.5, len(time_labels) - 0.5)
            ax_curve.set_ylim(0, max(smoothed.max() * 1.2, 0.05))
            ax_curve.set_xticks(x)
            ax_curve.set_xticklabels(time_labels if i == n_topics - 1 else [], fontsize=7)
            ax_curve.set_ylabel('Proportion', fontsize=8)
            ax_curve.tick_params(axis='y', labelsize=7)
            
            # Add topic label
            ax_curve.text(-0.02, 0.5, f'Topic {topic_idx}', transform=ax_curve.transAxes,
                         fontsize=9, fontweight='bold', va='center', ha='right',
                         color=color)
            
            ax_curve.spines['top'].set_visible(False)
            ax_curve.spines['right'].set_visible(False)
            ax_curve.grid(True, axis='y', linestyle=':', alpha=0.3)
        
        # Events at bottom
        ax_events = fig.add_subplot(gs[-1])
        ax_events.set_facecolor('#F5E6D3')
        if events:
            self._draw_event_annotations(ax_events, time_labels, events)
        else:
            ax_events.axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
        
        plt.tight_layout(rect=[0.02, 0, 1, 0.97])
        
        return self._save_or_show(fig, filename)
    
    def _draw_word_evolution_table(
        self,
        ax: plt.Axes,
        time_labels: List[str],
        word_evolution: List[List[str]],
        n_words: int = 5
    ):
        """Draw word evolution table at top of plot."""
        ax.axis('off')
        
        n_bins = len(time_labels)
        
        for i, (label, words) in enumerate(zip(time_labels, word_evolution)):
            x = i / n_bins + 0.5 / n_bins
            
            # Time label
            ax.text(x, 0.95, label, transform=ax.transAxes,
                   fontsize=8, fontweight='bold', ha='center', va='top')
            
            # Words (stacked vertically)
            if words:
                word_text = '\n'.join(words[:n_words])
                ax.text(x, 0.85, word_text, transform=ax.transAxes,
                       fontsize=7, ha='center', va='top', linespacing=1.2)
    
    def _draw_event_annotations(
        self,
        ax: plt.Axes,
        time_labels: List[str],
        events: List[Dict]
    ):
        """Draw event annotations at bottom of plot."""
        ax.set_xlim(-0.5, len(time_labels) - 0.5)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        for event in events:
            year = event.get('year')
            label = event.get('label', '')
            y_offset = event.get('y_offset', 0.5)
            
            # Find x position
            try:
                x_idx = time_labels.index(str(year))
            except ValueError:
                # Find closest
                years = [int(t) for t in time_labels]
                x_idx = np.argmin(np.abs(np.array(years) - year))
            
            # Draw annotation
            ax.annotate(
                label,
                xy=(x_idx, 0.9),
                xytext=(x_idx, y_offset),
                fontsize=7,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8)
            )
    
    def _save_or_show(self, fig, filename: Optional[str] = None):
        """Save figure or show it."""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor=fig.get_facecolor(), edgecolor='none')
            logger.info(f"Figure saved to {filepath}")
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return None


def plot_dynamic_topic_evolution(
    theta: np.ndarray,
    beta: np.ndarray,
    timestamps: np.ndarray,
    vocab: List[str],
    topic_indices: List[int],
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function to create dynamic topic evolution plot.
    
    Args:
        theta: Document-topic distribution
        beta: Topic-word distribution
        timestamps: Document timestamps
        vocab: Vocabulary
        topic_indices: Topics to visualize
        output_dir: Output directory
        filename: Output filename
        **kwargs: Additional arguments for plot_multi_topic_evolution
        
    Returns:
        Figure
    """
    visualizer = DynamicTopicEvolutionVisualizer(
        theta=theta,
        beta=beta,
        timestamps=timestamps,
        vocab=vocab,
        output_dir=output_dir
    )
    
    return visualizer.plot_multi_topic_evolution(
        topic_indices=topic_indices,
        filename=filename,
        **kwargs
    )


if __name__ == '__main__':
    # Example usage with simulated data
    np.random.seed(42)
    
    n_docs, n_topics, n_vocab = 1000, 10, 500
    
    # Simulate data
    theta = np.random.dirichlet(np.ones(n_topics) * 0.3, n_docs)
    beta = np.random.dirichlet(np.ones(n_vocab) * 0.1, n_topics)
    timestamps = np.random.randint(1880, 2010, n_docs)
    vocab = [f"word_{i}" for i in range(n_vocab)]
    
    # Create visualization
    output_dir = './test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = DynamicTopicEvolutionVisualizer(
        theta=theta,
        beta=beta,
        timestamps=timestamps,
        vocab=vocab,
        output_dir=output_dir
    )
    
    # Single topic evolution
    visualizer.plot_topic_evolution(
        topic_idx=0,
        bin_size=10,
        smooth_sigma=1.0,
        filename='topic_evolution_single.png'
    )
    
    # Multi-topic evolution
    visualizer.plot_multi_topic_evolution(
        topic_indices=[0, 1, 2],
        bin_size=10,
        smooth_sigma=1.0,
        filename='topic_evolution_multi.png'
    )
    
    print("Dynamic topic evolution plots saved!")

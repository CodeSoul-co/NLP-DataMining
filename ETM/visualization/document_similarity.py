"""
Document Similarity Retrieval Visualization

Creates CTM-style document similarity visualization showing:
- Query document topic distribution
- Top-k most similar documents by Hellinger distance
- 2x2 subplot layout with topic proportion bar charts

Reference: "A Correlated Topic Model of Science" paper Figure 3
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


class DocumentSimilarityVisualizer:
    """
    Visualize document similarity retrieval using topic distributions.
    
    Uses Hellinger distance to find similar documents and displays
    their topic proportions in a 2x2 grid layout.
    """
    
    def __init__(
        self,
        theta: np.ndarray,
        doc_titles: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150
    ):
        """
        Initialize document similarity visualizer.
        
        Args:
            theta: Document-topic distribution (N x K)
            doc_titles: Optional list of document titles
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.theta = theta
        self.num_docs, self.num_topics = theta.shape
        self.doc_titles = doc_titles if doc_titles else [f"Document {i}" for i in range(self.num_docs)]
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def hellinger_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Hellinger distance between two probability distributions.
        
        H(P, Q) = (1/sqrt(2)) * sqrt(sum((sqrt(p_i) - sqrt(q_i))^2))
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            Hellinger distance (0 to 1)
        """
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    
    def find_similar_documents(
        self,
        query_idx: int,
        top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Find top-k most similar documents to query document.
        
        Args:
            query_idx: Index of query document
            top_k: Number of similar documents to return
            
        Returns:
            List of (doc_idx, distance) tuples sorted by distance
        """
        query_dist = self.theta[query_idx]
        distances = []
        
        for i in range(self.num_docs):
            if i != query_idx:
                dist = self.hellinger_distance(query_dist, self.theta[i])
                distances.append((i, dist))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        return distances[:top_k]
    
    def plot_document_similarity(
        self,
        query_idx: int,
        top_k: int = 3,
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Create CTM-style document similarity visualization.
        
        Shows query document and top-k similar documents in 2x2 grid,
        each with topic proportion bar chart.
        
        Args:
            query_idx: Index of query document
            top_k: Number of similar documents (default 3 for 2x2 grid)
            filename: Output filename
            title: Overall figure title
            
        Returns:
            Figure
        """
        # Find similar documents
        similar_docs = self.find_similar_documents(query_idx, top_k)
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten()
        
        # Documents to plot: query + top 3 similar
        docs_to_plot = [(query_idx, 0.0)] + similar_docs[:3]
        labels = ['Query article', 'Closest match', '2nd closest match', '3rd closest match']
        
        for idx, (ax, (doc_idx, dist), label) in enumerate(zip(axes_flat, docs_to_plot, labels)):
            self._plot_topic_distribution(
                ax=ax,
                doc_idx=doc_idx,
                label=label,
                is_query=(idx == 0)
            )
        
        # Add overall title
        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return self._save_or_show(fig, filename)
    
    def _plot_topic_distribution(
        self,
        ax: plt.Axes,
        doc_idx: int,
        label: str,
        is_query: bool = False
    ):
        """
        Plot topic distribution for a single document.
        
        Args:
            ax: Matplotlib axes
            doc_idx: Document index
            label: Label for this subplot (e.g., "Query article")
            is_query: Whether this is the query document
        """
        # Get topic distribution
        topic_dist = self.theta[doc_idx]
        
        # Set white background
        ax.set_facecolor('white')
        
        # Create bar chart - thin black bars like reference image
        x = np.arange(self.num_topics)
        bars = ax.bar(x, topic_dist, color='black', width=0.4, edgecolor='black')
        
        # Set title (document title)
        doc_title = self.doc_titles[doc_idx]
        if len(doc_title) > 45:
            doc_title = doc_title[:42] + '...'
        ax.set_title(doc_title, fontsize=9, fontweight='bold', pad=8)
        
        # Add label (Query article, Closest match, etc.) in italics
        ax.text(0.5, 0.85, label, transform=ax.transAxes, 
               fontsize=9, fontstyle='italic', ha='center', va='top')
        
        # Set axis labels
        ax.set_xlabel('Topic index', fontsize=8)
        ax.set_ylabel('', fontsize=8)  # No y-label like reference
        
        # Set axis limits
        ax.set_xlim(-0.5, self.num_topics - 0.5)
        ax.set_ylim(0, max(0.7, topic_dist.max() * 1.1))
        
        # Set y-axis ticks
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ax.tick_params(axis='both', labelsize=7)
        
        # Clean style - only left and bottom spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Add box around subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('black')
    
    def plot_multiple_queries(
        self,
        query_indices: List[int],
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Create multiple document similarity visualizations.
        
        Args:
            query_indices: List of query document indices
            filename: Output filename
            
        Returns:
            Figure
        """
        n_queries = len(query_indices)
        fig, axes = plt.subplots(n_queries, 4, figsize=(14, 3 * n_queries), 
                                 facecolor='white')
        
        if n_queries == 1:
            axes = axes.reshape(1, -1)
        
        for row, query_idx in enumerate(query_indices):
            similar_docs = self.find_similar_documents(query_idx, top_k=3)
            docs_to_plot = [(query_idx, 0.0)] + similar_docs[:3]
            labels = ['Query article', 'Closest match', '2nd closest match', '3rd closest match']
            
            for col, ((doc_idx, dist), label) in enumerate(zip(docs_to_plot, labels)):
                self._plot_topic_distribution(
                    ax=axes[row, col],
                    doc_idx=doc_idx,
                    label=label,
                    is_query=(col == 0)
                )
        
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


def find_similar_documents(
    theta: np.ndarray,
    query_idx: int,
    top_k: int = 3,
    doc_titles: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Convenience function to find and visualize similar documents.
    
    Args:
        theta: Document-topic distribution
        query_idx: Query document index
        top_k: Number of similar documents
        doc_titles: Optional document titles
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Figure
    """
    visualizer = DocumentSimilarityVisualizer(
        theta=theta,
        doc_titles=doc_titles,
        output_dir=output_dir
    )
    
    return visualizer.plot_document_similarity(
        query_idx=query_idx,
        top_k=top_k,
        filename=filename
    )


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Simulate data
    n_docs, n_topics = 100, 20
    theta = np.random.dirichlet(np.ones(n_topics) * 0.3, n_docs)
    
    # Simulate document titles
    titles = [
        "Earth's Solid Iron Core May Skew Its Magnetic Field",
        "Do Anticracks Trigger Deep Earthquakes?",
        "Earth's Core Spins at Its Own Rate",
        "Superconductivity in a Grain of Salt",
        "Quantum Computing Advances",
    ] + [f"Document {i}" for i in range(5, n_docs)]
    
    # Create visualization
    output_dir = './test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = DocumentSimilarityVisualizer(
        theta=theta,
        doc_titles=titles,
        output_dir=output_dir
    )
    
    visualizer.plot_document_similarity(
        query_idx=0,
        top_k=3,
        filename='document_similarity.png'
    )
    
    print("Document similarity visualization saved!")

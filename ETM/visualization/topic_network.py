"""
Topic Network Graph Visualization

Creates force-directed network graphs showing topic relationships,
similar to the topic graph from academic papers.

Features:
- Nodes represent topics with top keywords as labels
- Node size proportional to topic popularity
- Edge thickness proportional to topic correlation
- Force-directed layout for natural clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


class TopicNetworkVisualizer:
    """
    Visualize topic relationships as a force-directed network graph.
    
    Each node represents a topic, labeled with its top keywords.
    Edges connect correlated topics, with thickness indicating correlation strength.
    """
    
    def __init__(
        self,
        theta: np.ndarray,
        topic_words: List[Tuple[int, List[Tuple[str, float]]]],
        topic_embeddings: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
        dpi: int = 150
    ):
        """
        Initialize topic network visualizer.
        
        Args:
            theta: Document-topic distribution (N x K)
            topic_words: List of (topic_idx, [(word, prob), ...])
            topic_embeddings: Optional topic embeddings for similarity calculation
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.theta = theta
        self.topic_words = topic_words
        self.topic_embeddings = topic_embeddings
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        self.num_docs, self.num_topics = theta.shape
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Compute topic popularity (average topic proportion)
        self.topic_popularity = theta.mean(axis=0)
        
        # Compute topic correlation matrix
        self.topic_correlation = self._compute_topic_correlation()
    
    def _compute_topic_correlation(self) -> np.ndarray:
        """
        Compute pairwise topic correlation from document-topic distribution.
        
        Returns:
            Correlation matrix (K x K)
        """
        # Use Pearson correlation between topic distributions
        correlation = np.corrcoef(self.theta.T)
        
        # If embeddings available, also consider embedding similarity
        if self.topic_embeddings is not None:
            # Cosine similarity
            norms = np.linalg.norm(self.topic_embeddings, axis=1, keepdims=True)
            normalized = self.topic_embeddings / (norms + 1e-8)
            embedding_sim = normalized @ normalized.T
            
            # Combine correlation and embedding similarity
            correlation = 0.5 * correlation + 0.5 * embedding_sim
        
        return correlation
    
    def _get_topic_label(self, topic_idx: int, n_words: int = 5) -> str:
        """Get topic label from top words."""
        for idx, words in self.topic_words:
            if idx == topic_idx:
                top_words = [w for w, _ in words[:n_words]]
                return '\n'.join(top_words)
        return f"Topic {topic_idx}"
    
    def _get_topic_short_label(self, topic_idx: int, n_words: int = 3) -> str:
        """Get short topic label for display."""
        for idx, words in self.topic_words:
            if idx == topic_idx:
                top_words = [w for w, _ in words[:n_words]]
                return '\n'.join(top_words)
        return f"T{topic_idx}"
    
    def plot_topic_network(
        self,
        correlation_threshold: float = 0.1,
        n_words: int = 5,
        min_node_size: float = 800,
        max_node_size: float = 4000,
        min_edge_width: float = 0.5,
        max_edge_width: float = 5.0,
        min_font_size: int = 8,
        max_font_size: int = 16,
        layout: str = 'spring',
        filename: Optional[str] = None,
        title: str = None,
        show_edge_labels: bool = False
    ) -> plt.Figure:
        """
        Create a force-directed topic network graph.
        
        Args:
            correlation_threshold: Minimum correlation to show edge
            n_words: Number of top words to show per topic
            min_node_size: Minimum node size
            max_node_size: Maximum node size
            min_edge_width: Minimum edge width
            max_edge_width: Maximum edge width
            min_font_size: Minimum font size for labels
            max_font_size: Maximum font size for labels
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular')
            filename: Output filename
            title: Custom title
            show_edge_labels: Whether to show correlation values on edges
            
        Returns:
            Figure
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for i in range(self.num_topics):
            G.add_node(i, 
                      popularity=self.topic_popularity[i],
                      label=self._get_topic_label(i, n_words))
        
        # Add edges based on correlation threshold
        for i in range(self.num_topics):
            for j in range(i + 1, self.num_topics):
                corr = self.topic_correlation[i, j]
                if corr > correlation_threshold:
                    G.add_edge(i, j, weight=corr)
        
        # Compute layout
        if layout == 'spring':
            # Use correlation as edge weights for spring layout
            # Higher correlation = stronger attraction
            pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42,
                                  weight='weight')
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Normalize popularity for node sizes
        pop_min, pop_max = self.topic_popularity.min(), self.topic_popularity.max()
        if pop_max > pop_min:
            norm_pop = (self.topic_popularity - pop_min) / (pop_max - pop_min)
        else:
            norm_pop = np.ones(self.num_topics) * 0.5
        
        node_sizes = min_node_size + norm_pop * (max_node_size - min_node_size)
        font_sizes = min_font_size + norm_pop * (max_font_size - min_font_size)
        
        # Use a rich combination of colormaps for maximum color diversity
        import colorsys
        node_colors = []
        # Create highly distinct colors using golden ratio for hue distribution
        golden_ratio = 0.618033988749895
        for i in range(self.num_topics):
            # Use golden ratio to spread hues more evenly
            hue = (i * golden_ratio) % 1.0
            # Vary saturation between 0.7-1.0 and value between 0.85-1.0
            sat = 0.75 + 0.25 * ((i * 3) % 5) / 4
            val = 0.85 + 0.15 * ((i * 7) % 3) / 2
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            node_colors.append((r, g, b, 0.9))
        
        # Draw edges with gradient based on correlation strength
        edges = G.edges(data=True)
        if edges:
            for (u, v, data) in edges:
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                weight = data.get('weight', 0.5)
                # Edge width based on correlation
                edge_width = min_edge_width + weight * (max_edge_width - min_edge_width)
                ax.plot(x, y, color='gray', linewidth=edge_width, alpha=0.5, zorder=1)
        
        # Draw nodes as colored circles with labels
        # Sort by size (largest first) so smaller circles are drawn on top
        sorted_indices = np.argsort(-node_sizes)
        
        for idx, i in enumerate(sorted_indices):
            x, y = pos[i]
            node_size = node_sizes[i]
            font_size = font_sizes[i]
            color = node_colors[i]
            
            # Get short label (topic number + top 2 words)
            label = self._get_topic_short_label(i, 2)
            
            # Draw filled circle with color - use zorder based on size (smaller = higher)
            radius = np.sqrt(node_size) / 100  # Convert size to radius
            z_order = 2 + (self.num_topics - idx)  # Smaller circles get higher zorder
            circle = plt.Circle((x, y), radius, facecolor=color, 
                               edgecolor='white', linewidth=3, alpha=0.75, zorder=z_order)
            ax.add_patch(circle)
            
            # Add topic number in center with black outline for visibility
            ax.annotate(f'T{i}', (x, y), fontsize=11, 
                       ha='center', va='center', zorder=z_order + 100,
                       fontweight='bold', color='white',
                       bbox=dict(boxstyle='circle,pad=0.1', facecolor='none', edgecolor='black', linewidth=1.5))
            
            # Add top words below the circle
            ax.annotate(label, (x, y - radius - 0.05), fontsize=8, 
                       ha='center', va='top', zorder=z_order + 100,
                       fontweight='normal', linespacing=1.0,
                       color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor='gray', linewidth=0.5, alpha=0.85))
        
        # Set axis properties
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            is_chinese = self._is_chinese_data()
            ax.set_title('主题关系网络图' if is_chinese else 'Topic Network Graph',
                        fontsize=14, fontweight='bold', pad=20)
        
        # Adjust limits to show all nodes
        x_vals = [pos[i][0] for i in range(self.num_topics)]
        y_vals = [pos[i][1] for i in range(self.num_topics)]
        margin = 0.3
        ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
        ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def plot_topic_correlation_network(
        self,
        correlation_threshold: float = 0.1,
        n_words: int = 5,
        filename: Optional[str] = None,
        title: str = None,
        layout_iterations: int = 300
    ) -> plt.Figure:
        """
        Create topic correlation network graph (CTM-style Figure 2).
        
        Replicates the classic topic graph from "A Correlated Topic Model of Science":
        - Ellipse nodes containing multi-line top phrases
        - Font size proportional to topic popularity
        - Thin black lines connecting correlated topics
        - Force-directed layout with overlap removal
        
        Args:
            correlation_threshold: Minimum correlation to show edge
            n_words: Number of top words per topic
            filename: Output filename
            title: Custom title
            layout_iterations: Number of layout iterations
            
        Returns:
            Figure
        """
        from matplotlib.patches import Ellipse
        
        # Create graph
        G = nx.Graph()
        for i in range(self.num_topics):
            G.add_node(i, popularity=self.topic_popularity[i])
        
        # Add edges based on correlation
        for i in range(self.num_topics):
            for j in range(i + 1, self.num_topics):
                corr = self.topic_correlation[i, j]
                if corr > correlation_threshold:
                    G.add_edge(i, j, weight=corr)
        
        # Force-directed layout
        pos = nx.spring_layout(
            G, 
            k=3.0 / np.sqrt(self.num_topics),
            iterations=layout_iterations,
            seed=42,
            weight='weight'
        )
        
        # Calculate node sizes based on labels (for overlap removal)
        node_sizes = {}
        for i in range(self.num_topics):
            label = self._get_topic_label(i, n_words)
            lines = label.split('\n')
            max_line_len = max(len(line) for line in lines)
            num_lines = len(lines)
            # Estimate ellipse dimensions
            node_sizes[i] = {
                'width': max_line_len * 0.018,
                'height': num_lines * 0.035
            }
        
        # Remove node overlaps
        pos = self._remove_ellipse_overlaps(pos, node_sizes, iterations=100)
        
        # Normalize popularity for font sizes
        pop_min, pop_max = self.topic_popularity.min(), self.topic_popularity.max()
        if pop_max > pop_min:
            norm_pop = (self.topic_popularity - pop_min) / (pop_max - pop_min)
        else:
            norm_pop = np.ones(self.num_topics) * 0.5
        
        # Font size range (like reference: larger topics have bigger fonts)
        min_font_size, max_font_size = 6, 12
        font_sizes = min_font_size + norm_pop * (max_font_size - min_font_size)
        
        # Create figure - larger for better readability
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
        ax.set_facecolor('white')
        
        # Draw edges first - thin black lines
        for (u, v, data) in G.edges(data=True):
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, 'k-', linewidth=0.5, alpha=1.0, zorder=1)
        
        # Draw nodes as ellipses with labels inside
        for i in range(self.num_topics):
            x, y = pos[i]
            font_size = font_sizes[i]
            
            # Get label text
            label = self._get_topic_label(i, n_words)
            lines = label.split('\n')
            max_line_len = max(len(line) for line in lines)
            num_lines = len(lines)
            
            # Calculate ellipse dimensions based on text and font size
            # Scale with font size for proportional sizing
            char_width = font_size * 0.0055
            line_height = font_size * 0.009
            text_width = max_line_len * char_width
            text_height = num_lines * line_height
            
            # Ellipse with padding
            ellipse_width = text_width * 1.4
            ellipse_height = text_height * 1.5
            
            # Draw ellipse - white fill, thin black border (like reference)
            ellipse = Ellipse(
                (x, y), ellipse_width, ellipse_height,
                facecolor='white', edgecolor='black', 
                linewidth=0.8, zorder=2
            )
            ax.add_patch(ellipse)
            
            # Add label inside ellipse
            ax.text(
                x, y, label,
                fontsize=font_size,
                ha='center', va='center',
                zorder=3,
                fontweight='normal',
                linespacing=0.9,
                color='black',
                fontfamily='serif'
            )
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # No title by default (like reference image)
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        # Set limits with margin
        x_vals = [pos[i][0] for i in range(self.num_topics)]
        y_vals = [pos[i][1] for i in range(self.num_topics)]
        margin = 0.15
        ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
        ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def _remove_ellipse_overlaps(
        self, 
        pos: Dict[int, Tuple[float, float]], 
        node_sizes: Dict[int, Dict[str, float]],
        iterations: int = 100
    ) -> Dict[int, Tuple[float, float]]:
        """
        Iteratively remove ellipse overlaps by pushing apart overlapping nodes.
        
        Args:
            pos: Node positions
            node_sizes: Dict of {node_id: {'width': w, 'height': h}}
            iterations: Number of adjustment iterations
            
        Returns:
            Adjusted positions
        """
        pos = {k: list(v) for k, v in pos.items()}
        
        for _ in range(iterations):
            moved = False
            for i in range(self.num_topics):
                for j in range(i + 1, self.num_topics):
                    dx = pos[j][0] - pos[i][0]
                    dy = pos[j][1] - pos[i][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    # Minimum distance based on ellipse sizes
                    min_dist_x = (node_sizes[i]['width'] + node_sizes[j]['width']) / 2 * 0.6
                    min_dist_y = (node_sizes[i]['height'] + node_sizes[j]['height']) / 2 * 0.6
                    min_dist = np.sqrt(min_dist_x**2 + min_dist_y**2) + 0.02
                    
                    if dist < min_dist and dist > 0:
                        overlap = (min_dist - dist) / 2
                        dx_norm, dy_norm = dx / dist, dy / dist
                        pos[i][0] -= dx_norm * overlap * 0.3
                        pos[i][1] -= dy_norm * overlap * 0.3
                        pos[j][0] += dx_norm * overlap * 0.3
                        pos[j][1] += dy_norm * overlap * 0.3
                        moved = True
            
            if not moved:
                break
        
        return {k: tuple(v) for k, v in pos.items()}
    
    def plot_topic_network_interactive(
        self,
        correlation_threshold: float = 0.1,
        n_words: int = 5,
        filename: Optional[str] = None
    ) -> str:
        """
        Create an interactive topic network using Plotly.
        
        Args:
            correlation_threshold: Minimum correlation to show edge
            n_words: Number of top words per topic
            filename: Output HTML filename
            
        Returns:
            Path to saved HTML file
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not available. Install with 'pip install plotly'")
            return None
        
        # Create graph
        G = nx.Graph()
        
        for i in range(self.num_topics):
            G.add_node(i, popularity=self.topic_popularity[i])
        
        for i in range(self.num_topics):
            for j in range(i + 1, self.num_topics):
                corr = self.topic_correlation[i, j]
                if corr > correlation_threshold:
                    G.add_edge(i, j, weight=corr)
        
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42, weight='weight')
        
        # Create edge traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = [pos[i][0] for i in range(self.num_topics)]
        node_y = [pos[i][1] for i in range(self.num_topics)]
        
        # Node sizes based on popularity
        pop_min, pop_max = self.topic_popularity.min(), self.topic_popularity.max()
        if pop_max > pop_min:
            norm_pop = (self.topic_popularity - pop_min) / (pop_max - pop_min)
        else:
            norm_pop = np.ones(self.num_topics) * 0.5
        
        node_sizes = 20 + norm_pop * 40
        
        # Node labels
        node_text = [self._get_topic_label(i, n_words).replace('\n', '<br>') 
                    for i in range(self.num_topics)]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='middle center',
            textfont=dict(size=10),
            marker=dict(
                size=node_sizes,
                color=self.topic_popularity,
                colorscale='Viridis',
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Topic Network',
                           showlegend=False,
                           hovermode='closest',
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_html(filepath)
            logger.info(f"Interactive network saved to {filepath}")
            return filepath
        
        return None
    
    def _is_chinese_data(self) -> bool:
        """Check if topic words contain Chinese characters."""
        if self.topic_words:
            for topic_id, words in self.topic_words[:1]:
                for word, _ in words[:3]:
                    if any('\u4e00' <= c <= '\u9fff' for c in word):
                        return True
        return False
    
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


def plot_topic_network(
    theta: np.ndarray,
    topic_words: List[Tuple[int, List[Tuple[str, float]]]],
    topic_embeddings: Optional[np.ndarray] = None,
    correlation_threshold: float = 0.1,
    n_words: int = 5,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function to create topic network graph.
    
    Args:
        theta: Document-topic distribution
        topic_words: Topic word lists
        topic_embeddings: Optional topic embeddings
        correlation_threshold: Minimum correlation for edges
        n_words: Number of words per topic label
        output_dir: Output directory
        filename: Output filename
        **kwargs: Additional arguments for plot_topic_network
        
    Returns:
        Figure
    """
    visualizer = TopicNetworkVisualizer(
        theta=theta,
        topic_words=topic_words,
        topic_embeddings=topic_embeddings,
        output_dir=output_dir
    )
    
    return visualizer.plot_topic_network(
        correlation_threshold=correlation_threshold,
        n_words=n_words,
        filename=filename,
        **kwargs
    )


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Simulate data
    n_docs, n_topics = 1000, 15
    theta = np.random.dirichlet(np.ones(n_topics) * 0.5, n_docs)
    
    # Simulate topic words
    words = ['gene', 'expression', 'protein', 'cell', 'dna', 'sequence',
             'brain', 'memory', 'neuron', 'cortex', 'stimulus',
             'star', 'galaxy', 'universe', 'planet', 'solar',
             'climate', 'ocean', 'temperature', 'carbon', 'atmosphere',
             'quantum', 'particle', 'electron', 'photon', 'energy']
    
    topic_words = []
    for i in range(n_topics):
        start = (i * 5) % len(words)
        topic_w = [(words[(start + j) % len(words)], 0.1 - j * 0.01) 
                   for j in range(10)]
        topic_words.append((i, topic_w))
    
    # Create visualization
    output_dir = './test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = TopicNetworkVisualizer(
        theta=theta,
        topic_words=topic_words,
        output_dir=output_dir
    )
    
    visualizer.plot_topic_network(
        correlation_threshold=0.05,
        n_words=5,
        filename='topic_network.png'
    )
    
    print("Topic network saved!")

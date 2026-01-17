"""
Dimension/Spatial Topic Analysis

Visualize topic distribution across different dimensions (e.g., regions, categories).
Supports Tab 4: Spatial/Dimension Distribution (Matrix Heatmap).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import logging
import os
import json

logger = logging.getLogger(__name__)


class DimensionAnalyzer:
    """
    Analyze and visualize topic distribution across dimensions.
    
    Dimensions can be:
    - Geographic regions (provinces, cities)
    - Categories (departments, types)
    - Any categorical variable in the dataset
    """
    
    def __init__(
        self,
        theta: np.ndarray,
        dimension_values: np.ndarray,
        topic_words: Optional[List[Tuple[int, List[Tuple[str, float]]]]] = None,
        dimension_name: str = "Dimension",
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 150
    ):
        """
        Initialize dimension analyzer.
        
        Args:
            theta: Document-topic distribution (N x K)
            dimension_values: Array of dimension values for each document (e.g., province names)
            topic_words: Optional list of (topic_idx, [(word, prob), ...])
            dimension_name: Name of the dimension (e.g., "Province", "Department")
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.theta = theta
        self.dimension_values = np.array(dimension_values)
        self.topic_words = topic_words
        self.dimension_name = dimension_name
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        self.num_docs, self.num_topics = theta.shape
        self.unique_dimensions = np.unique(dimension_values)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def _get_topic_label(self, topic_idx: int, max_words: int = 3) -> str:
        """Get topic label from top words"""
        if self.topic_words:
            for idx, words in self.topic_words:
                if idx == topic_idx:
                    top_words = [w for w, _ in words[:max_words]]
                    return f"T{topic_idx}: {', '.join(top_words)}"
        return f"Topic {topic_idx}"
    
    def _save_or_show(self, fig, filename: Optional[str] = None):
        """Save figure or show it"""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {filepath}")
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return None
    
    def compute_dimension_topic_distribution(
        self,
        aggregation: str = 'mean',
        normalize: bool = False
    ) -> pd.DataFrame:
        """
        Compute topic distribution for each dimension value.
        
        Args:
            aggregation: Aggregation method ('mean', 'sum', 'count')
            normalize: Whether to normalize rows to sum to 1
            
        Returns:
            DataFrame with dimensions as rows and topics as columns
        """
        # Create DataFrame
        df = pd.DataFrame({
            'dimension': self.dimension_values,
            **{f'topic_{k}': self.theta[:, k] for k in range(self.num_topics)}
        })
        
        topic_cols = [f'topic_{k}' for k in range(self.num_topics)]
        
        # Aggregate by dimension
        if aggregation == 'mean':
            dim_dist = df.groupby('dimension')[topic_cols].mean()
        elif aggregation == 'sum':
            dim_dist = df.groupby('dimension')[topic_cols].sum()
        elif aggregation == 'count':
            # Count documents where each topic is dominant
            for k in range(self.num_topics):
                df[f'dominant_{k}'] = (df[topic_cols].idxmax(axis=1) == f'topic_{k}').astype(int)
            dominant_cols = [f'dominant_{k}' for k in range(self.num_topics)]
            dim_dist = df.groupby('dimension')[dominant_cols].sum()
            dim_dist.columns = topic_cols
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Normalize if requested
        if normalize:
            dim_dist = dim_dist.div(dim_dist.sum(axis=1), axis=0)
        
        return dim_dist
    
    def plot_dimension_heatmap(
        self,
        top_k_topics: int = 10,
        top_k_dimensions: int = 20,
        aggregation: str = 'mean',
        normalize: bool = False,
        cmap: str = 'YlOrRd',
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot dimension-topic heatmap (Tab 4).
        
        Args:
            top_k_topics: Number of top topics to show
            top_k_dimensions: Number of top dimensions to show
            aggregation: Aggregation method
            normalize: Whether to normalize
            cmap: Colormap
            filename: Output filename
            
        Returns:
            Figure
        """
        dim_dist = self.compute_dimension_topic_distribution(aggregation, normalize)
        
        # Select top topics by overall average
        topic_cols = [f'topic_{k}' for k in range(self.num_topics)]
        avg_props = dim_dist[topic_cols].mean()
        top_topics = avg_props.nlargest(top_k_topics).index.tolist()
        
        # Select top dimensions by document count
        dim_counts = pd.Series(self.dimension_values).value_counts()
        top_dims = dim_counts.nlargest(top_k_dimensions).index.tolist()
        
        # Filter data
        plot_data = dim_dist.loc[dim_dist.index.isin(top_dims), top_topics]
        
        # Create topic labels
        topic_labels = []
        for col in top_topics:
            topic_idx = int(col.split('_')[1])
            topic_labels.append(self._get_topic_label(topic_idx, max_words=2))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            plot_data.T,
            ax=ax,
            cmap=cmap,
            annot=True if len(top_dims) <= 10 and len(top_topics) <= 10 else False,
            fmt='.2f',
            cbar_kws={'label': 'Topic Proportion' if not normalize else 'Normalized Proportion'},
            xticklabels=plot_data.index,
            yticklabels=topic_labels
        )
        
        ax.set_title(f'Topic Distribution by {self.dimension_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel(self.dimension_name, fontsize=12)
        ax.set_ylabel('Topic', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def plot_dimension_comparison(
        self,
        dimensions_to_compare: List[str],
        top_k_topics: int = 10,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare topic distributions across specific dimensions.
        
        Args:
            dimensions_to_compare: List of dimension values to compare
            top_k_topics: Number of top topics to show
            filename: Output filename
            
        Returns:
            Figure
        """
        dim_dist = self.compute_dimension_topic_distribution('mean')
        
        # Filter to requested dimensions
        plot_data = dim_dist.loc[dim_dist.index.isin(dimensions_to_compare)]
        
        # Select top topics
        topic_cols = [f'topic_{k}' for k in range(self.num_topics)]
        avg_props = plot_data[topic_cols].mean()
        top_topics = avg_props.nlargest(top_k_topics).index.tolist()
        
        plot_data = plot_data[top_topics]
        
        # Create topic labels
        topic_labels = []
        for col in top_topics:
            topic_idx = int(col.split('_')[1])
            topic_labels.append(self._get_topic_label(topic_idx, max_words=2))
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(top_topics))
        width = 0.8 / len(dimensions_to_compare)
        
        for i, dim in enumerate(dimensions_to_compare):
            if dim in plot_data.index:
                offset = (i - len(dimensions_to_compare) / 2 + 0.5) * width
                ax.bar(x + offset, plot_data.loc[dim].values, width, label=dim, alpha=0.8)
        
        ax.set_title(f'Topic Comparison Across {self.dimension_name}s', fontsize=16, fontweight='bold')
        ax.set_xlabel('Topic', fontsize=12)
        ax.set_ylabel('Average Proportion', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(topic_labels, rotation=45, ha='right')
        ax.legend(title=self.dimension_name)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_topic_by_dimension(
        self,
        topic_idx: int,
        top_k_dimensions: int = 15,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot a single topic's distribution across dimensions.
        
        Args:
            topic_idx: Topic index to visualize
            top_k_dimensions: Number of top dimensions to show
            filename: Output filename
            
        Returns:
            Figure
        """
        dim_dist = self.compute_dimension_topic_distribution('mean')
        
        topic_col = f'topic_{topic_idx}'
        topic_data = dim_dist[topic_col].sort_values(ascending=False).head(top_k_dimensions)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.YlOrRd(topic_data.values / topic_data.values.max())
        ax.barh(range(len(topic_data)), topic_data.values, color=colors)
        ax.set_yticks(range(len(topic_data)))
        ax.set_yticklabels(topic_data.index)
        ax.invert_yaxis()
        
        topic_label = self._get_topic_label(topic_idx)
        ax.set_title(f'{topic_label} Distribution by {self.dimension_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Average Topic Proportion', fontsize=12)
        ax.set_ylabel(self.dimension_name, fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def get_visualization_data_for_frontend(
        self,
        top_k_topics: int = 10,
        top_k_dimensions: int = 20
    ) -> Dict:
        """
        Get dimension analysis data in a format suitable for frontend.
        
        Returns:
            Dictionary with dimension analysis data for API response
        """
        dim_dist = self.compute_dimension_topic_distribution('mean')
        
        # Select top topics
        topic_cols = [f'topic_{k}' for k in range(self.num_topics)]
        avg_props = dim_dist[topic_cols].mean()
        top_topics = avg_props.nlargest(top_k_topics).index.tolist()
        
        # Select top dimensions
        dim_counts = pd.Series(self.dimension_values).value_counts()
        top_dims = dim_counts.nlargest(top_k_dimensions).index.tolist()
        
        # Filter data
        plot_data = dim_dist.loc[dim_dist.index.isin(top_dims), top_topics]
        
        # Build response
        heatmap_data = {
            'dimensions': plot_data.index.tolist(),
            'topics': [],
            'values': []
        }
        
        for col in top_topics:
            topic_idx = int(col.split('_')[1])
            heatmap_data['topics'].append({
                'id': topic_idx,
                'name': self._get_topic_label(topic_idx)
            })
            heatmap_data['values'].append(plot_data[col].values.tolist())
        
        # Dimension statistics
        dim_stats = {
            'dimension_name': self.dimension_name,
            'unique_count': len(self.unique_dimensions),
            'document_counts': dim_counts.head(top_k_dimensions).to_dict()
        }
        
        return {
            'heatmap_data': heatmap_data,
            'dimension_stats': dim_stats
        }
    
    def generate_dimension_report(
        self,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive dimension analysis report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Report dictionary
        """
        dim_dist = self.compute_dimension_topic_distribution('mean')
        
        # Find dominant topic for each dimension
        dominant_topics = {}
        for dim in dim_dist.index:
            topic_col = dim_dist.loc[dim].idxmax()
            topic_idx = int(topic_col.split('_')[1])
            dominant_topics[dim] = {
                'topic_id': topic_idx,
                'topic_name': self._get_topic_label(topic_idx),
                'proportion': float(dim_dist.loc[dim, topic_col])
            }
        
        # Find dimensions with highest concentration for each topic
        topic_hotspots = {}
        for k in range(self.num_topics):
            topic_col = f'topic_{k}'
            top_dim = dim_dist[topic_col].idxmax()
            topic_hotspots[k] = {
                'dimension': top_dim,
                'proportion': float(dim_dist.loc[top_dim, topic_col])
            }
        
        report = {
            'dimension_name': self.dimension_name,
            'num_documents': self.num_docs,
            'num_topics': self.num_topics,
            'num_dimensions': len(self.unique_dimensions),
            'dimensions': self.unique_dimensions.tolist(),
            'dominant_topics_by_dimension': dominant_topics,
            'topic_hotspots': topic_hotspots,
            'distribution_matrix': dim_dist.to_dict()
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Dimension report saved to {output_path}")
        
        return report


def analyze_dimension_topics(
    theta: np.ndarray,
    dimension_values: np.ndarray,
    topic_words: Optional[List] = None,
    dimension_name: str = "Dimension",
    output_dir: str = None
) -> Dict:
    """
    Convenience function to run full dimension analysis.
    
    Args:
        theta: Document-topic distribution
        dimension_values: Dimension values for each document
        topic_words: Optional topic words
        dimension_name: Name of the dimension
        output_dir: Output directory
        
    Returns:
        Analysis report
    """
    analyzer = DimensionAnalyzer(
        theta=theta,
        dimension_values=dimension_values,
        topic_words=topic_words,
        dimension_name=dimension_name,
        output_dir=output_dir
    )
    
    # Generate visualizations
    analyzer.plot_dimension_heatmap(filename="dimension_heatmap.png")
    
    # Generate report
    report = analyzer.generate_dimension_report(
        output_path=os.path.join(output_dir, "dimension_report.json") if output_dir else None
    )
    
    # Add frontend data
    report['frontend_data'] = analyzer.get_visualization_data_for_frontend()
    
    return report


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    
    num_docs = 500
    num_topics = 8
    
    # Generate random theta
    theta = np.random.dirichlet(np.ones(num_topics) * 0.5, size=num_docs)
    
    # Generate random dimension values (provinces)
    provinces = ['Beijing', 'Shanghai', 'Guangdong', 'Zhejiang', 'Jiangsu', 
                 'Shandong', 'Henan', 'Sichuan', 'Hubei', 'Hunan']
    dimension_values = np.random.choice(provinces, size=num_docs)
    
    # Generate topic words
    topic_words = [
        (k, [(f"word_{k}_{i}", 0.1 - i * 0.01) for i in range(10)])
        for k in range(num_topics)
    ]
    
    # Run analysis
    analyzer = DimensionAnalyzer(
        theta=theta,
        dimension_values=dimension_values,
        topic_words=topic_words,
        dimension_name="Province"
    )
    
    report = analyzer.generate_dimension_report()
    print(f"Generated dimension report with {len(report['dimensions'])} dimensions")

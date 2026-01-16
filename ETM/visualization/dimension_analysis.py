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
    
    def plot_covariate_effect(
        self,
        category_a: str,
        category_b: str,
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Plot covariate effect diagram (like 图6 协变量效应图).
        
        Shows topic preference between two categories. Each point is a topic,
        x-axis is the effect size (positive = prefer category_b, negative = prefer category_a).
        
        Args:
            category_a: First category name (left side, negative effect)
            category_b: Second category name (right side, positive effect)
            filename: Output filename
            title: Plot title
            
        Returns:
            Figure
        """
        # Compute topic distribution for each category
        dim_dist = self.compute_dimension_topic_distribution('mean')
        
        if category_a not in dim_dist.index or category_b not in dim_dist.index:
            available = dim_dist.index.tolist()
            logger.warning(f"Categories not found. Available: {available}")
            return None
        
        # Calculate effect size for each topic
        effects = []
        topic_labels = []
        
        for k in range(self.num_topics):
            topic_col = f'topic_{k}'
            if topic_col in dim_dist.columns:
                prop_a = dim_dist.loc[category_a, topic_col]
                prop_b = dim_dist.loc[category_b, topic_col]
                effect = prop_b - prop_a  # Positive = prefer B, Negative = prefer A
                effects.append(effect)
                topic_labels.append(f'Topic{k + 1}')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')
        ax.set_facecolor('white')
        
        # Sort by effect size for better visualization
        sorted_indices = np.argsort(effects)[::-1]
        
        # Y positions (spread topics vertically)
        y_positions = np.linspace(0.95, 0.05, len(effects))
        
        # Plot each topic as a point
        for i, idx in enumerate(sorted_indices):
            x = effects[idx]
            y = y_positions[i]
            ax.scatter(x, y, c='black', s=30, zorder=5)
            ax.annotate(topic_labels[idx], (x, y), 
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, va='center')
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        
        # X-axis label
        ax.set_xlabel(f'{category_a}……{category_b}', fontsize=12)
        
        # Set x limits symmetrically
        max_effect = max(abs(min(effects)), abs(max(effects)))
        ax.set_xlim(-max_effect * 1.2, max_effect * 1.2)
        
        # Title
        if title is None:
            title = f'图6  {category_a}/{category_b}主题偏好'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.02)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        return self._save_or_show(fig, filename)
    
    def compute_covariate_effects(
        self,
        category_a: str,
        category_b: str
    ) -> pd.DataFrame:
        """
        Compute covariate effects for all topics between two categories.
        
        Args:
            category_a: First category
            category_b: Second category
            
        Returns:
            DataFrame with topic effects
        """
        dim_dist = self.compute_dimension_topic_distribution('mean')
        
        results = []
        for k in range(self.num_topics):
            topic_col = f'topic_{k}'
            if topic_col in dim_dist.columns:
                prop_a = dim_dist.loc[category_a, topic_col] if category_a in dim_dist.index else 0
                prop_b = dim_dist.loc[category_b, topic_col] if category_b in dim_dist.index else 0
                effect = prop_b - prop_a
                
                results.append({
                    'topic': k,
                    'topic_label': f'Topic{k + 1}',
                    f'{category_a}_proportion': prop_a,
                    f'{category_b}_proportion': prop_b,
                    'effect': effect,
                    'preference': category_b if effect > 0 else category_a
                })
        
        return pd.DataFrame(results).sort_values('effect', ascending=False)
    
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
    
    def plot_multi_domain_temporal(
        self,
        timestamps: np.ndarray,
        freq: str = 'W',
        top_k_domains: int = 5,
        metric: str = 'count',
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Plot multi-domain topic distribution over time (like 图4).
        
        Creates a line chart comparing topic counts/strength across different
        domains (dimensions) over time periods.
        
        Args:
            timestamps: Array of timestamps for each document
            freq: Time frequency ('W'=week, 'M'=month, 'Q'=quarter, 'Y'=year)
            top_k_domains: Number of top domains to show
            metric: 'count' for topic count, 'strength' for average topic strength
            filename: Output filename
            title: Custom title
            
        Returns:
            Figure
        """
        # Create DataFrame with timestamps and dimensions
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'dimension': self.dimension_values,
            **{f'topic_{k}': self.theta[:, k] for k in range(self.num_topics)}
        })
        
        # Create time period column
        df['period'] = df['timestamp'].dt.to_period(freq)
        
        # Get top domains by document count
        domain_counts = df['dimension'].value_counts()
        top_domains = domain_counts.nlargest(top_k_domains).index.tolist()
        
        # Filter to top domains
        df_filtered = df[df['dimension'].isin(top_domains)]
        
        # Calculate metric per domain per period
        topic_cols = [f'topic_{k}' for k in range(self.num_topics)]
        
        if metric == 'count':
            # Count documents with dominant topic per domain per period
            df_filtered['dominant_topic'] = df_filtered[topic_cols].idxmax(axis=1)
            grouped = df_filtered.groupby(['period', 'dimension']).size().unstack(fill_value=0)
        else:
            # Average topic strength per domain per period
            grouped = df_filtered.groupby(['period', 'dimension'])[topic_cols].mean().sum(axis=1).unstack(fill_value=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('white')
        
        # Markers and colors for different domains
        markers = ['^', 'd', 's', 'o', 'v', 'p', 'h', '*', 'X', 'P']
        colors = ['#333333', '#666666', '#999999', '#CCCCCC', '#444444',
                  '#555555', '#777777', '#888888', '#AAAAAA', '#BBBBBB']
        
        # Plot each domain
        x_values = range(len(grouped.index))
        for i, domain in enumerate(top_domains):
            if domain in grouped.columns:
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                ax.plot(x_values, grouped[domain].values,
                       marker=marker, linewidth=1.5, markersize=8,
                       color=color, label=domain, markerfacecolor='white',
                       markeredgecolor=color, markeredgewidth=1.5)
        
        # Style the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#808080')
        ax.spines['bottom'].set_color('#808080')
        
        # X-axis labels
        x_labels = [str(p) for p in grouped.index]
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=45 if len(x_labels) > 10 else 0)
        
        # Check if Chinese data
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in str(top_domains[0])) if top_domains else False
        
        ax.set_xlabel('日期/周' if is_chinese else 'Date/Week', fontsize=11)
        y_label = '主题个数' if metric == 'count' else '主题强度'
        ax.set_ylabel(y_label if is_chinese else ('Topic Count' if metric == 'count' else 'Topic Strength'), fontsize=11)
        
        # Grid
        ax.yaxis.grid(True, alpha=0.3, linestyle='-', color='#CCCCCC')
        ax.xaxis.grid(False)
        
        # Legend at top
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
                 ncol=min(5, top_k_domains), fontsize=10, frameon=True,
                 framealpha=0.9)
        
        # Title
        if title is None:
            title = '各领域主题数分布' if is_chinese else 'Topic Distribution by Domain'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=40)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_domain_topic_comparison(
        self,
        timestamps: np.ndarray,
        topic_idx: int = None,
        freq: str = 'W',
        top_k_domains: int = 5,
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Plot specific topic strength comparison across domains over time.
        
        Args:
            timestamps: Array of timestamps for each document
            topic_idx: Specific topic to analyze (None for overall)
            freq: Time frequency
            top_k_domains: Number of top domains to show
            filename: Output filename
            title: Custom title
            
        Returns:
            Figure
        """
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'dimension': self.dimension_values,
            **{f'topic_{k}': self.theta[:, k] for k in range(self.num_topics)}
        })
        
        df['period'] = df['timestamp'].dt.to_period(freq)
        
        # Get top domains
        domain_counts = df['dimension'].value_counts()
        top_domains = domain_counts.nlargest(top_k_domains).index.tolist()
        df_filtered = df[df['dimension'].isin(top_domains)]
        
        # Calculate topic strength per domain per period
        if topic_idx is not None:
            topic_col = f'topic_{topic_idx}'
            grouped = df_filtered.groupby(['period', 'dimension'])[topic_col].mean().unstack(fill_value=0)
        else:
            # Average of all topics
            topic_cols = [f'topic_{k}' for k in range(self.num_topics)]
            grouped = df_filtered.groupby(['period', 'dimension'])[topic_cols].mean().mean(axis=1).unstack(fill_value=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('white')
        
        # Color palette
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12',
                  '#1ABC9C', '#E91E63', '#00BCD4', '#8BC34A', '#FF5722']
        markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
        
        x_values = range(len(grouped.index))
        for i, domain in enumerate(top_domains):
            if domain in grouped.columns:
                ax.plot(x_values, grouped[domain].values,
                       marker=markers[i % len(markers)], linewidth=2, markersize=8,
                       color=colors[i % len(colors)], label=domain, alpha=0.9)
        
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        x_labels = [str(p) for p in grouped.index]
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, fontsize=9)
        
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in str(top_domains[0])) if top_domains else False
        
        ax.set_xlabel('时间段' if is_chinese else 'Time Period', fontsize=11)
        ax.set_ylabel('主题强度' if is_chinese else 'Topic Strength', fontsize=11)
        
        ax.yaxis.grid(True, alpha=0.3, linestyle='-', color='#CCCCCC')
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=min(5, top_k_domains), fontsize=10, frameon=False)
        
        if title is None:
            if topic_idx is not None:
                topic_label = self._get_topic_label(topic_idx)
                title = f'{topic_label} - 各领域对比' if is_chinese else f'{topic_label} - Domain Comparison'
            else:
                title = '各领域主题强度对比' if is_chinese else 'Topic Strength by Domain'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
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

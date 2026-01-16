"""
Clustering Analysis Visualization

Visualize clustering iteration process and model convergence comparison:
- Clustering iteration visualization (Fig.2 style - scatter plot sequence)
- HDP vs DP convergence comparison (Fig.3 style - multi-subplot line charts)

Designed for SLAM object association and dynamic clustering scenarios
where the number of clusters is unknown and data arrives sequentially.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ClusteringAnalysisVisualizer:
    """
    Visualizer for clustering analysis and model comparison.
    
    Supports:
    - Clustering iteration process visualization
    - HDP vs DP convergence comparison
    - Ground truth comparison
    """
    
    def __init__(
        self,
        output_dir: str = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
    
    def _save_or_show(self, fig, filename: Optional[str] = None):
        """Save figure or show it."""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Figure saved to {filepath}")
            plt.close(fig)
            return filepath
        else:
            plt.show()
            return None
    
    def plot_clustering_iterations(
        self,
        iteration_data: List[Dict],
        trajectory: np.ndarray = None,
        ground_truth: Dict = None,
        n_cols: int = 4,
        filename: str = None,
        title: str = None
    ) -> plt.Figure:
        """
        Plot clustering iteration process (like Fig.2).
        
        Shows how clustering evolves over iterations, with each subplot
        representing a different iteration stage.
        
        Args:
            iteration_data: List of dicts, each containing:
                - 'points': np.ndarray of shape (N, 2) - point coordinates
                - 'labels': np.ndarray of cluster labels
                - 'iteration': iteration number
                - 'n_clusters': number of clusters found
            trajectory: Optional trajectory line coordinates (N, 2)
            ground_truth: Optional dict with ground truth data
            n_cols: Number of columns in subplot grid
            filename: Output filename
            title: Overall figure title
            
        Returns:
            Figure
        """
        n_plots = len(iteration_data)
        if ground_truth:
            n_plots += 1
        
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                                 facecolor='white')
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Color palette for clusters
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        plot_idx = 0
        
        # Plot ground truth first if provided
        if ground_truth:
            ax = axes[plot_idx]
            points = ground_truth['points']
            labels = ground_truth['labels']
            
            # Plot trajectory if provided
            if trajectory is not None:
                ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=1, alpha=0.5)
            
            # Plot points by cluster
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                color = colors[i % len(colors)]
                ax.scatter(points[mask, 0], points[mask, 1], 
                          c=[color], s=30, alpha=0.8, label=f'class {i+1}')
            
            ax.set_title('(a) Ground Truth', fontsize=11)
            ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
            ax.set_aspect('equal')
            ax.axis('off')
            plot_idx += 1
        
        # Plot each iteration
        for i, data in enumerate(iteration_data):
            if plot_idx >= len(axes):
                break
            
            ax = axes[plot_idx]
            points = data['points']
            labels = data['labels']
            iteration = data.get('iteration', i)
            n_clusters = data.get('n_clusters', len(np.unique(labels)))
            
            # Plot trajectory if provided
            if trajectory is not None:
                ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=1, alpha=0.5)
            
            # Plot points by cluster
            unique_labels = np.unique(labels)
            for j, label in enumerate(unique_labels):
                mask = labels == label
                color = colors[j % len(colors)]
                ax.scatter(points[mask, 0], points[mask, 1],
                          c=[color], s=30, alpha=0.8)
            
            # Add legend for first few plots
            if plot_idx < 2:
                for j in range(min(5, len(unique_labels))):
                    ax.scatter([], [], c=[colors[j]], label=f'class {j+1}')
                ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
            
            subplot_label = chr(ord('b') + i) if ground_truth else chr(ord('a') + i)
            ax.set_title(f'({subplot_label}) Iteration {iteration}, {n_clusters} objects', fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')
            plot_idx += 1
        
        # Hide unused axes
        for j in range(plot_idx, len(axes)):
            axes[j].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_convergence_comparison(
        self,
        hdp_data: List[Dict],
        dp_data: List[Dict] = None,
        ground_truth: int = 15,
        noise_configs: List[str] = None,
        n_cols: int = 3,
        filename: str = None,
        title: str = None
    ) -> plt.Figure:
        """
        Plot HDP vs DP convergence comparison (like Fig.3).
        
        Shows how different methods converge to the true number of clusters
        under different noise conditions.
        
        Args:
            hdp_data: List of dicts, each containing:
                - 'iterations': list of iteration numbers
                - 'n_objects': list of object counts per iteration
                - 'noise_config': optional noise configuration string
            dp_data: Similar structure for DP method (optional)
            ground_truth: True number of objects (for reference line)
            noise_configs: List of noise configuration labels
            n_cols: Number of columns in subplot grid
            filename: Output filename
            title: Overall figure title
            
        Returns:
            Figure
        """
        n_plots = len(hdp_data)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                                 facecolor='white')
        
        if n_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, hdp in enumerate(hdp_data):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Plot HDP line
            iterations = hdp['iterations']
            n_objects = hdp['n_objects']
            ax.plot(iterations, n_objects, 'b-', linewidth=2, label='HDP')
            
            # Plot DP line if provided
            if dp_data and i < len(dp_data):
                dp = dp_data[i]
                ax.plot(dp['iterations'], dp['n_objects'], 'r-', linewidth=2, label='DP')
            
            # Plot ground truth line
            ax.axhline(y=ground_truth, color='green', linestyle='--', 
                      linewidth=2, label='GT')
            
            # Style
            ax.set_xlabel('Iteration times', fontsize=10)
            ax.set_ylabel('amount of objects', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(False)
            
            # Set axis limits
            ax.set_xlim(0.5, max(iterations) + 0.5)
            y_max = max(n_objects) * 1.1
            if dp_data and i < len(dp_data):
                y_max = max(y_max, max(dp_data[i]['n_objects']) * 1.1)
            ax.set_ylim(ground_truth * 0.5, y_max)
            
            # Add noise config as title
            if noise_configs and i < len(noise_configs):
                ax.set_title(noise_configs[i], fontsize=10)
            elif 'noise_config' in hdp:
                ax.set_title(hdp['noise_config'], fontsize=10)
        
        # Hide unused axes
        for j in range(n_plots, len(axes)):
            axes[j].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_single_convergence(
        self,
        methods_data: Dict[str, Dict],
        ground_truth: int = None,
        filename: str = None,
        title: str = None
    ) -> plt.Figure:
        """
        Plot convergence comparison for a single noise condition.
        
        Args:
            methods_data: Dict mapping method names to data dicts:
                - 'iterations': list of iteration numbers
                - 'n_objects': list of object counts
            ground_truth: True number of objects
            filename: Output filename
            title: Plot title
            
        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        colors = {'HDP': 'blue', 'DP': 'red', 'ETM': 'purple', 'LDA': 'orange'}
        
        for method_name, data in methods_data.items():
            color = colors.get(method_name, 'gray')
            ax.plot(data['iterations'], data['n_objects'],
                   linewidth=2, label=method_name, color=color)
        
        if ground_truth:
            ax.axhline(y=ground_truth, color='green', linestyle='--',
                      linewidth=2, label='GT (Ground Truth)')
        
        ax.set_xlabel('Iteration times', fontsize=11)
        ax.set_ylabel('Amount of objects/clusters', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_topic_clustering_evolution(
        self,
        theta_history: List[np.ndarray],
        timestamps: List = None,
        method: str = 'tsne',
        filename: str = None,
        title: str = None
    ) -> plt.Figure:
        """
        Plot topic/document clustering evolution over training iterations.
        
        Useful for visualizing how ETM learns to cluster documents.
        
        Args:
            theta_history: List of theta matrices at different training stages
            timestamps: Optional labels for each stage
            method: Dimensionality reduction method ('tsne', 'pca')
            filename: Output filename
            title: Plot title
            
        Returns:
            Figure
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        n_stages = len(theta_history)
        n_cols = min(4, n_stages)
        n_rows = (n_stages + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                                 facecolor='white')
        axes = axes.flatten() if n_stages > 1 else [axes]
        
        for i, theta in enumerate(theta_history):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Reduce dimensions
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(theta)-1))
            else:
                reducer = PCA(n_components=2, random_state=42)
            
            coords = reducer.fit_transform(theta)
            
            # Get dominant topic for each document
            labels = np.argmax(theta, axis=1)
            
            # Plot
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                               cmap='tab10', s=20, alpha=0.7)
            
            n_clusters = len(np.unique(labels))
            stage_label = timestamps[i] if timestamps else f'Stage {i}'
            ax.set_title(f'{stage_label}\n{n_clusters} clusters', fontsize=10)
            ax.axis('off')
        
        # Hide unused axes
        for j in range(n_stages, len(axes)):
            axes[j].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def generate_convergence_report(
        self,
        methods_data: Dict[str, Dict],
        ground_truth: int = None
    ) -> pd.DataFrame:
        """
        Generate convergence statistics report.
        
        Args:
            methods_data: Dict mapping method names to convergence data
            ground_truth: True number of clusters
            
        Returns:
            DataFrame with convergence statistics
        """
        rows = []
        
        for method_name, data in methods_data.items():
            iterations = data['iterations']
            n_objects = data['n_objects']
            
            # Find convergence iteration (when it reaches ground truth)
            converged_iter = None
            if ground_truth:
                for i, n in enumerate(n_objects):
                    if n == ground_truth:
                        converged_iter = iterations[i]
                        break
            
            row = {
                'Method': method_name,
                'Final Clusters': n_objects[-1] if n_objects else None,
                'Min Clusters': min(n_objects) if n_objects else None,
                'Convergence Iteration': converged_iter,
                'Total Iterations': len(iterations),
                'Converged': converged_iter is not None
            }
            
            if ground_truth:
                row['Ground Truth'] = ground_truth
                row['Final Error'] = abs(n_objects[-1] - ground_truth) if n_objects else None
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def visualize_clustering_comparison(
    hdp_results: List[Dict],
    dp_results: List[Dict] = None,
    ground_truth: int = 15,
    output_dir: str = None
) -> Dict[str, str]:
    """
    Convenience function to generate clustering comparison visualizations.
    
    Args:
        hdp_results: HDP convergence data for different noise conditions
        dp_results: DP convergence data (optional)
        ground_truth: True number of clusters
        output_dir: Output directory
        
    Returns:
        Dictionary with paths to generated figures
    """
    visualizer = ClusteringAnalysisVisualizer(output_dir=output_dir)
    
    output_files = {}
    
    output_files['convergence'] = visualizer.plot_convergence_comparison(
        hdp_data=hdp_results,
        dp_data=dp_results,
        ground_truth=ground_truth,
        filename='convergence_comparison.png'
    )
    
    return output_files

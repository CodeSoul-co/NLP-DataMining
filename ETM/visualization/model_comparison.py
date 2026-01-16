"""
Model Training Comparison Visualization

Visualize and compare training metrics across different models:
- Perplexity curves (图7 style)
- Complexity curves (图8 style)
- Loss curves
- Other training metrics

Supports loading training logs from multiple models and aligning them
on the same x-axis (epochs/iterations) for comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import logging
import glob

logger = logging.getLogger(__name__)


class ModelComparisonVisualizer:
    """
    Visualizer for comparing training metrics across different models.
    
    Supports:
    - Loading training logs from JSON/CSV files
    - Aligning different models to the same x-axis
    - Plotting perplexity, complexity, loss curves
    - Publication-ready figures
    """
    
    def __init__(
        self,
        output_dir: str = None,
        figsize: Tuple[int, int] = (10, 7),
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
        
        # Store loaded model data
        self.models = {}  # model_name -> {'epochs': [], 'perplexity': [], 'complexity': [], ...}
    
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
    
    def load_training_log(
        self,
        log_path: str,
        model_name: str,
        epoch_key: str = 'epoch',
        perplexity_key: str = 'perplexity',
        complexity_key: str = 'complexity',
        loss_key: str = 'loss'
    ) -> Dict:
        """
        Load training log from a JSON or CSV file.
        
        Args:
            log_path: Path to the training log file
            model_name: Name to identify this model
            epoch_key: Key/column name for epoch numbers
            perplexity_key: Key/column name for perplexity values
            complexity_key: Key/column name for complexity values
            loss_key: Key/column name for loss values
            
        Returns:
            Dictionary with loaded data
        """
        if log_path.endswith('.json'):
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of epoch records
                epochs = [d.get(epoch_key, i) for i, d in enumerate(data)]
                perplexity = [d.get(perplexity_key) for d in data]
                complexity = [d.get(complexity_key) for d in data]
                loss = [d.get(loss_key) for d in data]
            elif isinstance(data, dict):
                # Dictionary with arrays
                epochs = data.get(epoch_key, data.get('epochs', list(range(len(data.get(perplexity_key, []))))))
                perplexity = data.get(perplexity_key, data.get('perplexities', []))
                complexity = data.get(complexity_key, data.get('complexities', []))
                loss = data.get(loss_key, data.get('losses', []))
            else:
                raise ValueError(f"Unsupported JSON structure in {log_path}")
                
        elif log_path.endswith('.csv'):
            df = pd.read_csv(log_path)
            epochs = df[epoch_key].tolist() if epoch_key in df.columns else list(range(len(df)))
            perplexity = df[perplexity_key].tolist() if perplexity_key in df.columns else []
            complexity = df[complexity_key].tolist() if complexity_key in df.columns else []
            loss = df[loss_key].tolist() if loss_key in df.columns else []
        else:
            raise ValueError(f"Unsupported file format: {log_path}")
        
        # Clean None values
        perplexity = [p for p in perplexity if p is not None]
        complexity = [c for c in complexity if c is not None]
        loss = [l for l in loss if l is not None]
        
        model_data = {
            'epochs': epochs[:len(perplexity)] if perplexity else epochs,
            'perplexity': perplexity,
            'complexity': complexity,
            'loss': loss,
            'log_path': log_path
        }
        
        self.models[model_name] = model_data
        logger.info(f"Loaded training log for {model_name}: {len(epochs)} epochs")
        
        return model_data
    
    def add_model_data(
        self,
        model_name: str,
        epochs: List[int],
        perplexity: List[float] = None,
        complexity: List[float] = None,
        loss: List[float] = None
    ):
        """
        Add model training data directly.
        
        Args:
            model_name: Name to identify this model
            epochs: List of epoch numbers
            perplexity: List of perplexity values
            complexity: List of complexity values
            loss: List of loss values
        """
        self.models[model_name] = {
            'epochs': epochs,
            'perplexity': perplexity or [],
            'complexity': complexity or [],
            'loss': loss or []
        }
        logger.info(f"Added model data for {model_name}: {len(epochs)} epochs")
    
    def plot_perplexity_comparison(
        self,
        model_names: List[str] = None,
        filename: str = None,
        title: str = None,
        max_epochs: int = None
    ) -> plt.Figure:
        """
        Plot perplexity comparison across models (like 图7).
        
        Args:
            model_names: List of model names to include (None for all)
            filename: Output filename
            title: Custom title
            max_epochs: Maximum epochs to show (for alignment)
            
        Returns:
            Figure
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Line styles for different models
        line_styles = ['--', '-.', '-', ':', (0, (3, 1, 1, 1))]
        markers = ['', '', '', '', '']  # No markers for clean look
        
        for i, name in enumerate(model_names):
            if name not in self.models:
                logger.warning(f"Model {name} not found")
                continue
            
            data = self.models[name]
            epochs = data['epochs']
            perplexity = data['perplexity']
            
            if not perplexity:
                logger.warning(f"No perplexity data for {name}")
                continue
            
            # Align to max_epochs if specified
            if max_epochs:
                epochs = [e for e in epochs if e <= max_epochs]
                perplexity = perplexity[:len(epochs)]
            
            style = line_styles[i % len(line_styles)]
            ax.plot(epochs, perplexity, linestyle=style, linewidth=1.5,
                   label=f'{name}模型' if self._is_chinese_name(name) else name,
                   color='black')
        
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xlabel('迭代次数' if self._has_chinese_models() else 'Iterations', fontsize=11)
        ax.set_ylabel('困惑度' if self._has_chinese_models() else 'Perplexity', fontsize=11)
        
        if title is None:
            title = f'{len(model_names)}种模型的内容困惑度对比' if self._has_chinese_models() else 'Model Perplexity Comparison'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=10, frameon=True)
        ax.grid(False)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_complexity_comparison(
        self,
        model_names: List[str] = None,
        filename: str = None,
        title: str = None,
        max_epochs: int = None
    ) -> plt.Figure:
        """
        Plot complexity comparison across models (like 图8).
        
        Args:
            model_names: List of model names to include (None for all)
            filename: Output filename
            title: Custom title
            max_epochs: Maximum epochs to show (for alignment)
            
        Returns:
            Figure
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Line styles for different models
        line_styles = ['--', '-.', '-', ':', (0, (3, 1, 1, 1))]
        
        for i, name in enumerate(model_names):
            if name not in self.models:
                logger.warning(f"Model {name} not found")
                continue
            
            data = self.models[name]
            epochs = data['epochs']
            complexity = data['complexity']
            
            if not complexity:
                logger.warning(f"No complexity data for {name}")
                continue
            
            # Align to max_epochs if specified
            if max_epochs:
                epochs = [e for e in epochs if e <= max_epochs]
                complexity = complexity[:len(epochs)]
            
            style = line_styles[i % len(line_styles)]
            ax.plot(epochs, complexity, linestyle=style, linewidth=1.5,
                   label=f'{name}模型' if self._is_chinese_name(name) else name,
                   color='black')
        
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xlabel('迭代次数' if self._has_chinese_models() else 'Iterations', fontsize=11)
        ax.set_ylabel('复杂度' if self._has_chinese_models() else 'Complexity', fontsize=11)
        
        if title is None:
            title = f'{len(model_names)}种模型的复杂度对比' if self._has_chinese_models() else 'Model Complexity Comparison'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='lower right', fontsize=10, frameon=True)
        ax.grid(False)
        
        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_loss_comparison(
        self,
        model_names: List[str] = None,
        filename: str = None,
        title: str = None,
        max_epochs: int = None
    ) -> plt.Figure:
        """
        Plot loss comparison across models.
        
        Args:
            model_names: List of model names to include (None for all)
            filename: Output filename
            title: Custom title
            max_epochs: Maximum epochs to show
            
        Returns:
            Figure
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']
        
        for i, name in enumerate(model_names):
            if name not in self.models:
                continue
            
            data = self.models[name]
            epochs = data['epochs']
            loss = data['loss']
            
            if not loss:
                continue
            
            if max_epochs:
                epochs = [e for e in epochs if e <= max_epochs]
                loss = loss[:len(epochs)]
            
            ax.plot(epochs, loss, linewidth=2,
                   label=name, color=colors[i % len(colors)])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(title or 'Model Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_combined_comparison(
        self,
        model_names: List[str] = None,
        filename: str = None,
        max_epochs: int = None
    ) -> plt.Figure:
        """
        Plot combined comparison with perplexity and complexity side by side.
        
        Args:
            model_names: List of model names to include
            filename: Output filename
            max_epochs: Maximum epochs to show
            
        Returns:
            Figure
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        
        line_styles = ['--', '-.', '-', ':', (0, (3, 1, 1, 1))]
        
        # Plot perplexity
        for i, name in enumerate(model_names):
            if name not in self.models:
                continue
            
            data = self.models[name]
            epochs = data['epochs']
            perplexity = data['perplexity']
            
            if perplexity:
                if max_epochs:
                    epochs_p = [e for e in epochs if e <= max_epochs]
                    perplexity = perplexity[:len(epochs_p)]
                else:
                    epochs_p = epochs[:len(perplexity)]
                
                ax1.plot(epochs_p, perplexity, linestyle=line_styles[i % len(line_styles)],
                        linewidth=1.5, label=f'{name}模型', color='black')
        
        ax1.set_xlabel('迭代次数', fontsize=11)
        ax1.set_ylabel('困惑度', fontsize=11)
        ax1.set_title(f'图7  {len(model_names)}种模型的内容困惑度对比', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot complexity
        for i, name in enumerate(model_names):
            if name not in self.models:
                continue
            
            data = self.models[name]
            epochs = data['epochs']
            complexity = data['complexity']
            
            if complexity:
                if max_epochs:
                    epochs_c = [e for e in epochs if e <= max_epochs]
                    complexity = complexity[:len(epochs_c)]
                else:
                    epochs_c = epochs[:len(complexity)]
                
                ax2.plot(epochs_c, complexity, linestyle=line_styles[i % len(line_styles)],
                        linewidth=1.5, label=f'{name}模型', color='black')
        
        ax2.set_xlabel('迭代次数', fontsize=11)
        ax2.set_ylabel('复杂度', fontsize=11)
        ax2.set_title(f'图8  {len(model_names)}种模型的复杂度对比', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def _is_chinese_name(self, name: str) -> bool:
        """Check if name contains Chinese characters."""
        return any('\u4e00' <= c <= '\u9fff' for c in name)
    
    def _has_chinese_models(self) -> bool:
        """Check if any model name contains Chinese."""
        return any(self._is_chinese_name(name) for name in self.models.keys())
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all loaded models.
        
        Returns:
            DataFrame with summary statistics
        """
        rows = []
        for name, data in self.models.items():
            row = {
                'Model': name,
                'Epochs': len(data['epochs']),
                'Final Perplexity': data['perplexity'][-1] if data['perplexity'] else None,
                'Min Perplexity': min(data['perplexity']) if data['perplexity'] else None,
                'Final Complexity': data['complexity'][-1] if data['complexity'] else None,
                'Final Loss': data['loss'][-1] if data['loss'] else None
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


def compare_model_training(
    log_paths: Dict[str, str],
    output_dir: str = None,
    max_epochs: int = None
) -> Dict[str, str]:
    """
    Convenience function to compare multiple model training logs.
    
    Args:
        log_paths: Dictionary mapping model names to log file paths
        output_dir: Directory to save visualizations
        max_epochs: Maximum epochs to show
        
    Returns:
        Dictionary with paths to generated figures
    """
    visualizer = ModelComparisonVisualizer(output_dir=output_dir)
    
    # Load all logs
    for name, path in log_paths.items():
        visualizer.load_training_log(path, name)
    
    # Generate plots
    output_files = {}
    
    output_files['perplexity'] = visualizer.plot_perplexity_comparison(
        filename='perplexity_comparison.png',
        max_epochs=max_epochs
    )
    
    output_files['complexity'] = visualizer.plot_complexity_comparison(
        filename='complexity_comparison.png',
        max_epochs=max_epochs
    )
    
    output_files['combined'] = visualizer.plot_combined_comparison(
        filename='combined_comparison.png',
        max_epochs=max_epochs
    )
    
    return output_files

"""
Topic Model Quantitative Evaluation Visualization

Creates CTM-style held-out log probability evaluation plots:
- Left: ETM vs LDA HOLD curves with error bars (10-fold CV)
- Right: Difference ΔHOLD = ETM - LDA with error bars

Reference: "A Correlated Topic Model of Science" paper Figure 4
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


class ModelEvaluationVisualizer:
    """
    Visualize topic model quantitative evaluation metrics.
    
    Compares ETM and LDA using held-out log probability across
    different numbers of topics (K), with 10-fold cross-validation.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5),
        dpi: int = 150
    ):
        """
        Initialize model evaluation visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def plot_holdout_comparison(
        self,
        k_values: List[int],
        etm_hold_means: List[float],
        etm_hold_stds: List[float],
        lda_hold_means: List[float],
        lda_hold_stds: List[float],
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Create CTM-style held-out log probability comparison plot.
        
        Left panel: ETM vs LDA HOLD curves with error bars
        Right panel: Difference ΔHOLD = ETM - LDA with error bars
        
        Args:
            k_values: List of topic numbers (K)
            etm_hold_means: Mean HOLD for ETM at each K (10-fold CV)
            etm_hold_stds: Std of HOLD for ETM at each K
            lda_hold_means: Mean HOLD for LDA at each K
            lda_hold_stds: Std of HOLD for LDA at each K
            filename: Output filename
            title: Overall figure title
            
        Returns:
            Figure
        """
        # Convert to numpy arrays
        k_values = np.array(k_values)
        etm_means = np.array(etm_hold_means)
        etm_stds = np.array(etm_hold_stds)
        lda_means = np.array(lda_hold_means)
        lda_stds = np.array(lda_hold_stds)
        
        # Calculate difference
        diff_means = etm_means - lda_means
        # Error propagation for difference: sqrt(std1^2 + std2^2)
        diff_stds = np.sqrt(etm_stds**2 + lda_stds**2)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, facecolor='white')
        
        # ===== Left Panel: HOLD comparison =====
        ax1.set_facecolor('white')
        
        # ETM curve (solid line with circles)
        ax1.errorbar(
            k_values, etm_means, yerr=etm_stds,
            fmt='o-', color='black', linewidth=1.2,
            markersize=5, capsize=3, capthick=1,
            label='ETM', markerfacecolor='white', markeredgecolor='black'
        )
        
        # LDA curve (dashed line with triangles)
        ax1.errorbar(
            k_values, lda_means, yerr=lda_stds,
            fmt='^--', color='black', linewidth=1.2,
            markersize=5, capsize=3, capthick=1,
            label='LDA', markerfacecolor='black', markeredgecolor='black'
        )
        
        ax1.set_xlabel('Number of topics', fontsize=10)
        ax1.set_ylabel('Held out log probability (HOLP)', fontsize=10)
        ax1.legend(loc='lower right', fontsize=9, frameon=True)
        ax1.tick_params(axis='both', labelsize=9)
        
        # Set x-axis ticks
        ax1.set_xticks(k_values[::2] if len(k_values) > 10 else k_values)
        
        # Grid
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        # Spines
        for spine in ax1.spines.values():
            spine.set_linewidth(0.8)
        
        # ===== Right Panel: Difference =====
        ax2.set_facecolor('white')
        
        # Difference curve with error bars
        ax2.errorbar(
            k_values, diff_means, yerr=diff_stds,
            fmt='o-', color='black', linewidth=1.2,
            markersize=5, capsize=3, capthick=1,
            markerfacecolor='white', markeredgecolor='black'
        )
        
        # Horizontal line at y=0
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        
        ax2.set_xlabel('Number of topics', fontsize=10)
        ax2.set_ylabel('HOLP difference (ETM−LDA)', fontsize=10)
        ax2.tick_params(axis='both', labelsize=9)
        
        # Set x-axis ticks
        ax2.set_xticks(k_values[::2] if len(k_values) > 10 else k_values)
        
        # Grid
        ax2.grid(True, linestyle=':', alpha=0.5)
        
        # Spines
        for spine in ax2.spines.values():
            spine.set_linewidth(0.8)
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def simulate_evaluation_data(
        self,
        k_values: List[int] = None,
        n_folds: int = 10,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Simulate realistic ETM vs LDA evaluation data.
        
        Simulates the expected pattern:
        - Small K: Both models similar
        - Medium K: ETM advantage grows, LDA overfitting
        - Large K: ETM stable, LDA declines
        
        Args:
            k_values: List of K values to evaluate
            n_folds: Number of CV folds
            seed: Random seed
            
        Returns:
            Dict with k_values, etm_means, etm_stds, lda_means, lda_stds
        """
        np.random.seed(seed)
        
        if k_values is None:
            k_values = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120]
        
        k_values = np.array(k_values)
        n_k = len(k_values)
        
        # Simulate ETM performance
        # ETM: improves with K, then plateaus
        etm_base = -115000 + 2000 * np.log(k_values + 1)
        etm_folds = np.zeros((n_k, n_folds))
        for i in range(n_k):
            # Variance decreases with more topics (more stable)
            noise_std = 300 / np.sqrt(k_values[i] / 10)
            etm_folds[i] = etm_base[i] + np.random.randn(n_folds) * noise_std
        
        etm_means = etm_folds.mean(axis=1)
        etm_stds = etm_folds.std(axis=1) / np.sqrt(n_folds)  # Standard error
        
        # Simulate LDA performance
        # LDA: improves initially, then overfits and declines
        lda_base = -115500 + 1500 * np.log(k_values + 1)
        # Add overfitting penalty for large K
        overfit_penalty = np.maximum(0, (k_values - 40) * 15)
        lda_base = lda_base - overfit_penalty
        
        lda_folds = np.zeros((n_k, n_folds))
        for i in range(n_k):
            # Higher variance for LDA, especially at large K
            noise_std = 400 + k_values[i] * 2
            lda_folds[i] = lda_base[i] + np.random.randn(n_folds) * noise_std
        
        lda_means = lda_folds.mean(axis=1)
        lda_stds = lda_folds.std(axis=1) / np.sqrt(n_folds)
        
        return {
            'k_values': k_values,
            'etm_means': etm_means,
            'etm_stds': etm_stds,
            'lda_means': lda_means,
            'lda_stds': lda_stds
        }
    
    def plot_predictive_perplexity(
        self,
        observed_pcts: List[int],
        etm_pp_means: List[float],
        etm_pp_stds: List[float],
        lda_pp_means: List[float],
        lda_pp_stds: List[float],
        filename: Optional[str] = None,
        title: str = None
    ) -> plt.Figure:
        """
        Create CTM-style predictive perplexity comparison plot.
        
        Evaluates document completion ability: given partial document,
        predict the held-out words.
        
        Left panel: ETM vs LDA PP curves with error bars
        Right panel: Difference ΔPP = ETM - LDA with error bars
        
        Args:
            observed_pcts: List of observed word percentages (e.g., [10, 20, ..., 90])
            etm_pp_means: Mean PP for ETM at each percentage (10-fold CV)
            etm_pp_stds: Std of PP for ETM
            lda_pp_means: Mean PP for LDA
            lda_pp_stds: Std of PP for LDA
            filename: Output filename
            title: Overall figure title
            
        Returns:
            Figure
        """
        # Convert to numpy arrays
        pcts = np.array(observed_pcts)
        etm_means = np.array(etm_pp_means)
        etm_stds = np.array(etm_pp_stds)
        lda_means = np.array(lda_pp_means)
        lda_stds = np.array(lda_pp_stds)
        
        # Calculate difference (ETM - LDA, negative is better for ETM)
        diff_means = etm_means - lda_means
        diff_stds = np.sqrt(etm_stds**2 + lda_stds**2)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, facecolor='white')
        
        # ===== Left Panel: PP comparison =====
        ax1.set_facecolor('white')
        
        # ETM curve (solid line with circles)
        ax1.errorbar(
            pcts, etm_means, yerr=etm_stds,
            fmt='o-', color='black', linewidth=1.2,
            markersize=5, capsize=3, capthick=1,
            label='ETM', markerfacecolor='white', markeredgecolor='black'
        )
        
        # LDA curve (dashed line with triangles)
        ax1.errorbar(
            pcts, lda_means, yerr=lda_stds,
            fmt='^--', color='black', linewidth=1.2,
            markersize=5, capsize=3, capthick=1,
            label='LDA', markerfacecolor='black', markeredgecolor='black'
        )
        
        ax1.set_xlabel('% of observed words', fontsize=10)
        ax1.set_ylabel('Predictive perplexity (PP)', fontsize=10)
        ax1.legend(loc='upper right', fontsize=9, frameon=True)
        ax1.tick_params(axis='both', labelsize=9)
        
        # Set x-axis ticks
        ax1.set_xticks(pcts)
        
        # Grid
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        # Spines
        for spine in ax1.spines.values():
            spine.set_linewidth(0.8)
        
        # ===== Right Panel: Difference =====
        ax2.set_facecolor('white')
        
        # Difference curve with error bars
        ax2.errorbar(
            pcts, diff_means, yerr=diff_stds,
            fmt='o-', color='black', linewidth=1.2,
            markersize=5, capsize=3, capthick=1,
            markerfacecolor='white', markeredgecolor='black'
        )
        
        # Horizontal line at y=0
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        
        ax2.set_xlabel('% of observed words', fontsize=10)
        ax2.set_ylabel('PP difference (ETM−LDA)', fontsize=10)
        ax2.tick_params(axis='both', labelsize=9)
        
        # Set x-axis ticks
        ax2.set_xticks(pcts)
        
        # Grid
        ax2.grid(True, linestyle=':', alpha=0.5)
        
        # Spines
        for spine in ax2.spines.values():
            spine.set_linewidth(0.8)
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        return self._save_or_show(fig, filename)
    
    def simulate_perplexity_data(
        self,
        observed_pcts: List[int] = None,
        n_folds: int = 10,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Simulate realistic ETM vs LDA predictive perplexity data.
        
        Simulates the expected pattern:
        - Low observed %: Both high PP, ETM slightly better
        - Medium observed %: ETM advantage grows
        - High observed %: Both converge, ETM still better
        
        Args:
            observed_pcts: List of observed word percentages
            n_folds: Number of CV folds
            seed: Random seed
            
        Returns:
            Dict with observed_pcts, etm_means, etm_stds, lda_means, lda_stds
        """
        np.random.seed(seed)
        
        if observed_pcts is None:
            observed_pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        pcts = np.array(observed_pcts)
        n_pcts = len(pcts)
        
        # Simulate ETM performance
        # PP decreases (improves) as more words are observed
        etm_base = 2500 - 8 * pcts - 0.05 * pcts**2
        etm_folds = np.zeros((n_pcts, n_folds))
        for i in range(n_pcts):
            noise_std = 50 + (100 - pcts[i]) * 0.5
            etm_folds[i] = etm_base[i] + np.random.randn(n_folds) * noise_std
        
        etm_means = etm_folds.mean(axis=1)
        etm_stds = etm_folds.std(axis=1) / np.sqrt(n_folds)
        
        # Simulate LDA performance
        # LDA: higher PP (worse), especially at low observed %
        lda_base = 2600 - 6 * pcts - 0.03 * pcts**2
        lda_folds = np.zeros((n_pcts, n_folds))
        for i in range(n_pcts):
            noise_std = 60 + (100 - pcts[i]) * 0.8
            lda_folds[i] = lda_base[i] + np.random.randn(n_folds) * noise_std
        
        lda_means = lda_folds.mean(axis=1)
        lda_stds = lda_folds.std(axis=1) / np.sqrt(n_folds)
        
        return {
            'observed_pcts': pcts,
            'etm_means': etm_means,
            'etm_stds': etm_stds,
            'lda_means': lda_means,
            'lda_stds': lda_stds
        }
    
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


def plot_holdout_comparison(
    k_values: List[int],
    etm_hold_means: List[float],
    etm_hold_stds: List[float],
    lda_hold_means: List[float],
    lda_hold_stds: List[float],
    output_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> plt.Figure:
    """
    Convenience function to create held-out log probability comparison plot.
    
    Args:
        k_values: List of topic numbers
        etm_hold_means: ETM HOLD means
        etm_hold_stds: ETM HOLD standard errors
        lda_hold_means: LDA HOLD means
        lda_hold_stds: LDA HOLD standard errors
        output_dir: Output directory
        filename: Output filename
        
    Returns:
        Figure
    """
    visualizer = ModelEvaluationVisualizer(output_dir=output_dir)
    
    return visualizer.plot_holdout_comparison(
        k_values=k_values,
        etm_hold_means=etm_hold_means,
        etm_hold_stds=etm_hold_stds,
        lda_hold_means=lda_hold_means,
        lda_hold_stds=lda_hold_stds,
        filename=filename
    )


if __name__ == '__main__':
    # Example usage with simulated data
    output_dir = './test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = ModelEvaluationVisualizer(output_dir=output_dir)
    
    # Simulate evaluation data
    data = visualizer.simulate_evaluation_data()
    
    # Create plot
    visualizer.plot_holdout_comparison(
        k_values=data['k_values'],
        etm_hold_means=data['etm_means'],
        etm_hold_stds=data['etm_stds'],
        lda_hold_means=data['lda_means'],
        lda_hold_stds=data['lda_stds'],
        filename='holdout_comparison.png'
    )
    
    print("Held-out log probability comparison plot saved!")

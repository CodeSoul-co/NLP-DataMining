"""
Topic K Evaluation Module for ETM

This module provides tools for evaluating different numbers of topics (K) 
and visualizing metrics like coherence and perplexity to help select the optimal K.

Similar to LDA paper conventions, this generates:
- Coherence score curves (NPMI, C_V, U_Mass)
- Perplexity curves
- Combined evaluation plots
- Elbow method visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
import pandas as pd
import logging
from scipy.special import softmax

# Configure logging
logger = logging.getLogger(__name__)


class TopicKEvaluator:
    """
    Evaluator for selecting optimal number of topics (K).
    
    Provides methods to:
    - Calculate topic coherence scores
    - Calculate perplexity
    - Visualize K evaluation curves
    - Recommend optimal K based on metrics
    """
    
    def __init__(
        self,
        output_dir: str = None,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 150,
        random_state: int = 42
    ):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Default figure size
            dpi: Default figure DPI
            random_state: Random state for reproducibility
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        self.random_state = random_state
        
        # Store evaluation results
        self.k_values = []
        self.coherence_scores = {}
        self.perplexity_scores = []
        self.diversity_scores = []
        
        # Set plot style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')
    
    def _save_or_show(self, fig, filename: Optional[str] = None):
        """Save figure to file or show it."""
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            logger.info(f"Figure saved to {filepath}")
            return filepath
        else:
            plt.show()
            return None
    
    def calculate_coherence_npmi(
        self,
        topic_word_dist: np.ndarray,
        doc_term_matrix: np.ndarray,
        vocab: List[str],
        top_n: int = 10,
        epsilon: float = 1e-12
    ) -> float:
        """
        Calculate NPMI (Normalized Pointwise Mutual Information) coherence.
        
        NPMI is one of the most reliable coherence metrics, ranging from -1 to 1.
        Higher values indicate more coherent topics.
        
        Args:
            topic_word_dist: Topic-word distribution matrix (K x V)
            doc_term_matrix: Document-term matrix (D x V)
            vocab: Vocabulary list
            top_n: Number of top words per topic to consider
            epsilon: Small value to avoid log(0)
            
        Returns:
            Average NPMI coherence score across all topics
        """
        num_topics = topic_word_dist.shape[0]
        num_docs = doc_term_matrix.shape[0]
        
        # Binarize document-term matrix (word presence)
        doc_term_binary = (doc_term_matrix > 0).astype(float)
        
        # Word document frequencies
        word_doc_freq = doc_term_binary.sum(axis=0) + epsilon
        
        topic_coherences = []
        
        for k in range(num_topics):
            # Get top words for this topic
            top_word_indices = np.argsort(-topic_word_dist[k])[:top_n]
            
            npmi_pairs = []
            for i in range(len(top_word_indices)):
                for j in range(i + 1, len(top_word_indices)):
                    w_i = top_word_indices[i]
                    w_j = top_word_indices[j]
                    
                    # P(w_i), P(w_j)
                    p_i = word_doc_freq[w_i] / num_docs
                    p_j = word_doc_freq[w_j] / num_docs
                    
                    # P(w_i, w_j) - co-occurrence
                    co_occur = ((doc_term_binary[:, w_i] > 0) & 
                               (doc_term_binary[:, w_j] > 0)).sum()
                    p_ij = (co_occur + epsilon) / num_docs
                    
                    # PMI = log(P(w_i, w_j) / (P(w_i) * P(w_j)))
                    pmi = np.log(p_ij / (p_i * p_j + epsilon) + epsilon)
                    
                    # NPMI = PMI / -log(P(w_i, w_j))
                    npmi = pmi / (-np.log(p_ij + epsilon) + epsilon)
                    npmi_pairs.append(npmi)
            
            if npmi_pairs:
                topic_coherences.append(np.mean(npmi_pairs))
        
        return np.mean(topic_coherences) if topic_coherences else 0.0
    
    def calculate_coherence_umass(
        self,
        topic_word_dist: np.ndarray,
        doc_term_matrix: np.ndarray,
        top_n: int = 10,
        epsilon: float = 1e-12
    ) -> float:
        """
        Calculate UMass coherence score.
        
        UMass coherence uses document co-occurrence and is typically negative.
        Less negative values indicate more coherent topics.
        
        Args:
            topic_word_dist: Topic-word distribution matrix (K x V)
            doc_term_matrix: Document-term matrix (D x V)
            top_n: Number of top words per topic
            epsilon: Small value to avoid log(0)
            
        Returns:
            Average UMass coherence score
        """
        num_topics = topic_word_dist.shape[0]
        
        # Binarize
        doc_term_binary = (doc_term_matrix > 0).astype(float)
        word_doc_freq = doc_term_binary.sum(axis=0) + epsilon
        
        topic_coherences = []
        
        for k in range(num_topics):
            top_word_indices = np.argsort(-topic_word_dist[k])[:top_n]
            
            umass_sum = 0
            count = 0
            
            for i in range(1, len(top_word_indices)):
                for j in range(i):
                    w_i = top_word_indices[i]
                    w_j = top_word_indices[j]
                    
                    # D(w_i, w_j) - co-occurrence count
                    co_occur = ((doc_term_binary[:, w_i] > 0) & 
                               (doc_term_binary[:, w_j] > 0)).sum()
                    
                    # D(w_j) - document frequency of w_j
                    d_j = word_doc_freq[w_j]
                    
                    # UMass = log((D(w_i, w_j) + epsilon) / D(w_j))
                    umass_sum += np.log((co_occur + epsilon) / d_j)
                    count += 1
            
            if count > 0:
                topic_coherences.append(umass_sum / count)
        
        return np.mean(topic_coherences) if topic_coherences else 0.0
    
    def calculate_perplexity(
        self,
        theta: np.ndarray,
        beta: np.ndarray,
        doc_term_matrix: np.ndarray,
        epsilon: float = 1e-12
    ) -> float:
        """
        Calculate perplexity of the topic model.
        
        Lower perplexity indicates better model fit.
        Perplexity = exp(-1/N * sum(log(p(w|d))))
        
        Args:
            theta: Document-topic distribution (D x K)
            beta: Topic-word distribution (K x V)
            doc_term_matrix: Document-term matrix (D x V)
            epsilon: Small value for numerical stability
            
        Returns:
            Perplexity score
        """
        # Reconstruct document-word probabilities
        # P(w|d) = sum_k theta[d,k] * beta[k,w]
        doc_word_prob = np.dot(theta, beta) + epsilon
        
        # Normalize
        doc_word_prob = doc_word_prob / doc_word_prob.sum(axis=1, keepdims=True)
        
        # Calculate log-likelihood
        # Only consider words that appear in documents
        log_likelihood = 0
        total_words = 0
        
        for d in range(doc_term_matrix.shape[0]):
            word_counts = doc_term_matrix[d]
            nonzero_indices = np.where(word_counts > 0)[0]
            
            for w in nonzero_indices:
                count = word_counts[w]
                log_likelihood += count * np.log(doc_word_prob[d, w] + epsilon)
                total_words += count
        
        # Perplexity = exp(-log_likelihood / total_words)
        if total_words > 0:
            perplexity = np.exp(-log_likelihood / total_words)
        else:
            perplexity = float('inf')
        
        return perplexity
    
    def calculate_topic_diversity(
        self,
        topic_word_dist: np.ndarray,
        top_n: int = 25
    ) -> float:
        """
        Calculate topic diversity (percentage of unique words in top words).
        
        Higher diversity indicates more distinct topics.
        
        Args:
            topic_word_dist: Topic-word distribution (K x V)
            top_n: Number of top words per topic
            
        Returns:
            Diversity score (0 to 1)
        """
        num_topics = topic_word_dist.shape[0]
        
        all_top_words = set()
        total_words = 0
        
        for k in range(num_topics):
            top_indices = np.argsort(-topic_word_dist[k])[:top_n]
            all_top_words.update(top_indices)
            total_words += top_n
        
        diversity = len(all_top_words) / total_words if total_words > 0 else 0
        return diversity
    
    def add_evaluation_result(
        self,
        k: int,
        coherence_npmi: float = None,
        coherence_umass: float = None,
        perplexity: float = None,
        diversity: float = None
    ):
        """
        Add evaluation result for a specific K value.
        
        Args:
            k: Number of topics
            coherence_npmi: NPMI coherence score
            coherence_umass: UMass coherence score
            perplexity: Perplexity score
            diversity: Topic diversity score
        """
        if k not in self.k_values:
            self.k_values.append(k)
            self.k_values.sort()
        
        idx = self.k_values.index(k)
        
        # Ensure lists are long enough
        while len(self.perplexity_scores) <= idx:
            self.perplexity_scores.append(None)
        while len(self.diversity_scores) <= idx:
            self.diversity_scores.append(None)
        
        if coherence_npmi is not None:
            if 'npmi' not in self.coherence_scores:
                self.coherence_scores['npmi'] = [None] * len(self.k_values)
            while len(self.coherence_scores['npmi']) <= idx:
                self.coherence_scores['npmi'].append(None)
            self.coherence_scores['npmi'][idx] = coherence_npmi
        
        if coherence_umass is not None:
            if 'umass' not in self.coherence_scores:
                self.coherence_scores['umass'] = [None] * len(self.k_values)
            while len(self.coherence_scores['umass']) <= idx:
                self.coherence_scores['umass'].append(None)
            self.coherence_scores['umass'][idx] = coherence_umass
        
        if perplexity is not None:
            self.perplexity_scores[idx] = perplexity
        
        if diversity is not None:
            self.diversity_scores[idx] = diversity
    
    def load_from_training_history(
        self,
        history_files: List[str],
        k_values: List[int] = None
    ):
        """
        Load evaluation metrics from training history JSON files.
        
        Args:
            history_files: List of paths to training_history.json files
            k_values: Corresponding K values (if not in history)
        """
        for i, filepath in enumerate(history_files):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # Try to extract K from history or use provided value
                k = history.get('num_topics', k_values[i] if k_values else None)
                if k is None:
                    logger.warning(f"Could not determine K for {filepath}")
                    continue
                
                # Extract metrics from final epoch
                final_metrics = history.get('final_metrics', {})
                
                perplexity = final_metrics.get('perplexity')
                coherence = final_metrics.get('coherence')
                diversity = final_metrics.get('topic_diversity')
                
                self.add_evaluation_result(
                    k=k,
                    coherence_npmi=coherence,
                    perplexity=perplexity,
                    diversity=diversity
                )
                
                logger.info(f"Loaded metrics for K={k} from {filepath}")
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
    
    def plot_coherence_curve(
        self,
        metric: str = 'npmi',
        filename: Optional[str] = None,
        title: str = None,
        show_optimal: bool = True
    ) -> Optional[str]:
        """
        Plot coherence score curve across different K values.
        
        Args:
            metric: Coherence metric to plot ('npmi' or 'umass')
            filename: Output filename
            title: Plot title
            show_optimal: Whether to mark the optimal K
            
        Returns:
            Path to saved figure or None
        """
        if metric not in self.coherence_scores:
            logger.warning(f"No {metric} coherence scores available")
            return None
        
        scores = self.coherence_scores[metric]
        valid_k = [k for k, s in zip(self.k_values, scores) if s is not None]
        valid_scores = [s for s in scores if s is not None]
        
        if not valid_scores:
            logger.warning("No valid coherence scores to plot")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot coherence curve
        ax.plot(valid_k, valid_scores, 'b-o', linewidth=2, markersize=8, 
                label=f'{metric.upper()} Coherence')
        ax.fill_between(valid_k, valid_scores, alpha=0.2)
        
        # Mark optimal K (highest coherence)
        if show_optimal:
            optimal_idx = np.argmax(valid_scores)
            optimal_k = valid_k[optimal_idx]
            optimal_score = valid_scores[optimal_idx]
            
            ax.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.7,
                      label=f'Optimal K={optimal_k}')
            ax.scatter([optimal_k], [optimal_score], color='r', s=150, 
                      zorder=5, marker='*')
            ax.annotate(f'K={optimal_k}\n{optimal_score:.4f}',
                       xy=(optimal_k, optimal_score),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Number of Topics (K)', fontsize=12)
        ax.set_ylabel(f'{metric.upper()} Coherence Score', fontsize=12)
        ax.set_title(title or f'Topic Coherence ({metric.upper()}) vs Number of Topics',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(valid_k)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_perplexity_curve(
        self,
        filename: Optional[str] = None,
        title: str = None,
        show_elbow: bool = True
    ) -> Optional[str]:
        """
        Plot perplexity curve across different K values.
        
        Lower perplexity indicates better fit, but may overfit with too many topics.
        
        Args:
            filename: Output filename
            title: Plot title
            show_elbow: Whether to detect and mark elbow point
            
        Returns:
            Path to saved figure or None
        """
        valid_k = [k for k, s in zip(self.k_values, self.perplexity_scores) 
                   if s is not None]
        valid_scores = [s for s in self.perplexity_scores if s is not None]
        
        if not valid_scores:
            logger.warning("No valid perplexity scores to plot")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot perplexity curve
        ax.plot(valid_k, valid_scores, 'g-o', linewidth=2, markersize=8,
                label='Perplexity')
        ax.fill_between(valid_k, valid_scores, alpha=0.2, color='green')
        
        # Detect elbow point using second derivative
        if show_elbow and len(valid_scores) >= 3:
            elbow_k = self._find_elbow(valid_k, valid_scores)
            if elbow_k:
                elbow_idx = valid_k.index(elbow_k)
                elbow_score = valid_scores[elbow_idx]
                
                ax.axvline(x=elbow_k, color='r', linestyle='--', alpha=0.7,
                          label=f'Elbow K={elbow_k}')
                ax.scatter([elbow_k], [elbow_score], color='r', s=150,
                          zorder=5, marker='*')
        
        ax.set_xlabel('Number of Topics (K)', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title(title or 'Perplexity vs Number of Topics', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(valid_k)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_combined_evaluation(
        self,
        filename: Optional[str] = None,
        title: str = None
    ) -> Optional[str]:
        """
        Plot combined evaluation with coherence, perplexity, and diversity.
        
        This is the main visualization for K selection, similar to LDA papers.
        
        Args:
            filename: Output filename
            title: Plot title
            
        Returns:
            Path to saved figure or None
        """
        # Determine which metrics are available
        has_coherence = any(len([s for s in scores if s is not None]) > 0 
                          for scores in self.coherence_scores.values())
        has_perplexity = any(s is not None for s in self.perplexity_scores)
        has_diversity = any(s is not None for s in self.diversity_scores)
        
        num_plots = sum([has_coherence, has_perplexity, has_diversity])
        if num_plots == 0:
            logger.warning("No metrics available to plot")
            return None
        
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        optimal_k_suggestions = []
        
        # Plot coherence
        if has_coherence:
            ax = axes[plot_idx]
            colors = {'npmi': 'blue', 'umass': 'purple', 'cv': 'cyan'}
            
            for metric, scores in self.coherence_scores.items():
                valid_k = [k for k, s in zip(self.k_values, scores) if s is not None]
                valid_scores = [s for s in scores if s is not None]
                
                if valid_scores:
                    ax.plot(valid_k, valid_scores, '-o', linewidth=2, markersize=6,
                           color=colors.get(metric, 'blue'),
                           label=f'{metric.upper()}')
                    
                    # Find optimal K for this metric
                    optimal_idx = np.argmax(valid_scores)
                    optimal_k_suggestions.append(valid_k[optimal_idx])
            
            ax.set_xlabel('Number of Topics (K)', fontsize=11)
            ax.set_ylabel('Coherence Score', fontsize=11)
            ax.set_title('Topic Coherence', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            if self.k_values:
                ax.set_xticks(self.k_values)
            plot_idx += 1
        
        # Plot perplexity
        if has_perplexity:
            ax = axes[plot_idx]
            valid_k = [k for k, s in zip(self.k_values, self.perplexity_scores) 
                      if s is not None]
            valid_scores = [s for s in self.perplexity_scores if s is not None]
            
            if valid_scores:
                ax.plot(valid_k, valid_scores, 'g-o', linewidth=2, markersize=6)
                ax.fill_between(valid_k, valid_scores, alpha=0.2, color='green')
                
                # Find elbow
                elbow_k = self._find_elbow(valid_k, valid_scores)
                if elbow_k:
                    optimal_k_suggestions.append(elbow_k)
                    ax.axvline(x=elbow_k, color='r', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Number of Topics (K)', fontsize=11)
            ax.set_ylabel('Perplexity', fontsize=11)
            ax.set_title('Model Perplexity', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if self.k_values:
                ax.set_xticks(self.k_values)
            plot_idx += 1
        
        # Plot diversity
        if has_diversity:
            ax = axes[plot_idx]
            valid_k = [k for k, s in zip(self.k_values, self.diversity_scores) 
                      if s is not None]
            valid_scores = [s for s in self.diversity_scores if s is not None]
            
            if valid_scores:
                ax.plot(valid_k, valid_scores, 'm-o', linewidth=2, markersize=6)
                ax.fill_between(valid_k, valid_scores, alpha=0.2, color='magenta')
            
            ax.set_xlabel('Number of Topics (K)', fontsize=11)
            ax.set_ylabel('Topic Diversity', fontsize=11)
            ax.set_title('Topic Diversity', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            if self.k_values:
                ax.set_xticks(self.k_values)
        
        # Add overall title with recommendation
        if optimal_k_suggestions:
            from collections import Counter
            k_counts = Counter(optimal_k_suggestions)
            recommended_k = k_counts.most_common(1)[0][0]
            fig.suptitle(
                title or f'Topic Number (K) Evaluation - Recommended K={recommended_k}',
                fontsize=14, fontweight='bold', y=1.02
            )
        else:
            fig.suptitle(title or 'Topic Number (K) Evaluation',
                        fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return self._save_or_show(fig, filename)
    
    def plot_stm_style_evaluation(
        self,
        k_values: List[int],
        semantic_coherence: List[float],
        exclusivity: List[float],
        held_out_likelihood: List[float] = None,
        filename: Optional[str] = None,
        title: str = None
    ) -> Optional[str]:
        """
        Plot STM-style multi-metric evaluation (like 图2 主题数量评估指标).
        
        Shows semantic coherence, exclusivity, and held-out likelihood
        on the same plot with multiple Y-axes.
        
        Args:
            k_values: List of K values evaluated
            semantic_coherence: Semantic coherence scores for each K
            exclusivity: Exclusivity scores for each K
            held_out_likelihood: Optional held-out likelihood scores
            filename: Output filename
            title: Plot title
            
        Returns:
            Path to saved figure or None
        """
        fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
        ax1.set_facecolor('white')
        
        # Plot semantic coherence on left Y-axis
        line1, = ax1.plot(k_values, semantic_coherence, 'k-', linewidth=1.5,
                         marker='D', markersize=6, label='语义一致性')
        ax1.set_xlabel('主题数', fontsize=12)
        ax1.set_ylabel('语义一致性', fontsize=12)
        ax1.tick_params(axis='y')
        
        # Create second Y-axis for exclusivity
        ax2 = ax1.twinx()
        line2, = ax2.plot(k_values, exclusivity, 'k-', linewidth=1.5,
                         marker='o', markersize=6, label='独占性')
        ax2.set_ylabel('独占性', fontsize=12)
        ax2.tick_params(axis='y')
        
        # Create third Y-axis for held-out likelihood if provided
        lines = [line1, line2]
        labels = ['语义一致性', '独占性']
        
        if held_out_likelihood is not None:
            ax3 = ax1.twinx()
            # Offset the third axis
            ax3.spines['right'].set_position(('outward', 60))
            line3, = ax3.plot(k_values, held_out_likelihood, 'k-', linewidth=1.5,
                             marker='^', markersize=6, label='保留数据似然值')
            ax3.set_ylabel('保留数据似然值', fontsize=12)
            ax3.tick_params(axis='y')
            lines.append(line3)
            labels.append('保留数据似然值')
        
        # Add legend
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12),
                  ncol=3, fontsize=10, frameon=True)
        
        # Style
        ax1.spines['top'].set_visible(False)
        
        # Set x-axis ticks
        ax1.set_xticks(k_values)
        
        # Title
        if title is None:
            title = '图2  主题数量评估指标'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.02)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        return self._save_or_show(fig, filename)
    
    def add_stm_metrics(
        self,
        k: int,
        semantic_coherence: float = None,
        exclusivity: float = None,
        held_out_likelihood: float = None
    ):
        """
        Add STM-style metrics for a specific K value.
        
        Args:
            k: Number of topics
            semantic_coherence: Semantic coherence score
            exclusivity: Exclusivity score
            held_out_likelihood: Held-out likelihood score
        """
        if not hasattr(self, 'stm_metrics'):
            self.stm_metrics = {
                'k_values': [],
                'semantic_coherence': [],
                'exclusivity': [],
                'held_out_likelihood': []
            }
        
        self.stm_metrics['k_values'].append(k)
        self.stm_metrics['semantic_coherence'].append(semantic_coherence)
        self.stm_metrics['exclusivity'].append(exclusivity)
        self.stm_metrics['held_out_likelihood'].append(held_out_likelihood)
    
    def plot_stm_metrics(
        self,
        filename: Optional[str] = None,
        title: str = None
    ) -> Optional[str]:
        """
        Plot accumulated STM metrics.
        
        Call add_stm_metrics() for each K first, then call this method.
        
        Args:
            filename: Output filename
            title: Plot title
            
        Returns:
            Path to saved figure or None
        """
        if not hasattr(self, 'stm_metrics') or not self.stm_metrics['k_values']:
            logger.warning("No STM metrics available. Call add_stm_metrics() first.")
            return None
        
        # Sort by K
        sorted_indices = np.argsort(self.stm_metrics['k_values'])
        k_values = [self.stm_metrics['k_values'][i] for i in sorted_indices]
        semantic_coherence = [self.stm_metrics['semantic_coherence'][i] for i in sorted_indices]
        exclusivity = [self.stm_metrics['exclusivity'][i] for i in sorted_indices]
        held_out_likelihood = [self.stm_metrics['held_out_likelihood'][i] for i in sorted_indices]
        
        # Filter out None values
        valid_hol = [h for h in held_out_likelihood if h is not None]
        
        return self.plot_stm_style_evaluation(
            k_values=k_values,
            semantic_coherence=semantic_coherence,
            exclusivity=exclusivity,
            held_out_likelihood=held_out_likelihood if valid_hol else None,
            filename=filename,
            title=title
        )
    
    def _find_elbow(
        self,
        k_values: List[int],
        scores: List[float]
    ) -> Optional[int]:
        """
        Find elbow point in the curve using the kneedle algorithm.
        
        Args:
            k_values: List of K values
            scores: Corresponding scores
            
        Returns:
            K value at elbow point or None
        """
        if len(k_values) < 3:
            return None
        
        # Normalize
        k_norm = np.array(k_values)
        s_norm = np.array(scores)
        
        k_norm = (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min() + 1e-10)
        s_norm = (s_norm - s_norm.min()) / (s_norm.max() - s_norm.min() + 1e-10)
        
        # Calculate distance from line connecting first and last points
        p1 = np.array([k_norm[0], s_norm[0]])
        p2 = np.array([k_norm[-1], s_norm[-1]])
        
        distances = []
        for i in range(len(k_norm)):
            p = np.array([k_norm[i], s_norm[i]])
            # Distance from point to line
            d = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)
            distances.append(d)
        
        # Elbow is the point with maximum distance
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]
    
    def get_recommendation(self) -> Dict:
        """
        Get K recommendation based on all available metrics.
        
        Returns:
            Dictionary with recommendation and reasoning
        """
        recommendations = {}
        
        # Coherence-based recommendation (higher is better)
        for metric, scores in self.coherence_scores.items():
            valid_k = [k for k, s in zip(self.k_values, scores) if s is not None]
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                optimal_idx = np.argmax(valid_scores)
                recommendations[f'coherence_{metric}'] = {
                    'k': valid_k[optimal_idx],
                    'score': valid_scores[optimal_idx],
                    'reasoning': f'Highest {metric.upper()} coherence'
                }
        
        # Perplexity-based recommendation (elbow method)
        valid_k = [k for k, s in zip(self.k_values, self.perplexity_scores) 
                   if s is not None]
        valid_scores = [s for s in self.perplexity_scores if s is not None]
        if valid_scores:
            elbow_k = self._find_elbow(valid_k, valid_scores)
            if elbow_k:
                recommendations['perplexity_elbow'] = {
                    'k': elbow_k,
                    'score': valid_scores[valid_k.index(elbow_k)],
                    'reasoning': 'Elbow point in perplexity curve'
                }
        
        # Overall recommendation (mode of all suggestions)
        if recommendations:
            all_k = [r['k'] for r in recommendations.values()]
            from collections import Counter
            k_counts = Counter(all_k)
            final_k = k_counts.most_common(1)[0][0]
            
            return {
                'recommended_k': final_k,
                'confidence': k_counts[final_k] / len(all_k),
                'individual_recommendations': recommendations,
                'all_k_values': self.k_values
            }
        
        return {'recommended_k': None, 'message': 'Insufficient data for recommendation'}
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export evaluation results to a pandas DataFrame.
        
        Returns:
            DataFrame with all evaluation metrics
        """
        data = {'K': self.k_values}
        
        for metric, scores in self.coherence_scores.items():
            data[f'coherence_{metric}'] = scores[:len(self.k_values)]
        
        if self.perplexity_scores:
            data['perplexity'] = self.perplexity_scores[:len(self.k_values)]
        
        if self.diversity_scores:
            data['diversity'] = self.diversity_scores[:len(self.k_values)]
        
        return pd.DataFrame(data)
    
    def save_results(self, filepath: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            filepath: Output file path
        """
        results = {
            'k_values': self.k_values,
            'coherence_scores': self.coherence_scores,
            'perplexity_scores': self.perplexity_scores,
            'diversity_scores': self.diversity_scores,
            'recommendation': self.get_recommendation()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")


def evaluate_k_range(
    k_range: List[int],
    doc_term_matrix: np.ndarray,
    vocab: List[str],
    model_results_dir: str,
    output_dir: str = None
) -> TopicKEvaluator:
    """
    Convenience function to evaluate a range of K values.
    
    This function expects pre-trained model results for each K value
    in the model_results_dir with subdirectories named by K value.
    
    Args:
        k_range: List of K values to evaluate
        doc_term_matrix: Document-term matrix
        vocab: Vocabulary list
        model_results_dir: Directory containing model results
        output_dir: Directory to save visualizations
        
    Returns:
        TopicKEvaluator with all results
    """
    evaluator = TopicKEvaluator(output_dir=output_dir)
    
    for k in k_range:
        k_dir = os.path.join(model_results_dir, f'k_{k}')
        if not os.path.exists(k_dir):
            logger.warning(f"No results found for K={k} at {k_dir}")
            continue
        
        try:
            # Load model outputs
            theta_path = os.path.join(k_dir, 'theta.npy')
            beta_path = os.path.join(k_dir, 'beta.npy')
            
            if os.path.exists(theta_path) and os.path.exists(beta_path):
                theta = np.load(theta_path)
                beta = np.load(beta_path)
                
                # Calculate metrics
                coherence_npmi = evaluator.calculate_coherence_npmi(
                    beta, doc_term_matrix, vocab
                )
                coherence_umass = evaluator.calculate_coherence_umass(
                    beta, doc_term_matrix
                )
                perplexity = evaluator.calculate_perplexity(
                    theta, beta, doc_term_matrix
                )
                diversity = evaluator.calculate_topic_diversity(beta)
                
                evaluator.add_evaluation_result(
                    k=k,
                    coherence_npmi=coherence_npmi,
                    coherence_umass=coherence_umass,
                    perplexity=perplexity,
                    diversity=diversity
                )
                
                logger.info(f"Evaluated K={k}: NPMI={coherence_npmi:.4f}, "
                           f"Perplexity={perplexity:.2f}, Diversity={diversity:.4f}")
        
        except Exception as e:
            logger.error(f"Error evaluating K={k}: {e}")
    
    return evaluator

"""
Baseline Evaluator - Unified evaluator for baseline models

Ensures that outputs from baseline models (LDA, ETM, CTM) are compatible with THETA's evaluation and visualization system.

Evaluation Metrics:
- Perplexity
- Topic Diversity TD
- Topic Diversity iRBO (Rank-based diversity)
- Topic Coherence NPMI (Normalized Pointwise Mutual Information)
- Topic Coherence C_V
- Topic Coherence UMass
- Topic Exclusivity
"""

import os
import json
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd

# Import THETA evaluation modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.topic_metrics import (
    compute_all_metrics,
    compute_topic_diversity_td,
    compute_topic_diversity_inverted_rbo,
    compute_topic_coherence_npmi,
    compute_topic_coherence_cv,
    compute_topic_coherence_umass,
    compute_topic_exclusivity,
    compute_perplexity
)

from visualization.topic_visualizer import TopicVisualizer


class BaselineEvaluator:
    """
    Baseline model evaluator
    
    Uses the same evaluation metrics and visualization methods as THETA to ensure fair comparison.
    """
    
    def __init__(
        self,
        result_dir: str,
        dataset: str,
        bow_matrix: np.ndarray = None,
        vocab: List[str] = None
    ):
        """
        Initialize evaluator
        
        Args:
            result_dir: Baseline result directory
            dataset: Dataset name
            bow_matrix: BOW matrix (for coherence computation)
            vocab: Vocabulary
        """
        self.result_dir = result_dir
        self.dataset = dataset
        self.bow_matrix = bow_matrix
        self.vocab = vocab
        
        # Load BOW matrix (if not provided)
        if self.bow_matrix is None:
            bow_path = os.path.join(result_dir, 'bow_matrix.npz')
            if os.path.exists(bow_path):
                self.bow_matrix = sp.load_npz(bow_path)
        
        # Load vocabulary (if not provided)
        if self.vocab is None:
            vocab_path = os.path.join(result_dir, 'vocab.json')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.vocab = json.load(f)
    
    def evaluate_model(
        self,
        model_name: str,
        theta: np.ndarray,
        beta: np.ndarray,
        num_topics: int = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model
        
        Args:
            model_name: Model name ('lda', 'etm', 'ctm')
            theta: Document-topic distribution (D x K)
            beta: Topic-word distribution (K x V)
            num_topics: Number of topics
            
        Returns:
            Evaluation metrics dictionary
        """
        if num_topics is None:
            num_topics = beta.shape[0]
        
        print(f"\nEvaluating {model_name.upper()} (K={num_topics})...")
        
        # Use THETA evaluation functions
        metrics = compute_all_metrics(
            beta=beta,
            theta=theta,
            doc_term_matrix=self.bow_matrix,
            top_k_coherence=10,
            top_k_diversity=25,
            compute_extended=True
        )
        
        # Add model information
        metrics['model'] = model_name
        metrics['num_topics'] = num_topics
        metrics['num_docs'] = theta.shape[0]
        metrics['vocab_size'] = beta.shape[1]
        
        return metrics
    
    def evaluate_from_files(
        self,
        model_name: str,
        num_topics: int = 20
    ) -> Dict[str, Any]:
        """
        Load results from files and evaluate
        
        Args:
            model_name: Model name
            num_topics: Number of topics
            
        Returns:
            Evaluation metrics dictionary
        """
        # Determine model directory
        if model_name == 'ctm':
            model_dir = os.path.join(self.result_dir, 'ctm_zeroshot')
            if not os.path.exists(model_dir):
                model_dir = os.path.join(self.result_dir, 'ctm_combined')
        else:
            model_dir = os.path.join(self.result_dir, model_name)
        
        # Load theta and beta
        theta_path = os.path.join(model_dir, f'theta_k{num_topics}.npy')
        beta_path = os.path.join(model_dir, f'beta_k{num_topics}.npy')
        
        if not os.path.exists(theta_path) or not os.path.exists(beta_path):
            raise FileNotFoundError(f"Cannot find result files for {model_name}: {theta_path}, {beta_path}")
        
        theta = np.load(theta_path)
        beta = np.load(beta_path)
        
        return self.evaluate_model(model_name, theta, beta, num_topics)
    
    def evaluate_all_baselines(
        self,
        models: List[str] = None,
        num_topics: int = 20
    ) -> pd.DataFrame:
        """
        Evaluate all baseline models
        
        Args:
            models: Model list
            num_topics: Number of topics
            
        Returns:
            Evaluation results DataFrame
        """
        if models is None:
            models = ['lda', 'etm', 'ctm']
        
        results = []
        
        for model_name in models:
            try:
                metrics = self.evaluate_from_files(model_name, num_topics)
                results.append(metrics)
                print(f"\n{model_name.upper()} evaluation completed:")
                print(f"  - Perplexity: {metrics.get('perplexity', 'N/A')}")
                print(f"  - Diversity (TD): {metrics['topic_diversity_td']:.4f}")
                print(f"  - Coherence (NPMI): {metrics['topic_coherence_npmi_avg']:.4f}")
                print(f"  - Exclusivity: {metrics.get('topic_exclusivity_avg', 'N/A')}")
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Select key columns
        key_columns = [
            'model', 'num_topics', 'perplexity',
            'topic_diversity_td', 'topic_diversity_irbo',
            'topic_coherence_npmi_avg', 'topic_coherence_cv_avg',
            'topic_coherence_umass_avg', 'topic_exclusivity_avg'
        ]
        df = df[[c for c in key_columns if c in df.columns]]
        
        return df
    
    def save_evaluation_results(
        self,
        results: pd.DataFrame,
        output_path: str = None
    ):
        """
        Save evaluation results (compatible with THETA format)
        
        Args:
            results: Evaluation results DataFrame
            output_path: Output path
        """
        if output_path is None:
            output_path = os.path.join(self.result_dir, 'baseline_evaluation_metrics.csv')
        
        results.to_csv(output_path, index=False)
        print(f"\nEvaluation results saved to: {output_path}")
        
        # Also save in JSON format
        json_path = output_path.replace('.csv', '.json')
        results.to_json(json_path, orient='records', indent=2)
    
    def visualize_model(
        self,
        model_name: str,
        theta: np.ndarray,
        beta: np.ndarray,
        topic_words: Dict[str, List[str]],
        output_dir: str = None
    ):
        """
        Visualize model results using THETA's visualization tools
        
        Args:
            model_name: Model name
            theta: Document-topic distribution
            beta: Topic-word distribution
            topic_words: Topic words dictionary
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = os.path.join(self.result_dir, model_name, 'visualizations')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizer
        visualizer = TopicVisualizer(output_dir=output_dir)
        
        # Convert topic_words format to THETA format
        # THETA format: List[Tuple[int, List[Tuple[str, float]]]]
        topic_words_theta_format = []
        for topic_key, words in topic_words.items():
            topic_idx = int(topic_key.split('_')[1])
            # Get corresponding probabilities
            word_probs = []
            for word in words:
                if self.vocab and word in self.vocab:
                    word_idx = self.vocab.index(word)
                    prob = beta[topic_idx, word_idx]
                else:
                    prob = 1.0 / len(words)  # Uniform distribution as fallback
                word_probs.append((word, float(prob)))
            topic_words_theta_format.append((topic_idx, word_probs))
        
        # Sort
        topic_words_theta_format.sort(key=lambda x: x[0])
        
        print(f"\nGenerating {model_name.upper()} visualizations...")
        
        # 1. Topic words bar chart
        visualizer.visualize_topic_words(
            topic_words_theta_format,
            num_words=10,
            filename=f'{model_name}_topic_words.png'
        )
        
        # 2. Topic word clouds
        visualizer.visualize_all_wordclouds(
            topic_words_theta_format,
            num_words=30,
            filename=f'{model_name}_wordclouds.png'
        )
        
        # 3. Topic similarity heatmap
        visualizer.visualize_topic_similarity(
            beta,
            topic_words_theta_format,
            filename=f'{model_name}_topic_similarity.png'
        )
        
        # 4. Document-topic distribution
        visualizer.visualize_document_topics(
            theta,
            method='umap',
            topic_words=topic_words_theta_format,
            filename=f'{model_name}_document_topics.png'
        )
        
        # 5. Topic proportions
        visualizer.visualize_topic_proportions(
            theta,
            topic_words=topic_words_theta_format,
            filename=f'{model_name}_topic_proportions.png'
        )
        
        print(f"Visualizations saved to: {output_dir}")
    
    def visualize_from_files(
        self,
        model_name: str,
        num_topics: int = 20
    ):
        """
        Load results from files and visualize
        
        Args:
            model_name: Model name
            num_topics: Number of topics
        """
        # Determine model directory
        if model_name == 'ctm':
            model_dir = os.path.join(self.result_dir, 'ctm_zeroshot')
            if not os.path.exists(model_dir):
                model_dir = os.path.join(self.result_dir, 'ctm_combined')
        else:
            model_dir = os.path.join(self.result_dir, model_name)
        
        # Load data
        theta = np.load(os.path.join(model_dir, f'theta_k{num_topics}.npy'))
        beta = np.load(os.path.join(model_dir, f'beta_k{num_topics}.npy'))
        
        with open(os.path.join(model_dir, f'topic_words_k{num_topics}.json'), 'r', encoding='utf-8') as f:
            topic_words = json.load(f)
        
        self.visualize_model(model_name, theta, beta, topic_words)
    
    def compare_with_theta(
        self,
        theta_result_dir: str,
        mode: str = 'zero_shot',
        num_topics: int = 20
    ) -> pd.DataFrame:
        """
        Compare with THETA method
        
        Args:
            theta_result_dir: THETA result directory
            mode: THETA mode
            num_topics: Number of topics
            
        Returns:
            Comparison results DataFrame
        """
        # Load THETA results
        theta_model_dir = os.path.join(theta_result_dir, self.dataset, mode, 'model')
        
        # Find latest timestamp
        import glob
        theta_files = sorted(glob.glob(os.path.join(theta_model_dir, 'theta_*.npy')), reverse=True)
        
        if not theta_files:
            print(f"Cannot find THETA results: {theta_model_dir}")
            return None
        
        # Extract timestamp
        timestamp = os.path.basename(theta_files[0]).replace('theta_', '').replace('.npy', '')
        
        theta_theta = np.load(os.path.join(theta_model_dir, f'theta_{timestamp}.npy'))
        theta_beta = np.load(os.path.join(theta_model_dir, f'beta_{timestamp}.npy'))
        
        # Evaluate THETA
        theta_metrics = self.evaluate_model('THETA', theta_theta, theta_beta)
        
        # Evaluate Baseline
        baseline_results = self.evaluate_all_baselines(num_topics=num_topics)
        
        # Merge results
        theta_df = pd.DataFrame([theta_metrics])
        all_results = pd.concat([baseline_results, theta_df], ignore_index=True)
        
        return all_results


def compare_all_models(
    dataset: str,
    baseline_dir: str = None,
    theta_dir: str = None,
    mode: str = 'zero_shot',
    num_topics: int = 20,
    models: List[str] = None
) -> pd.DataFrame:
    """
    Convenience function to compare all models
    
    Args:
        dataset: Dataset name
        baseline_dir: Baseline result directory
        theta_dir: THETA result directory
        mode: THETA mode
        num_topics: Number of topics
        models: Baseline model list
        
    Returns:
        Comparison results DataFrame
    """
    # Import config for default paths
    from config import RESULT_DIR
    
    if baseline_dir is None:
        baseline_dir = os.path.join(str(RESULT_DIR), 'baseline')
    if theta_dir is None:
        theta_dir = os.path.join(str(RESULT_DIR), '0.6B')
    
    if models is None:
        models = ['lda', 'etm', 'ctm']
    
    # Create evaluator
    evaluator = BaselineEvaluator(
        result_dir=os.path.join(baseline_dir, dataset),
        dataset=dataset
    )
    
    # Evaluate Baseline
    baseline_results = evaluator.evaluate_all_baselines(models=models, num_topics=num_topics)
    
    # Try to load THETA results
    try:
        theta_model_dir = os.path.join(theta_dir, dataset, mode, 'model')
        import glob
        theta_files = sorted(glob.glob(os.path.join(theta_model_dir, 'theta_*.npy')), reverse=True)
        
        if theta_files:
            timestamp = os.path.basename(theta_files[0]).replace('theta_', '').replace('.npy', '')
            theta_theta = np.load(os.path.join(theta_model_dir, f'theta_{timestamp}.npy'))
            theta_beta = np.load(os.path.join(theta_model_dir, f'beta_{timestamp}.npy'))
            
            theta_metrics = evaluator.evaluate_model('THETA', theta_theta, theta_beta)
            theta_df = pd.DataFrame([theta_metrics])
            baseline_results = pd.concat([baseline_results, theta_df], ignore_index=True)
    except Exception as e:
        print(f"Failed to load THETA results: {e}")
    
    # Save results
    output_path = os.path.join(baseline_dir, dataset, 'comparison_results.csv')
    baseline_results.to_csv(output_path, index=False)
    print(f"\nComparison results saved to: {output_path}")
    
    return baseline_results


def print_comparison_table(results: pd.DataFrame):
    """
    Print comparison table
    
    Args:
        results: Comparison results DataFrame
    """
    print("\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    
    # Format output
    display_cols = {
        'model': 'Model',
        'perplexity': 'Perplexity↓',
        'topic_diversity_td': 'TD↑',
        'topic_diversity_irbo': 'iRBO↑',
        'topic_coherence_npmi_avg': 'NPMI↑',
        'topic_coherence_cv_avg': 'C_V↑',
        'topic_exclusivity_avg': 'Exclusivity↑'
    }
    
    # Select existing columns
    cols = [c for c in display_cols.keys() if c in results.columns]
    df_display = results[cols].copy()
    df_display.columns = [display_cols[c] for c in cols]
    
    # Format numeric values
    for col in df_display.columns[1:]:
        df_display[col] = df_display[col].apply(
            lambda x: f'{x:.4f}' if pd.notna(x) and isinstance(x, (int, float)) else str(x)
        )
    
    print(df_display.to_string(index=False))
    print("="*80)
    print("↑ = higher is better, ↓ = lower is better")

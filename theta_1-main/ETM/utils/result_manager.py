"""
Result Manager - Standardized result directory management for topic models

Directory Structure:
/root/autodl-tmp/result/baseline/{dataset}/{model}/
├── bow/                    # BOW data and vocabulary
│   ├── bow_matrix.npz
│   ├── vocab.json
│   ├── vocab.txt
│   └── vocab_embeddings.npy (optional)
├── model/                  # Model parameter files
│   ├── theta_k{K}.npy
│   ├── beta_k{K}.npy
│   ├── beta_over_time_k{K}.npy (DTM only)
│   ├── model_k{K}.pt
│   └── training_history_k{K}.json
├── evaluation/             # Evaluation results
│   ├── metrics_k{K}.json
│   └── metrics_k{K}.csv
├── topicwords/             # Topic words related
│   ├── topic_words_k{K}.json
│   └── topic_evolution_k{K}.json (DTM only)
└── visualization_k{K}_{lang}_{timestamp}/  # Visualization results

Usage:
    # Initialize manager
    manager = ResultManager('/root/autodl-tmp/result/baseline', 'edu_data', 'dtm', num_topics=20)
    
    # Save all results
    manager.save_all(theta, beta, vocab, topic_words, metrics=metrics)
    
    # Load all results
    data = manager.load_all(num_topics=20)
    
    # Migrate old structure
    ResultManager.migrate_old_structure(old_dir, new_result_dir, dataset, model)
"""

import os
import json
import csv
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class ResultManager:
    """Standardized result directory manager for topic models."""
    
    def __init__(
        self,
        result_dir: str,
        dataset: str,
        model: str,
        num_topics: int = None
    ):
        """
        Initialize result manager.
        
        Args:
            result_dir: Root result directory (e.g., /root/autodl-tmp/result/baseline)
            dataset: Dataset name (e.g., edu_data)
            model: Model name (e.g., dtm, lda, ctm, etm)
            num_topics: Number of topics (optional)
        """
        self.result_dir = Path(result_dir)
        self.dataset = dataset
        self.model = model
        self.num_topics = num_topics
        
        self.base_dir = self.result_dir / dataset / model
        self.bow_dir = self.base_dir / 'bow'
        self.model_dir = self.base_dir / 'model'
        self.evaluation_dir = self.base_dir / 'evaluation'
        self.topicwords_dir = self.base_dir / 'topicwords'
        
        self._create_dirs()
    
    def _create_dirs(self):
        """Create all necessary directories."""
        for dir_path in [self.bow_dir, self.model_dir, self.evaluation_dir, self.topicwords_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _get_k_suffix(self, num_topics: int = None) -> str:
        """Get topic number suffix for filenames."""
        k = num_topics or self.num_topics
        return f'_k{k}' if k else ''
    
    def save_bow(
        self,
        bow_matrix: Union[np.ndarray, sp.spmatrix],
        vocab: Union[List[str], Dict[str, int]],
        vocab_embeddings: np.ndarray = None
    ):
        """Save BOW data."""
        # Save as dense npy format
        bow_dense = bow_matrix.toarray() if sp.issparse(bow_matrix) else bow_matrix
        np.save(self.bow_dir / 'bow_matrix.npy', bow_dense)
        
        vocab_list = list(vocab.keys()) if isinstance(vocab, dict) else list(vocab)
        
        with open(self.bow_dir / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(vocab_list, f, ensure_ascii=False, indent=2)
        
        with open(self.bow_dir / 'vocab.txt', 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(f'{word}\n')
        
        if vocab_embeddings is not None:
            np.save(self.bow_dir / 'vocab_embeddings.npy', vocab_embeddings)
        
        print(f"  [OK] BOW data saved to: {self.bow_dir}")
    
    def load_bow(self) -> Dict[str, Any]:
        """Load BOW data."""
        result = {}
        
        bow_path = self.bow_dir / 'bow_matrix.npy'
        if bow_path.exists():
            result['bow_matrix'] = np.load(bow_path)
        
        vocab_path = self.bow_dir / 'vocab.json'
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                result['vocab'] = json.load(f)
        else:
            vocab_txt_path = self.bow_dir / 'vocab.txt'
            if vocab_txt_path.exists():
                with open(vocab_txt_path, 'r', encoding='utf-8') as f:
                    result['vocab'] = [line.strip() for line in f]
        
        emb_path = self.bow_dir / 'vocab_embeddings.npy'
        if emb_path.exists():
            result['vocab_embeddings'] = np.load(emb_path)
        
        return result
    
    def save_model_params(
        self,
        theta: np.ndarray,
        beta: np.ndarray,
        num_topics: int = None,
        beta_over_time: np.ndarray = None,
        training_history: Dict = None,
        model_state: Any = None,
        additional_params: Dict = None
    ):
        """Save model parameters."""
        k_suffix = self._get_k_suffix(num_topics)
        
        np.save(self.model_dir / f'theta{k_suffix}.npy', theta)
        np.save(self.model_dir / f'beta{k_suffix}.npy', beta)
        
        if beta_over_time is not None:
            np.save(self.model_dir / f'beta_over_time{k_suffix}.npy', beta_over_time)
        
        if training_history is not None:
            with open(self.model_dir / f'training_history{k_suffix}.json', 'w') as f:
                json.dump(training_history, f, indent=2)
        
        if model_state is not None:
            import torch
            torch.save(model_state, self.model_dir / f'model{k_suffix}.pt')
        
        if additional_params is not None:
            with open(self.model_dir / f'params{k_suffix}.json', 'w') as f:
                json.dump(additional_params, f, indent=2)
        
        print(f"  [OK] Model params saved to: {self.model_dir}")
    
    def load_model_params(self, num_topics: int = None) -> Dict[str, Any]:
        """Load model parameters."""
        k_suffix = self._get_k_suffix(num_topics)
        result = {}
        
        theta_path = self.model_dir / f'theta{k_suffix}.npy'
        if theta_path.exists():
            result['theta'] = np.load(theta_path)
        
        beta_path = self.model_dir / f'beta{k_suffix}.npy'
        if beta_path.exists():
            result['beta'] = np.load(beta_path)
        
        beta_time_path = self.model_dir / f'beta_over_time{k_suffix}.npy'
        if beta_time_path.exists():
            result['beta_over_time'] = np.load(beta_time_path)
        
        history_path = self.model_dir / f'training_history{k_suffix}.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                result['training_history'] = json.load(f)
        
        return result
    
    def save_evaluation(
        self,
        metrics: Dict[str, Any],
        num_topics: int = None,
        save_csv: bool = True
    ):
        """Save evaluation results in JSON and CSV formats."""
        k_suffix = self._get_k_suffix(num_topics)
        
        json_path = self.evaluation_dir / f'metrics{k_suffix}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        if save_csv:
            csv_path = self.evaluation_dir / f'metrics{k_suffix}.csv'
            self._save_metrics_csv(metrics, csv_path)
        
        print(f"  [OK] Evaluation saved to: {self.evaluation_dir}")
    
    def _save_metrics_csv(self, metrics: Dict[str, Any], csv_path: Path):
        """Save metrics as CSV format."""
        flat_metrics = self._flatten_dict(metrics)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in flat_metrics.items():
                if isinstance(value, (int, float)):
                    writer.writerow([key, f'{value:.6f}' if isinstance(value, float) else value])
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                    avg_value = sum(value) / len(value)
                    writer.writerow([f'{key}_avg', f'{avg_value:.6f}'])
                    for i, v in enumerate(value):
                        writer.writerow([f'{key}_{i}', f'{v:.6f}' if isinstance(v, float) else v])
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def load_evaluation(self, num_topics: int = None) -> Dict[str, Any]:
        """Load evaluation results."""
        k_suffix = self._get_k_suffix(num_topics)
        json_path = self.evaluation_dir / f'metrics{k_suffix}.json'
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_topic_words(
        self,
        topic_words: Union[Dict, List],
        num_topics: int = None,
        topic_evolution: Dict = None
    ):
        """Save topic words and evolution data."""
        k_suffix = self._get_k_suffix(num_topics)
        
        with open(self.topicwords_dir / f'topic_words{k_suffix}.json', 'w', encoding='utf-8') as f:
            json.dump(topic_words, f, ensure_ascii=False, indent=2)
        
        if topic_evolution is not None:
            with open(self.topicwords_dir / f'topic_evolution{k_suffix}.json', 'w', encoding='utf-8') as f:
                json.dump(topic_evolution, f, ensure_ascii=False, indent=2)
        
        print(f"  [OK] Topic words saved to: {self.topicwords_dir}")
    
    def load_topic_words(self, num_topics: int = None) -> Dict[str, Any]:
        """Load topic words and evolution data."""
        k_suffix = self._get_k_suffix(num_topics)
        result = {}
        
        tw_path = self.topicwords_dir / f'topic_words{k_suffix}.json'
        if tw_path.exists():
            with open(tw_path, 'r', encoding='utf-8') as f:
                result['topic_words'] = json.load(f)
        
        te_path = self.topicwords_dir / f'topic_evolution{k_suffix}.json'
        if te_path.exists():
            with open(te_path, 'r', encoding='utf-8') as f:
                result['topic_evolution'] = json.load(f)
        
        return result
    
    def get_visualization_dir(
        self,
        num_topics: int = None,
        language: str = 'zh',
        timestamp: str = None
    ) -> Path:
        """Get visualization output directory."""
        k = num_topics or self.num_topics
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        viz_dir = self.base_dir / f'visualization_k{k}_{language}_{timestamp}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        return viz_dir
    
    def save_all(
        self,
        theta: np.ndarray,
        beta: np.ndarray,
        vocab: List[str],
        topic_words: Union[Dict, List],
        metrics: Dict[str, Any] = None,
        num_topics: int = None,
        bow_matrix: Union[np.ndarray, sp.spmatrix] = None,
        beta_over_time: np.ndarray = None,
        topic_evolution: Dict = None,
        training_history: Dict = None,
        model_state: Any = None
    ):
        """Save all results to standardized directory structure."""
        print(f"\n{'='*60}")
        print(f"Saving results to: {self.base_dir}")
        print(f"{'='*60}")
        
        if bow_matrix is not None:
            self.save_bow(bow_matrix, vocab)
        
        self.save_model_params(
            theta=theta,
            beta=beta,
            num_topics=num_topics,
            beta_over_time=beta_over_time,
            training_history=training_history,
            model_state=model_state
        )
        
        self.save_topic_words(
            topic_words=topic_words,
            num_topics=num_topics,
            topic_evolution=topic_evolution
        )
        
        if metrics is not None:
            self.save_evaluation(metrics, num_topics)
        
        print(f"{'='*60}\n")
    
    def load_all(self, num_topics: int = None) -> Dict[str, Any]:
        """Load all results from standardized directory structure."""
        result = {}
        
        bow_data = self.load_bow()
        result.update(bow_data)
        
        model_data = self.load_model_params(num_topics)
        result.update(model_data)
        
        tw_data = self.load_topic_words(num_topics)
        result.update(tw_data)
        
        metrics = self.load_evaluation(num_topics)
        if metrics:
            result['metrics'] = metrics
        
        return result
    
    @classmethod
    def migrate_old_structure(
        cls,
        old_dir: str,
        new_result_dir: str,
        dataset: str,
        model: str,
        num_topics_list: List[int] = None
    ):
        """
        Migrate old flat directory structure to new standardized structure.
        
        Args:
            old_dir: Old directory path
            new_result_dir: New result root directory
            dataset: Dataset name
            model: Model name
            num_topics_list: List of topic numbers to migrate (auto-detected if None)
        """
        old_path = Path(old_dir)
        
        if not old_path.exists():
            print(f"Old directory not found: {old_dir}")
            return
        
        if num_topics_list is None:
            num_topics_list = []
            for f in old_path.glob('theta_k*.npy'):
                k = int(f.stem.split('_k')[1])
                if k not in num_topics_list:
                    num_topics_list.append(k)
        
        print(f"Migrating data from: {old_dir}")
        print(f"Found topic numbers: {num_topics_list}")
        
        for num_topics in num_topics_list:
            print(f"\n--- Migrating K={num_topics} ---")
            manager = cls(new_result_dir, dataset, model, num_topics)
            k_suffix = f'_k{num_topics}'
            
            for old_name, new_subdir in [
                (f'theta{k_suffix}.npy', 'model'),
                (f'beta{k_suffix}.npy', 'model'),
                (f'beta_over_time{k_suffix}.npy', 'model'),
                (f'training_history{k_suffix}.json', 'model'),
            ]:
                old_file = old_path / old_name
                if old_file.exists():
                    new_dir = manager.base_dir / new_subdir
                    new_file = new_dir / old_name
                    if not new_file.exists():
                        import shutil
                        shutil.copy2(old_file, new_file)
                        print(f"  [OK] Copied: {old_name} -> {new_subdir}/")
            
            for old_name in [f'topic_words{k_suffix}.json', f'topic_evolution{k_suffix}.json']:
                old_file = old_path / old_name
                if old_file.exists():
                    new_file = manager.topicwords_dir / old_name
                    if not new_file.exists():
                        import shutil
                        shutil.copy2(old_file, new_file)
                        print(f"  [OK] Copied: {old_name} -> topicwords/")
            
            old_metrics = old_path / f'metrics{k_suffix}.json'
            if old_metrics.exists():
                new_metrics = manager.evaluation_dir / f'metrics{k_suffix}.json'
                if not new_metrics.exists():
                    import shutil
                    shutil.copy2(old_metrics, new_metrics)
                    print(f"  [OK] Copied: metrics{k_suffix}.json -> evaluation/")
                    
                    with open(old_metrics, 'r') as f:
                        metrics = json.load(f)
                    manager._save_metrics_csv(metrics, manager.evaluation_dir / f'metrics{k_suffix}.csv')
                    print(f"  [OK] Generated: metrics{k_suffix}.csv")
        
        print(f"\nMigration complete!")


def migrate_baseline_results(
    old_base_dir: str = '/root/autodl-tmp/result/baseline',
    dataset: str = 'edu_data',
    model: str = 'dtm'
):
    """Convenience function to migrate baseline results."""
    old_dir = os.path.join(old_base_dir, dataset, model)
    ResultManager.migrate_old_structure(
        old_dir=old_dir,
        new_result_dir=old_base_dir,
        dataset=dataset,
        model=model
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Result Manager')
    parser.add_argument('--action', type=str, default='migrate', choices=['migrate', 'test'])
    parser.add_argument('--dataset', type=str, default='edu_data')
    parser.add_argument('--model', type=str, default='dtm')
    parser.add_argument('--result_dir', type=str, default='/root/autodl-tmp/result/baseline')
    
    args = parser.parse_args()
    
    if args.action == 'migrate':
        migrate_baseline_results(args.result_dir, args.dataset, args.model)
    elif args.action == 'test':
        manager = ResultManager(args.result_dir, args.dataset, args.model, num_topics=20)
        print(f"Base dir: {manager.base_dir}")
        print(f"BOW dir: {manager.bow_dir}")
        print(f"Model dir: {manager.model_dir}")
        print(f"Evaluation dir: {manager.evaluation_dir}")
        print(f"Topic words dir: {manager.topicwords_dir}")

#!/usr/bin/env python
"""
Experiment Manager

Provides experiment listing, querying, and selection features:
1. API calls (for frontend use)
2. Command-line interactive selection (for testing)

Usage:
    python experiment_manager.py --list-data --dataset edu_data
    
    python experiment_manager.py --list-models --dataset edu_data --model lda
    
    python experiment_manager.py --select-data --dataset edu_data
    
    python experiment_manager.py --select-model --dataset edu_data --model lda
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

# Default paths - use environment variable or auto-detect from this file's location
def _get_default_result_dir():
    """Get default result directory from environment or relative to project root."""
    if os.environ.get("RESULT_DIR"):
        return os.environ["RESULT_DIR"]
    # Auto-detect: this file is at ETM/experiment_manager.py
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / "result")

RESULT_DIR = _get_default_result_dir()


class ExperimentManager:
    """Experiment Manager"""
    
    def __init__(self, result_dir: str = RESULT_DIR):
        self.result_dir = Path(result_dir)
    
    def list_data_experiments(self, dataset: str, model_type: str = 'baseline') -> List[Dict[str, Any]]:
        """
        List data preprocessing experiments
        
        Args:
            dataset: Dataset name
            model_type: Model type ('baseline' or 'theta')
        
        Returns:
            List of experiments, each containing exp_id, created_at, config, etc.
        """
        if model_type == 'theta':
            data_dir = self.result_dir / 'theta' / dataset / 'data'
        else:
            data_dir = self.result_dir / 'baseline' / dataset / 'data'
        
        if not data_dir.exists():
            return []
        
        experiments = []
        for exp_dir in sorted(data_dir.iterdir()):
            if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                exp_info = self._load_experiment_info(exp_dir)
                experiments.append(exp_info)
        
        experiments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return experiments
    
    def list_model_experiments(self, dataset: str, model: str, model_type: str = 'baseline') -> List[Dict[str, Any]]:
        """
        List model training experiments
        
        Args:
            dataset: Dataset name
            model: Model name (lda, hdp, etc.)
            model_type: Model type ('baseline' or 'theta')
        
        Returns:
            List of experiments
        """
        if model_type == 'theta':
            models_dir = self.result_dir / 'theta' / dataset / 'models'
        else:
            models_dir = self.result_dir / 'baseline' / dataset / 'models' / model
        
        if not models_dir.exists():
            return []
        
        experiments = []
        
        if model_type == 'theta':
            for size_mode_dir in models_dir.iterdir():
                if size_mode_dir.is_dir():
                    for exp_dir in size_mode_dir.iterdir():
                        if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                            exp_info = self._load_experiment_info(exp_dir)
                            exp_info['model_config'] = size_mode_dir.name
                            experiments.append(exp_info)
        else:
            for exp_dir in sorted(models_dir.iterdir()):
                if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
                    exp_info = self._load_experiment_info(exp_dir)
                    experiments.append(exp_info)
        
        experiments.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return experiments
    
    def _load_experiment_info(self, exp_dir: Path) -> Dict[str, Any]:
        """Load experiment info"""
        exp_info = {
            'exp_id': exp_dir.name,
            'path': str(exp_dir),
            'created_at': None,
            'config': {}
        }
        
        config_path = exp_dir / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                exp_info['config'] = config
                exp_info['created_at'] = config.get('created_at')
            except:
                pass
        
        if not exp_info['created_at']:
            # exp_20260205_171229_vocab3500 -> 2026-02-05 17:12:29
            parts = exp_dir.name.split('_')
            if len(parts) >= 3:
                try:
                    date_str = parts[1]  # 20260205
                    time_str = parts[2]  # 171229
                    exp_info['created_at'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                except:
                    pass
        
        parts = exp_dir.name.split('_')
        if len(parts) > 3:
            exp_info['label'] = '_'.join(parts[3:])
        else:
            exp_info['label'] = None
        
        return exp_info
    
    def get_latest_data_experiment(self, dataset: str, model_type: str = 'baseline') -> Optional[str]:
        """Get latest data experiment ID"""
        experiments = self.list_data_experiments(dataset, model_type)
        if experiments:
            return experiments[0]['exp_id']
        return None
    
    def get_latest_model_experiment(self, dataset: str, model: str, model_type: str = 'baseline') -> Optional[str]:
        """Get latest model experiment ID"""
        experiments = self.list_model_experiments(dataset, model, model_type)
        if experiments:
            return experiments[0]['exp_id']
        return None
    
    def find_data_experiment(self, dataset: str, query: str, model_type: str = 'baseline') -> Optional[str]:
        """
        Fuzzy search for data experiment
        
        Args:
            dataset: Dataset name
            query: Query string (can be full exp_id, label, or 'latest')
        
        Returns:
            Matching experiment path, or None
        """
        if query == 'latest':
            exp_id = self.get_latest_data_experiment(dataset, model_type)
            if exp_id:
                if model_type == 'theta':
                    return str(self.result_dir / 'theta' / dataset / 'data' / exp_id)
                return str(self.result_dir / 'baseline' / dataset / 'data' / exp_id)
            return None
        
        experiments = self.list_data_experiments(dataset, model_type)
        
        for exp in experiments:
            if exp['exp_id'] == query:
                return exp['path']
        
        for exp in experiments:
            if query in exp['exp_id']:
                return exp['path']
            if exp.get('label') and query in exp['label']:
                return exp['path']
        
        return None
    
    def find_model_experiment(self, dataset: str, model: str, query: str, model_type: str = 'baseline') -> Optional[str]:
        """
        Fuzzy search for model experiment
        
        Args:
            dataset: Dataset name
            model: Model name
            query: Query string (can be full exp_id, label, or 'latest')
        
        Returns:
            Matching experiment path, or None
        """
        if query == 'latest':
            exp_id = self.get_latest_model_experiment(dataset, model, model_type)
            if exp_id:
                if model_type == 'theta':
                    return str(self.result_dir / 'theta' / dataset / 'models' / model / exp_id)
                return str(self.result_dir / 'baseline' / dataset / 'models' / model / exp_id)
            return None
        
        experiments = self.list_model_experiments(dataset, model, model_type)
        
        for exp in experiments:
            if exp['exp_id'] == query:
                return exp['path']
        
        for exp in experiments:
            if query in exp['exp_id']:
                return exp['path']
            if exp.get('label') and query in exp['label']:
                return exp['path']
        
        return None


def interactive_select_data(manager: ExperimentManager, dataset: str, model_type: str = 'baseline') -> Optional[str]:
    """Interactive data experiment selection"""
    experiments = manager.list_data_experiments(dataset, model_type)
    
    if not experiments:
        print(f"No data experiments found: {dataset}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Select data preprocessing experiment ({dataset})")
    print(f"{'='*60}")
    
    for i, exp in enumerate(experiments):
        config = exp.get('config', {})
        vocab_size = config.get('vocab_size', '?')
        created = exp.get('created_at', '?')
        label = exp.get('label', '')
        
        latest_mark = " [latest]" if i == 0 else ""
        label_str = f" ({label})" if label else ""
        
        print(f"  [{i+1}] {exp['exp_id']}{latest_mark}")
        print(f"      vocab_size={vocab_size}, created={created}{label_str}")
    
    print(f"\n  [0] Cancel")
    print(f"{'='*60}")
    
    while True:
        try:
            choice = input("Select (enter number, press Enter for latest): ").strip()
            if choice == '':
                choice = 1
            else:
                choice = int(choice)
            
            if choice == 0:
                return None
            if 1 <= choice <= len(experiments):
                selected = experiments[choice - 1]
                print(f"\nSelected: {selected['exp_id']}")
                return selected['path']
            print("Invalid choice, please try again")
        except ValueError:
            print("Please enter a number")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None


def interactive_select_model(manager: ExperimentManager, dataset: str, model: str, model_type: str = 'baseline') -> Optional[str]:
    """Interactive model experiment selection"""
    experiments = manager.list_model_experiments(dataset, model, model_type)
    
    if not experiments:
        print(f"No model experiments found: {dataset}/{model}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Select model training experiment ({dataset}/{model})")
    print(f"{'='*60}")
    
    for i, exp in enumerate(experiments):
        config = exp.get('config', {})
        num_topics = config.get('num_topics', '?')
        data_exp = config.get('data_exp', '?')
        created = exp.get('created_at', '?')
        label = exp.get('label', '')
        
        latest_mark = " [latest]" if i == 0 else ""
        label_str = f" ({label})" if label else ""
        
        print(f"  [{i+1}] {exp['exp_id']}{latest_mark}")
        print(f"      num_topics={num_topics}, data_exp={data_exp[:20]}...{label_str}")
    
    print(f"\n  [0] Cancel")
    print(f"{'='*60}")
    
    while True:
        try:
            choice = input("Select (enter number, press Enter for latest): ").strip()
            if choice == '':
                choice = 1
            else:
                choice = int(choice)
            
            if choice == 0:
                return None
            if 1 <= choice <= len(experiments):
                selected = experiments[choice - 1]
                print(f"\nSelected: {selected['exp_id']}")
                return selected['path']
            print("Invalid choice, please try again")
        except ValueError:
            print("Please enter a number")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None


def print_experiments_json(experiments: List[Dict], pretty: bool = True):
    """Print experiments in JSON format (for API use)"""
    if pretty:
        print(json.dumps(experiments, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(experiments, ensure_ascii=False))


def print_experiments_table(experiments: List[Dict], exp_type: str = 'data'):
    """Print experiments in table format"""
    if not experiments:
        print("No experiments found")
        return
    
    print(f"\n{'='*80}")
    if exp_type == 'data':
        print(f"{'No.':<4} {'Experiment ID':<35} {'vocab_size':<12} {'Created':<20}")
    else:
        print(f"{'No.':<4} {'Experiment ID':<35} {'num_topics':<12} {'Created':<20}")
    print(f"{'='*80}")
    
    for i, exp in enumerate(experiments):
        config = exp.get('config', {})
        if exp_type == 'data':
            param = config.get('vocab_size', '?')
        else:
            param = config.get('num_topics', '?')
        created = exp.get('created_at', '?')[:19] if exp.get('created_at') else '?'
        
        print(f"{i+1:<4} {exp['exp_id']:<35} {str(param):<12} {created:<20}")
    
    print(f"{'='*80}")
    print(f"Total: {len(experiments)} experiments")


def main():
    parser = argparse.ArgumentParser(description='Experiment Manager')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--model-type', type=str, default='baseline', choices=['baseline', 'theta'])
    
    parser.add_argument('--list-data', action='store_true', help='List data experiments')
    parser.add_argument('--list-models', action='store_true', help='List model experiments')
    parser.add_argument('--select-data', action='store_true', help='Interactive data experiment selection')
    parser.add_argument('--select-model', action='store_true', help='Interactive model experiment selection')
    parser.add_argument('--find-data', type=str, help='Find data experiment (supports fuzzy matching)')
    parser.add_argument('--find-model', type=str, help='Find model experiment (supports fuzzy matching)')
    
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.list_data:
        if not args.dataset:
            print("Error: --list-data requires --dataset parameter")
            return
        experiments = manager.list_data_experiments(args.dataset, args.model_type)
        if args.json:
            print_experiments_json(experiments)
        else:
            print_experiments_table(experiments, 'data')
    
    elif args.list_models:
        if not args.dataset or not args.model:
            print("Error: --list-models requires --dataset and --model parameters")
            return
        experiments = manager.list_model_experiments(args.dataset, args.model, args.model_type)
        if args.json:
            print_experiments_json(experiments)
        else:
            print_experiments_table(experiments, 'model')
    
    elif args.select_data:
        if not args.dataset:
            print("Error: --select-data requires --dataset parameter")
            return
        result = interactive_select_data(manager, args.dataset, args.model_type)
        if result:
            print(f"\nOutput: {result}")
    
    elif args.select_model:
        if not args.dataset or not args.model:
            print("Error: --select-model requires --dataset and --model parameters")
            return
        result = interactive_select_model(manager, args.dataset, args.model, args.model_type)
        if result:
            print(f"\nOutput: {result}")
    
    elif args.find_data:
        if not args.dataset:
            print("Error: --find-data requires --dataset parameter")
            return
        result = manager.find_data_experiment(args.dataset, args.find_data, args.model_type)
        if result:
            print(result)
        else:
            print(f"No matching experiment found: {args.find_data}")
    
    elif args.find_model:
        if not args.dataset or not args.model:
            print("Error: --find-model requires --dataset and --model parameters")
            return
        result = manager.find_model_experiment(args.dataset, args.model, args.find_model, args.model_type)
        if result:
            print(result)
        else:
            print(f"No matching experiment found: {args.find_model}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

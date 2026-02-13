#!/usr/bin/env python
"""
THETA Topic Model Pipeline - Unified entry script

Supports running complete workflow for multiple topic models:
Data processing -> Model training -> Evaluation -> Visualization -> Result saving

Supported Models:
    THETA:     Main model using Qwen embeddings (0.6B/4B/8B)
    
    Traditional Baselines:
        LDA:       Latent Dirichlet Allocation (sklearn)
        HDP:       Hierarchical Dirichlet Process (auto topic number)
        STM:       Structural Topic Model (requires covariates, auto-skipped if none)
        BTM:       Biterm Topic Model (for short texts)
    
    Neural Baselines:
        ETM:       Embedded Topic Model (Word2Vec + VAE)
        CTM:       Contextualized Topic Model (SBERT + VAE)
        DTM:       Dynamic Topic Model (time-aware)
        NVDM:      Neural Variational Document Model
        GSM:       Gaussian Softmax Model
        ProdLDA:   Product of Experts LDA
        BERTopic:  BERT-based topic modeling (auto topic number)

Usage:
    # THETA model (requires model size and mode)
    python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --mode zero_shot
    python run_pipeline.py --dataset socialTwitter --models theta --model_size 4B --mode supervised
    
    # Traditional baseline models
    python run_pipeline.py --dataset socialTwitter --models lda
    python run_pipeline.py --dataset socialTwitter --models lda,hdp,stm,btm
    
    # Neural baseline models
    python run_pipeline.py --dataset socialTwitter --models etm,ctm,nvdm,gsm,prodlda
    
    # DTM model (requires timestamp data)
    python run_pipeline.py --dataset edu_data --models dtm
    
    # BERTopic (auto topic number)
    python run_pipeline.py --dataset socialTwitter --models bertopic
    
    # HDP (auto topic number)
    python run_pipeline.py --dataset socialTwitter --models hdp
    
    # Skip training, only evaluate and visualize
    python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --skip-train
    
    # Check if data files exist
    python run_pipeline.py --dataset socialTwitter --models theta --model_size 4B --check-only
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.sparse as sp
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from config import DATASET_CONFIGS, RESULT_DIR, DATA_DIR
from model.baselines.stm import CovariatesRequiredError

# Supported models and model sizes
# THETA: Main model using Qwen embeddings
# Baselines: Traditional (lda, hdp, stm, btm) and Neural (etm, ctm, dtm, nvdm, gsm, prodlda, bertopic)
ALL_MODELS = [
    'theta',      # THETA - Main model (Qwen embedding + VAE)
    'lda',        # LDA - Latent Dirichlet Allocation (sklearn)
    'hdp',        # HDP - Hierarchical Dirichlet Process (auto topic number)
    'stm',        # STM - Structural Topic Model (requires covariates, auto-skipped if none)
    'btm',        # BTM - Biterm Topic Model (for short texts)
    'etm',        # ETM - Embedded Topic Model (Word2Vec)
    'ctm',        # CTM - Contextualized Topic Model (SBERT)
    'dtm',        # DTM - Dynamic Topic Model (time-aware)
    'nvdm',       # NVDM - Neural Variational Document Model
    'gsm',        # GSM - Gaussian Softmax Model
    'prodlda',    # ProdLDA - Product of Experts LDA
    'bertopic',   # BERTopic - BERT-based topic modeling
]
MODEL_SIZES = ['0.6B', '4B', '8B']


def find_latest_data_exp(dataset: str, data_exp: str = None) -> str:
    """Find data experiment directory.
    
    Args:
        dataset: Dataset name
        data_exp: Specific experiment ID (if None, auto-select latest)
    
    Returns:
        Path to data experiment directory
    """
    data_base = Path(RESULT_DIR) / 'baseline' / dataset / 'data'
    
    if data_exp:
        # Use specified experiment
        exp_dir = data_base / data_exp
        if exp_dir.exists():
            return str(exp_dir)
        else:
            raise FileNotFoundError(f"Data experiment not found: {exp_dir}")
    
    # Auto-select latest experiment
    if not data_base.exists():
        raise FileNotFoundError(f"No data experiments found in: {data_base}")
    
    exp_dirs = sorted([d for d in data_base.iterdir() if d.is_dir() and d.name.startswith('exp_')])
    if not exp_dirs:
        raise FileNotFoundError(f"No data experiments found in: {data_base}")
    
    latest = exp_dirs[-1]  # Sorted by name, latest timestamp is last
    print(f"  [Auto] Using latest data experiment: {latest.name}")
    return str(latest)


def generate_model_exp_id(exp_name: str = None) -> str:
    """Generate experiment ID with timestamp.
    
    Args:
        exp_name: Optional experiment name tag
    
    Returns:
        Experiment ID like 'exp_20260205_171000' or 'exp_20260205_171000_k15'
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_id = f"exp_{timestamp}"
    if exp_name:
        exp_id = f"{exp_id}_{exp_name}"
    return exp_id


def parse_args():
    parser = argparse.ArgumentParser(
        description='ETM Topic Model Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--models', type=str, required=True,
                        help='Model list (comma-separated): theta,lda,hdp,stm,btm,etm,ctm,dtm,nvdm,gsm,prodlda,bertopic')
    parser.add_argument('--mode', type=str, default='zero_shot',
                        choices=['zero_shot', 'supervised', 'unsupervised'],
                        help='THETA mode (default: zero_shot)')
    parser.add_argument('--num_topics', type=int, default=20)
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512, help='Encoder hidden dimension (128-1024)')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate (0.00001-0.1)')
    parser.add_argument('--kl_start', type=float, default=0.0, help='KL annealing start weight (0-1)')
    parser.add_argument('--kl_end', type=float, default=1.0, help='KL annealing end weight (0-1)')
    parser.add_argument('--kl_warmup', type=int, default=50, help='KL warmup epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--no_early_stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-viz', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--language', type=str, default='en', choices=['en', 'zh'])
    parser.add_argument('--model_size', type=str, default='0.6B',
                        choices=MODEL_SIZES,
                        help='Qwen model size: 0.6B, 4B, 8B (THETA specific)')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check if data files exist, do not run')
    parser.add_argument('--prepare', action='store_true',
                        help='Preprocess data (generate embedding and BOW)')
    
    # Model-specific parameters
    parser.add_argument('--max_iter', type=int, default=100, help='Max iterations for LDA')
    parser.add_argument('--max_topics', type=int, default=150, help='Max topics for HDP')
    parser.add_argument('--n_iter', type=int, default=100, help='Gibbs sampling iterations for BTM')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha prior for HDP/BTM')
    parser.add_argument('--beta', type=float, default=0.01, help='Beta prior for BTM')
    parser.add_argument('--inference_type', type=str, default='zeroshot', 
                        choices=['zeroshot', 'combined'], help='CTM inference type')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for neural models')
    
    # Experiment management
    parser.add_argument('--data_exp', type=str, default=None,
                        help='Data experiment ID to use (default: auto-select latest)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name tag (appended to exp_id)')
    
    return parser.parse_args()


def get_model_list(models_str: str) -> List[str]:
    models = [m.strip().lower() for m in models_str.split(',')]
    for m in models:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model: {m}. Supported: {ALL_MODELS}")
    return models


def check_theta_data_files(dataset: str, model_size: str, mode: str, data_exp_dir: str = '') -> Dict[str, Any]:
    """Check if data files required for THETA model exist"""
    result_base = Path(RESULT_DIR) / model_size / dataset
    
    if data_exp_dir:
        # New exp structure: data/{exp_id}/embeddings/ and data/{exp_id}/bow/
        exp_path = Path(data_exp_dir)
        emb_candidates = [
            exp_path / 'embeddings' / 'embeddings.npy',
            exp_path / 'embeddings' / f'{dataset}_{mode}_embeddings.npy',
        ]
        bow_base = exp_path / 'bow'
    else:
        # Legacy structure: {mode}/embeddings/ and bow/
        emb_candidates = [
            result_base / mode / 'embeddings' / f'{dataset}_{mode}_embeddings.npy',
            result_base / mode / 'embeddings' / 'embeddings.npy',
        ]
        bow_base = result_base / 'bow'
    
    emb_path = emb_candidates[0]  # default (for error message)
    for candidate in emb_candidates:
        if candidate.exists():
            emb_path = candidate
            break
    
    # Also check 4B/8B format: {model_size}/{dataset}/embedding/
    if not emb_path.exists() and not data_exp_dir:
        emb_path_v2_dir = result_base / 'embedding'
        if emb_path_v2_dir.exists():
            for f in emb_path_v2_dir.glob(f'{mode}_embeddings_*.npy'):
                emb_path = f
                break
    
    # Files to check
    files_to_check = {
        'embeddings': emb_path,
        'bow_matrix': bow_base / 'bow_matrix.npy',
        'vocab': bow_base / 'vocab.txt',
        'vocab_embeddings': bow_base / 'vocab_embeddings.npy',
    }
    
    status = {'all_exist': True, 'files': {}}
    
    for name, path in files_to_check.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status['files'][name] = {
            'path': str(path),
            'exists': exists,
            'size_mb': round(size / 1024 / 1024, 2) if exists else 0
        }
        if not exists:
            status['all_exist'] = False
    
    return status


def check_baseline_data_files(dataset: str) -> Dict[str, Any]:
    """Check if data files required for Baseline model exist"""
    result_base = Path(RESULT_DIR) / 'baseline' / dataset
    data_path = Path(DATA_DIR) / dataset
    
    # Files to check
    files_to_check = {
        'raw_data': data_path / f'{dataset}_cleaned.csv',
    }
    
    # Check possible data filenames
    if not files_to_check['raw_data'].exists():
        for alt_name in ['cleaned.csv', 'data.csv', f'{dataset}.csv']:
            alt_path = data_path / alt_name
            if alt_path.exists():
                files_to_check['raw_data'] = alt_path
                break
    
    status = {'all_exist': True, 'files': {}}
    
    for name, path in files_to_check.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status['files'][name] = {
            'path': str(path),
            'exists': exists,
            'size_mb': round(size / 1024 / 1024, 2) if exists else 0
        }
        if not exists:
            status['all_exist'] = False
    
    return status


def print_data_check_result(model: str, status: Dict[str, Any]):
    """Print data check result"""
    print(f"\n[{model.upper()}] Data file check:")
    all_ok = status['all_exist']
    
    for name, info in status['files'].items():
        icon = '✓' if info['exists'] else '✗'
        size_str = f"({info['size_mb']} MB)" if info['exists'] else "(missing)"
        print(f"  {icon} {name}: {size_str}")
        if not info['exists']:
            print(f"      Path: {info['path']}")
    
    if all_ok:
        print(f"  -> All files ready, can run")
    else:
        print(f"  -> Missing required files, cannot run")
    
    return all_ok


def _is_complete_theta_exp(exp_dir: Path) -> bool:
    """Check if a THETA data experiment has all required files."""
    return (
        (exp_dir / 'bow' / 'bow_matrix.npy').exists() and
        (exp_dir / 'embeddings' / 'embeddings.npy').exists()
    )


def find_theta_data_exp(dataset: str, model_size: str, data_exp: str = '') -> str:
    """Find THETA data experiment directory.
    
    Args:
        dataset: Dataset name
        model_size: Model size (0.6B, 4B, 8B)
        data_exp: Data experiment ID, 'select' for interactive, '' for latest/legacy
    
    Returns:
        Path to data experiment directory, or empty string for legacy mode
    """
    data_base = Path(RESULT_DIR) / model_size / dataset / 'data'
    
    # Check if new exp structure exists
    if data_base.exists():
        exp_dirs = sorted(data_base.glob('exp_*'), key=lambda p: p.stat().st_mtime, reverse=True)
        if exp_dirs:
            if data_exp and data_exp != 'select':
                # Explicit data_exp specified — use it even if incomplete
                for d in exp_dirs:
                    if d.name == data_exp:
                        return str(d)
                # Fuzzy match
                for d in exp_dirs:
                    if data_exp in d.name:
                        return str(d)
                print(f"[Warning] data_exp '{data_exp}' not found, falling back")
            else:
                # Auto-select: only use complete experiments
                for d in exp_dirs:
                    if _is_complete_theta_exp(d):
                        return str(d)
                # No complete exp found — fall through to legacy
    
    # Legacy mode: no exp structure or no complete experiments
    return ''


def run_theta(args) -> Dict[str, Any]:
    """THETA model workflow"""
    print(f"\n{'='*70}")
    print(f"[THETA] Dataset: {args.dataset}, Model: {args.model_size}, Mode: {args.mode}")
    print(f"{'='*70}")
    
    result = {'model': 'theta', 'dataset': args.dataset, 'mode': args.mode, 'model_size': args.model_size}
    
    # Resolve data experiment
    data_exp_dir = find_theta_data_exp(args.dataset, args.model_size, getattr(args, 'data_exp', ''))
    if data_exp_dir:
        data_exp_id = Path(data_exp_dir).name
        print(f"  Data experiment: {data_exp_id}")
        print(f"  Data directory:  {data_exp_dir}")
    
    # Check data files (use exp dir if available)
    status = check_theta_data_files(args.dataset, args.model_size, args.mode, data_exp_dir)
    if not print_data_check_result(f'theta-{args.model_size}', status):
        result['train_status'] = 'data_missing'
        return result
    
    # Generate training experiment ID
    train_exp_id = generate_model_exp_id(args.exp_name)
    train_exp_dir = Path(RESULT_DIR) / args.model_size / args.dataset / 'models' / train_exp_id
    train_exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Train experiment: {train_exp_id}")
    print(f"  Output directory: {train_exp_dir}")
    
    # Save training config
    train_config = {
        'exp_id': train_exp_id,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': args.dataset,
        'model_size': args.model_size,
        'mode': args.mode,
        'data_exp': Path(data_exp_dir).name if data_exp_dir else 'legacy',
        'num_topics': args.num_topics,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'kl_start': args.kl_start,
        'kl_end': args.kl_end,
        'kl_warmup': args.kl_warmup,
        'patience': args.patience,
    }
    with open(train_exp_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)
    
    # Call main.py pipeline command
    import subprocess
    cmd = [
        sys.executable, 'main.py', 'pipeline',
        '--dataset', args.dataset,
        '--mode', args.mode,
        '--model_size', args.model_size,
        '--num_topics', str(args.num_topics),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--hidden_dim', str(args.hidden_dim),
        '--learning_rate', str(args.learning_rate),
        '--kl_start', str(args.kl_start),
        '--kl_end', str(args.kl_end),
        '--kl_warmup', str(args.kl_warmup),
        '--patience', str(args.patience),
        '--language', args.language
    ]
    # Pass experiment IDs to main.py
    if data_exp_dir:
        cmd.extend(['--data_exp', Path(data_exp_dir).name])
    cmd.extend(['--train_exp', train_exp_id])
    
    if args.no_early_stopping:
        cmd.append('--no_early_stopping')
    if args.skip_viz:
        cmd.append('--skip_viz')
    if args.skip_eval:
        cmd.append('--skip_eval')
    
    if args.skip_train:
        print("  [SKIP] Training skipped")
        result['train_status'] = 'skipped'
    else:
        print("  Running THETA pipeline...")
        ret = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        result['train_status'] = 'completed' if ret.returncode == 0 else 'failed'
    
    result['train_exp'] = train_exp_id
    result['data_exp'] = Path(data_exp_dir).name if data_exp_dir else ''
    return result


def run_baseline(model_name: str, args) -> Dict[str, Any]:
    """Baseline model workflow (LDA/HDP/STM/BTM/ETM/CTM/DTM/NVDM/GSM/ProdLDA/BERTopic)"""
    print(f"\n{'='*70}")
    print(f"[{model_name.upper()}] Dataset: {args.dataset}")
    print(f"{'='*70}")
    
    result = {'model': model_name, 'dataset': args.dataset}
    
    from model.baseline_trainer import BaselineTrainer
    from evaluation.unified_evaluator import UnifiedEvaluator
    from visualization.run_visualization import run_baseline_visualization
    
    # Find data experiment directory
    try:
        data_exp_dir = find_latest_data_exp(args.dataset, args.data_exp)
        print(f"  Data experiment: {Path(data_exp_dir).name}")
    except FileNotFoundError as e:
        print(f"  [Error] {e}")
        print(f"  Please run data preprocessing first:")
        print(f"    bash scripts/03_prepare_data.sh --dataset {args.dataset} --model {model_name}")
        return {'model': model_name, 'error': str(e)}
    
    # New directory structure: baseline/{dataset}/models/{model}/{exp_id}/
    base_dir = Path(RESULT_DIR) / 'baseline' / args.dataset
    
    # If skip-train, find the latest existing experiment instead of creating new one
    if args.skip_train:
        existing_exps = sorted(
            (base_dir / 'models' / model_name).glob('exp_*'),
            key=lambda p: p.name, reverse=True
        ) if (base_dir / 'models' / model_name).exists() else []
        # Find the latest exp that has theta files (i.e. was actually trained)
        model_dir = None
        for exp_dir in existing_exps:
            if list(exp_dir.rglob('theta_k*.npy')):
                model_dir = exp_dir
                break
        if model_dir is None:
            print(f"  [Error] No existing trained experiment found for {model_name}")
            result['train_status'] = 'no_existing_exp'
            return result
        model_exp_id = model_dir.name
        print(f"  Using existing experiment: {model_exp_id}")
    else:
        # Generate model experiment ID
        model_exp_id = generate_model_exp_id(args.exp_name)
        model_dir = base_dir / 'models' / model_name / model_exp_id
        model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Model experiment: {model_exp_id}")
    print(f"  Output directory: {model_dir}")
    
    # Save config.json for this model experiment
    config = {
        'exp_id': model_exp_id,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'dataset': args.dataset,
        'data_exp': Path(data_exp_dir).name,
        'num_topics': args.num_topics,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
    }
    with open(model_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # === Training ===
    train_result = None
    if not args.skip_train:
        print(f"\n[Training {model_name.upper()}]")
        trainer = BaselineTrainer(
            dataset=args.dataset,
            num_topics=args.num_topics,
            data_exp_dir=data_exp_dir,  # Use data experiment directory
            output_dir=str(model_dir),   # Use model experiment directory
            data_dir=str(DATA_DIR),
        )
        # Load data from data_exp_dir (no need to regenerate)
        trainer.load_preprocessed_data()
        
        # Traditional models
        try:
            if model_name == 'lda':
                train_result = trainer.train_lda(max_iter=args.max_iter)
            elif model_name == 'hdp':
                train_result = trainer.train_hdp(max_topics=args.max_topics, alpha=args.alpha)
            elif model_name == 'stm':
                train_result = trainer.train_stm(max_iter=args.max_iter)
            elif model_name == 'btm':
                train_result = trainer.train_btm(n_iter=args.n_iter, alpha=args.alpha, beta=args.beta)
            # Neural models
            elif model_name == 'etm':
                train_result = trainer.train_etm(epochs=args.epochs, batch_size=args.batch_size, 
                                                 learning_rate=args.learning_rate, hidden_dim=args.hidden_dim)
            elif model_name == 'ctm':
                train_result = trainer.train_ctm(inference_type=args.inference_type, epochs=args.epochs, 
                                                 batch_size=args.batch_size, learning_rate=args.learning_rate)
            elif model_name == 'dtm':
                train_result = trainer.train_dtm(epochs=args.epochs, batch_size=args.batch_size,
                                                 learning_rate=args.learning_rate, hidden_dim=args.hidden_dim)
            elif model_name == 'nvdm':
                train_result = trainer.train_nvdm(epochs=args.epochs, batch_size=args.batch_size,
                                                  learning_rate=args.learning_rate, hidden_dim=args.hidden_dim)
            elif model_name == 'gsm':
                train_result = trainer.train_gsm(epochs=args.epochs, batch_size=args.batch_size,
                                                 learning_rate=args.learning_rate, hidden_dim=args.hidden_dim)
            elif model_name == 'prodlda':
                train_result = trainer.train_prodlda(epochs=args.epochs, batch_size=args.batch_size,
                                                     learning_rate=args.learning_rate, hidden_dim=args.hidden_dim)
            elif model_name == 'bertopic':
                train_result = trainer.train_bertopic()
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except CovariatesRequiredError as e:
            print(f"\n  [SKIP] {model_name.upper()}: {e}")
            result['train_status'] = 'skipped_no_covariates'
            result['skip_reason'] = str(e)
            return result
        
        result['train_status'] = 'completed'
        result['train_time'] = train_result.get('train_time', 0)
    else:
        print(f"  [Skip] Training")
        result['train_status'] = 'skipped'
    
    # === Evaluation ===
    if not args.skip_eval:
        print(f"\n[Evaluating {model_name.upper()}]")
        # Load BOW and vocab from data experiment directory
        bow_path = Path(data_exp_dir) / 'bow_matrix.npy'
        vocab_path = Path(data_exp_dir) / 'vocab.json'
        
        # Determine actual num_topics (HDP/BERTopic may have different actual topics)
        actual_num_topics = args.num_topics
        if model_name in ('hdp', 'bertopic'):
            if train_result and 'actual_num_topics' in train_result:
                actual_num_topics = train_result['actual_num_topics']
                print(f"  Using {model_name.upper()} actual topics: {actual_num_topics}")
            else:
                # If skip-train, try to find existing theta file to get actual K
                import glob
                search_dirs = [
                    str(model_dir / model_name / 'theta_k*.npy'),
                    str(model_dir / model_name / 'model' / 'theta_k*.npy'),
                ]
                theta_files = []
                for pattern in search_dirs:
                    theta_files = glob.glob(pattern)
                    if theta_files:
                        break
                if theta_files:
                    # Extract K from filename like theta_k50.npy
                    import re
                    match = re.search(r'theta_k(\d+)\.npy', theta_files[0])
                    if match:
                        actual_num_topics = int(match.group(1))
                        print(f"  Detected {model_name.upper()} topics from file: {actual_num_topics}")
        
        # Check multiple possible paths for theta/beta (different models save to different locations)
        # Path priority:
        # 1. model_dir/{model_name}/theta_k{K}.npy (standard)
        # 2. model_dir/{model_name}_zeroshot/theta_k{K}.npy (CTM zeroshot)
        # 3. model_dir/{model_name}_combined/theta_k{K}.npy (CTM combined)
        # 4. model_dir/theta_k{K}.npy (direct)
        # 5. model_dir/{model_name}/model/theta_k{K}.npy (HDP/neural)
        
        theta_path = model_dir / model_name / f'theta_k{actual_num_topics}.npy'
        beta_path = model_dir / model_name / f'beta_k{actual_num_topics}.npy'
        
        # CTM saves to ctm_zeroshot/ or ctm_combined/
        if not theta_path.exists() and model_name == 'ctm':
            for suffix in ['zeroshot', 'combined']:
                ctm_path = model_dir / f'ctm_{suffix}' / f'theta_k{actual_num_topics}.npy'
                if ctm_path.exists():
                    theta_path = ctm_path
                    beta_path = model_dir / f'ctm_{suffix}' / f'beta_k{actual_num_topics}.npy'
                    break
        
        # If not found, check directly in model_dir
        if not theta_path.exists():
            theta_path = model_dir / f'theta_k{actual_num_topics}.npy'
            beta_path = model_dir / f'beta_k{actual_num_topics}.npy'
        
        # If still not found, check model/ subdirectory (HDP/neural models)
        if not theta_path.exists():
            theta_path = model_dir / model_name / 'model' / f'theta_k{actual_num_topics}.npy'
            beta_path = model_dir / model_name / 'model' / f'beta_k{actual_num_topics}.npy'
        
        if all(p.exists() for p in [bow_path, vocab_path, theta_path, beta_path]):
            bow_matrix = np.load(bow_path)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            theta = np.load(theta_path)
            beta = np.load(beta_path)
            
            training_history = None
            history_path = model_dir / f'training_history_k{args.num_topics}.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    training_history = json.load(f)
            
            evaluator = UnifiedEvaluator(
                beta=beta, theta=theta, bow_matrix=bow_matrix, vocab=vocab,
                training_history=training_history,
                dataset=args.dataset, output_dir=str(model_dir), num_topics=actual_num_topics
            )
            metrics = evaluator.compute_all_metrics()
            evaluator.save_metrics()
            # Only generate plots if visualization is enabled
            if not args.skip_viz:
                evaluator.generate_training_plots()
                evaluator.generate_metrics_plots()
            
            result['eval_status'] = 'completed'
            result['metrics'] = {
                'td': metrics.get('topic_diversity_td'),
                'npmi': metrics.get('topic_coherence_npmi_avg'),
                'exclusivity': metrics.get('topic_exclusivity_avg'),
                'ppl': metrics.get('perplexity')
            }
        else:
            print(f"  [Skip] Files not found")
            result['eval_status'] = 'files_not_found'
    else:
        print(f"  [Skip] Evaluation")
        result['eval_status'] = 'skipped'
    
    # === Visualization ===
    if not args.skip_viz:
        print(f"\n[Visualizing {model_name.upper()}]")
        try:
            from visualization.visualization_generator import VisualizationGenerator
            
            # Determine actual num_topics for visualization (same logic as evaluation)
            viz_num_topics = args.num_topics
            if model_name in ('hdp', 'bertopic'):
                if train_result and 'actual_num_topics' in train_result:
                    viz_num_topics = train_result['actual_num_topics']
                else:
                    import glob as glob_mod
                    for search_pattern in [
                        str(model_dir / model_name / 'theta_k*.npy'),
                        str(model_dir / model_name / 'model' / 'theta_k*.npy'),
                        str(model_dir / f'{model_name}_zeroshot' / 'theta_k*.npy'),
                    ]:
                        found = glob_mod.glob(search_pattern)
                        if found:
                            import re as re_mod
                            m = re_mod.search(r'theta_k(\d+)\.npy', found[0])
                            if m:
                                viz_num_topics = int(m.group(1))
                            break
            
            # Search multiple possible paths for theta/beta (same as evaluation)
            theta_path = model_dir / model_name / f'theta_k{viz_num_topics}.npy'
            beta_path = model_dir / model_name / f'beta_k{viz_num_topics}.npy'
            
            # CTM saves to ctm_zeroshot/ or ctm_combined/
            if not theta_path.exists() and model_name == 'ctm':
                for suffix in ['zeroshot', 'combined']:
                    ctm_path = model_dir / f'ctm_{suffix}' / f'theta_k{viz_num_topics}.npy'
                    if ctm_path.exists():
                        theta_path = ctm_path
                        beta_path = model_dir / f'ctm_{suffix}' / f'beta_k{viz_num_topics}.npy'
                        break
            
            # Check directly in model_dir
            if not theta_path.exists():
                theta_path = model_dir / f'theta_k{viz_num_topics}.npy'
                beta_path = model_dir / f'beta_k{viz_num_topics}.npy'
            
            # Check model/ subdirectory (HDP/STM/neural models)
            if not theta_path.exists():
                theta_path = model_dir / model_name / 'model' / f'theta_k{viz_num_topics}.npy'
                beta_path = model_dir / model_name / 'model' / f'beta_k{viz_num_topics}.npy'
            
            # Find topic_words in same directory as theta
            topic_words_dir = theta_path.parent
            topic_words_path = topic_words_dir / f'topic_words_k{viz_num_topics}.json'
            
            if theta_path.exists() and beta_path.exists():
                theta = np.load(theta_path)
                beta = np.load(beta_path)
                
                # Sanitize: replace inf/nan with 0 to avoid distance matrix errors
                theta = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
                beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Load vocab from data experiment
                with open(Path(data_exp_dir) / 'vocab.json', 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                
                # Load or generate topic_words in the format expected by VisualizationGenerator:
                # [(topic_id, [(word, weight), ...]), ...]
                topic_words = None
                if topic_words_path.exists():
                    with open(topic_words_path, 'r', encoding='utf-8') as f:
                        raw_topic_words = json.load(f)
                    
                    if isinstance(raw_topic_words, dict):
                        # Convert dict format {"topic_0": ["word1", ...]} to list of tuples
                        topic_words = []
                        for k, v in sorted(raw_topic_words.items(), key=lambda x: int(x[0].split('_')[-1]) if '_' in x[0] else int(x[0])):
                            tid = int(k.split('_')[-1]) if '_' in k else int(k)
                            if isinstance(v, list) and len(v) > 0:
                                if isinstance(v[0], (list, tuple)):
                                    words_with_weights = [(w, float(s)) for w, s in v]
                                else:
                                    # Just word list, get weights from beta
                                    words_with_weights = []
                                    for word in v:
                                        if word in vocab:
                                            widx = vocab.index(word)
                                            weight = float(beta[tid, widx]) if tid < beta.shape[0] else 0.0
                                        else:
                                            weight = 0.0
                                        words_with_weights.append((word, weight))
                            else:
                                words_with_weights = []
                            topic_words.append((tid, words_with_weights))
                    elif isinstance(raw_topic_words, list):
                        topic_words = raw_topic_words
                
                # Fallback: generate topic_words from beta if not available
                if topic_words is None:
                    print(f"  Generating topic_words from beta matrix")
                    topic_words = []
                    for tid in range(beta.shape[0]):
                        top_indices = np.argsort(beta[tid])[-20:][::-1]
                        words_with_weights = [
                            (vocab[i] if i < len(vocab) else f'word_{i}', float(beta[tid, i]))
                            for i in top_indices
                        ]
                        topic_words.append((tid, words_with_weights))
                
                # Create visualization output directory
                viz_dir = model_dir / 'visualization'
                viz_dir.mkdir(parents=True, exist_ok=True)
                
                generator = VisualizationGenerator(
                    theta=theta,
                    beta=beta,
                    vocab=vocab,
                    topic_words=topic_words,
                    output_dir=str(viz_dir),
                    language=args.language,
                    dpi=300
                )
                generator.generate_all()
                result['viz_status'] = 'completed'
            else:
                print(f"  [Skip] theta/beta files not found")
                print(f"  Searched: {theta_path}")
                result['viz_status'] = 'files_not_found'
        except Exception as e:
            print(f"  [Error] {e}")
            import traceback; traceback.print_exc()
            result['viz_status'] = f'error: {str(e)}'
    else:
        print(f"  [Skip] Visualization")
        result['viz_status'] = 'skipped'
    
    return result


def print_summary(results: List[Dict]):
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        model = r.get('model', '?').upper()
        mode = f" ({r['mode']})" if 'mode' in r else ''
        print(f"\n{model}{mode} on {r.get('dataset', '?')}")
        print(f"  Train: {r.get('train_status', 'N/A')}")
        print(f"  Eval:  {r.get('eval_status', 'N/A')}")
        print(f"  Viz:   {r.get('viz_status', 'N/A')}")
        if 'metrics' in r and r['metrics']:
            m = r['metrics']
            if m.get('td'): print(f"  TD: {m['td']:.4f}")
            if m.get('npmi'): print(f"  NPMI: {m['npmi']:.4f}")
            if m.get('ppl'): print(f"  PPL: {m['ppl']:.2f}")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    models = get_model_list(args.models)
    
    print(f"{'='*70}")
    print(f"ETM Pipeline: {args.dataset}")
    print(f"Models: {', '.join(models)}")
    if 'theta' in models:
        print(f"Model Size: {args.model_size}")
        print(f"Mode: {args.mode}")
    print(f"{'='*70}")
    
    # Check data files only mode
    if args.check_only:
        print("\n[Check Mode] Only checking data files, not running training")
        for model_name in models:
            if model_name == 'theta':
                status = check_theta_data_files(args.dataset, args.model_size, args.mode)
                print_data_check_result(f'theta-{args.model_size}', status)
            else:
                status = check_baseline_data_files(args.dataset)
                print_data_check_result(model_name, status)
        return
    
    # Data preprocessing mode
    if args.prepare:
        print("\n[Preprocessing Mode] Generating embedding and BOW")
        import subprocess
        for model_name in models:
            if model_name == 'theta':
                cmd = [
                    sys.executable, 'prepare_data.py',
                    '--dataset', args.dataset,
                    '--model', 'theta',
                    '--model_size', args.model_size,
                    '--mode', args.mode,
                    '--vocab_size', str(DATASET_CONFIGS.get(args.dataset, {}).get('vocab_size', 5000)),
                    '--batch_size', str(args.batch_size),
                    '--gpu', str(args.gpu)
                ]
            else:
                cmd = [
                    sys.executable, 'prepare_data.py',
                    '--dataset', args.dataset,
                    '--model', 'baseline',
                    '--vocab_size', str(DATASET_CONFIGS.get(args.dataset, {}).get('vocab_size', 5000)),
                    '--batch_size', str(args.batch_size),
                    '--gpu', str(args.gpu)
                ]
            subprocess.run(cmd, cwd=str(Path(__file__).parent))
        return
    
    results = []
    for model_name in models:
        try:
            if model_name == 'theta':
                result = run_theta(args)
            else:
                result = run_baseline(model_name, args)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'model': model_name, 'status': f'error: {e}'})
    
    print_summary(results)


if __name__ == '__main__':
    main()

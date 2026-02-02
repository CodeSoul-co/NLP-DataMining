#!/usr/bin/env python
"""
ETM Topic Model Pipeline - Unified entry script

Supports running complete workflow for multiple topic models (THETA, LDA, ETM, CTM, DTM):
Data processing -> Model training -> Evaluation -> Visualization -> Result saving

Usage:
    # THETA model (requires model size and mode)
    python run_pipeline.py --dataset socialTwitter --models theta --model_size 0.6B --mode zero_shot
    python run_pipeline.py --dataset socialTwitter --models theta --model_size 4B --mode supervised
    
    # Baseline models (no model_size needed)
    python run_pipeline.py --dataset socialTwitter --models lda
    python run_pipeline.py --dataset socialTwitter --models lda,etm,ctm
    
    # DTM model (requires timestamp data)
    python run_pipeline.py --dataset edu_data --models dtm
    
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

# Supported models and model sizes
ALL_MODELS = ['theta', 'lda', 'etm', 'ctm', 'dtm']
MODEL_SIZES = ['0.6B', '4B', '8B']


def parse_args():
    parser = argparse.ArgumentParser(
        description='ETM Topic Model Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--models', type=str, required=True,
                        help='Model list: theta,lda,etm,ctm')
    parser.add_argument('--mode', type=str, default='zero_shot',
                        choices=['zero_shot', 'supervised', 'unsupervised'],
                        help='THETA mode (default: zero_shot)')
    parser.add_argument('--num_topics', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
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
    return parser.parse_args()


def get_model_list(models_str: str) -> List[str]:
    models = [m.strip().lower() for m in models_str.split(',')]
    for m in models:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model: {m}. Supported: {ALL_MODELS}")
    return models


def check_theta_data_files(dataset: str, model_size: str, mode: str) -> Dict[str, Any]:
    """Check if data files required for THETA model exist"""
    result_base = Path(RESULT_DIR) / model_size / dataset
    
    # 0.6B structure: {model_size}/{dataset}/{mode}/embeddings/
    # 4B/8B structure: {model_size}/{dataset}/embedding/ (with timestamp)
    
    # Check embeddings - supports two directory structures
    emb_path_v1 = result_base / mode / 'embeddings' / f'{dataset}_{mode}_embeddings.npy'
    emb_path_v2_dir = result_base / 'embedding'
    
    emb_path = emb_path_v1
    if not emb_path_v1.exists() and emb_path_v2_dir.exists():
        # Find 4B/8B format embedding file
        for f in emb_path_v2_dir.glob(f'{mode}_embeddings_*.npy'):
            emb_path = f
            break
    
    # Files to check
    files_to_check = {
        'embeddings': emb_path,
        'bow_matrix': result_base / 'bow' / 'bow_matrix.npz',
        'vocab': result_base / 'bow' / 'vocab.txt',
        'vocab_embeddings': result_base / 'bow' / 'vocab_embeddings.npy',
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


def run_theta(args) -> Dict[str, Any]:
    """THETA model workflow"""
    print(f"\n{'='*70}")
    print(f"[THETA] Dataset: {args.dataset}, Model: {args.model_size}, Mode: {args.mode}")
    print(f"{'='*70}")
    
    result = {'model': 'theta', 'dataset': args.dataset, 'mode': args.mode, 'model_size': args.model_size}
    
    # Check data files
    status = check_theta_data_files(args.dataset, args.model_size, args.mode)
    if not print_data_check_result(f'theta-{args.model_size}', status):
        result['train_status'] = 'data_missing'
        return result
    
    # Call main.py pipeline command
    import subprocess
    cmd = [
        sys.executable, 'main.py', 'pipeline',
        '--dataset', args.dataset,
        '--mode', args.mode,
        '--model_size', args.model_size,
        '--num_topics', str(args.num_topics),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size)
    ]
    
    if args.skip_train:
        print("  [SKIP] Training skipped")
        result['train_status'] = 'skipped'
    else:
        print("  Running THETA pipeline...")
        ret = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        result['train_status'] = 'completed' if ret.returncode == 0 else 'failed'
    
    return result


def run_baseline(model_name: str, args) -> Dict[str, Any]:
    """Baseline model workflow (LDA/ETM/CTM/DTM)"""
    print(f"\n{'='*70}")
    print(f"[{model_name.upper()}] Dataset: {args.dataset}")
    print(f"{'='*70}")
    
    result = {'model': model_name, 'dataset': args.dataset}
    
    from model.baseline_trainer import BaselineTrainer
    from evaluation.unified_evaluator import UnifiedEvaluator
    from visualization.run_visualization import run_baseline_visualization
    
    dataset_config = DATASET_CONFIGS.get(args.dataset, {})
    vocab_size = dataset_config.get('vocab_size', 5000)
    
    result_dir = Path(RESULT_DIR) / 'baseline' / args.dataset
    model_dir = result_dir / ('ctm_zeroshot' if model_name == 'ctm' else model_name)
    
    # === Training ===
    if not args.skip_train:
        print(f"\n[Training {model_name.upper()}]")
        trainer = BaselineTrainer(
            dataset=args.dataset,
            num_topics=args.num_topics,
            vocab_size=vocab_size,
            data_dir=str(DATA_DIR),
            result_dir=str(Path(RESULT_DIR) / 'baseline')
        )
        # DTM and CTM need SBERT embedding
        trainer.prepare_data(generate_sbert=(model_name in ['ctm', 'dtm']))
        
        if model_name == 'lda':
            train_result = trainer.train_lda(max_iter=100)
        elif model_name == 'etm':
            train_result = trainer.train_etm(epochs=args.epochs, batch_size=args.batch_size)
        elif model_name == 'ctm':
            train_result = trainer.train_ctm(inference_type='zeroshot', epochs=args.epochs, batch_size=args.batch_size)
        elif model_name == 'dtm':
            train_result = trainer.train_dtm(epochs=args.epochs, batch_size=args.batch_size)
        
        result['train_status'] = 'completed'
        result['train_time'] = train_result.get('train_time', 0)
    else:
        print(f"  [Skip] Training")
        result['train_status'] = 'skipped'
    
    # === Evaluation ===
    if not args.skip_eval:
        print(f"\n[Evaluating {model_name.upper()}]")
        bow_path = result_dir / 'bow_matrix.npz'
        vocab_path = result_dir / 'vocab.json'
        theta_path = model_dir / f'theta_k{args.num_topics}.npy'
        beta_path = model_dir / f'beta_k{args.num_topics}.npy'
        
        if all(p.exists() for p in [bow_path, vocab_path, theta_path, beta_path]):
            bow_matrix = sp.load_npz(bow_path)
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
                training_history=training_history, model_name=model_name,
                dataset=args.dataset, output_dir=str(model_dir), num_topics=args.num_topics
            )
            metrics = evaluator.compute_all_metrics()
            evaluator.save_metrics()
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
            run_baseline_visualization(
                result_dir=str(Path(RESULT_DIR) / 'baseline'),
                dataset=args.dataset,
                model='ctm_zeroshot' if model_name == 'ctm' else model_name,
                num_topics=args.num_topics,
                language=args.language
            )
            result['viz_status'] = 'completed'
        except Exception as e:
            print(f"  [Error] {e}")
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

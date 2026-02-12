"""
Engine B: Qwen3-Embedding Training Pipeline

Controls all hyperparameters and supports --dev mode for debugging.

Usage:
    # Zero-shot embedding (no training)
    python main.py --mode zero_shot --dataset hatespeech
    python main.py --mode zero_shot --dataset all
    
    # Unsupervised training (SimCSE)
    python main.py --mode unsupervised --dataset germanCoal --epochs 3
    python main.py --mode unsupervised --dataset FCPB --epochs 3
    
    # Supervised training (with labels)
    python main.py --mode supervised --dataset hatespeech --epochs 3
    python main.py --mode supervised --dataset mental_health --epochs 3
    
    # With debug mode
    python main.py --mode supervised --dataset hatespeech --dev
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DatasetLoader, get_dataset_summary
from embedder import QwenEmbedder, EmbeddingManager


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Qwen3-Embedding Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "supervised", "unsupervised", "joint_unsupervised", "joint_supervised", "generate", "generate_vocab_embeddings"],
        help="Training mode: zero_shot, supervised, unsupervised (per-dataset), "
             "joint_unsupervised, joint_supervised (all datasets together), "
             "generate (use trained model to generate embeddings), "
             "generate_vocab_embeddings (generate embeddings for vocabulary words)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="hatespeech",
        help="Dataset name or 'all' for all datasets"
    )
    
    # Model settings
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/qwen3_embedding_0.6B",
        help="Path to Qwen3-Embedding model"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for embedding generation (smaller = less GPU memory, default 16 for RTX 4090)"
    )
    
    # Data settings
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (None for all)"
    )
    
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle data before processing"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/embedding/outputs",
        help="Output directory for embeddings"
    )
    
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/root/autodl-tmp/result",
        help="Result directory (versioned, no overwrite)"
    )
    
    parser.add_argument(
        "--model_size",
        type=str,
        default="0.6B",
        choices=["0.6B", "4B", "8B"],
        help="Qwen model size for result directory structure"
    )
    
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="",
        help="Experiment directory for saving embeddings (overrides default result path)"
    )
    
    # Embedding settings
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2 normalize embeddings"
    )
    
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable L2 normalization"
    )
    
    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, None for auto)"
    )
    
    # Debug mode
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode with debug prints and dimension info"
    )
    
    # Validation
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate embeddings after generation"
    )
    
    # Training settings (for supervised/unsupervised modes)
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning"
    )
    
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA, use full fine-tuning"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for generate mode"
    )
    
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Override label column for supervised training (e.g. --label_column province)"
    )
    
    parser.add_argument(
        "--vocab_file",
        type=str,
        default=None,
        help="Path to vocabulary JSON file (for generate_vocab_embeddings mode)"
    )
    
    return parser.parse_args()


def print_header(args):
    """Print run header with configuration"""
    print("=" * 70)
    print("Qwen3-Embedding Training Pipeline")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Dev mode: {args.dev}")
    print("-" * 70)
    print("Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Max length: {args.max_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Normalize: {not args.no_normalize}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Result dir: {args.result_dir}")
    print("=" * 70)


def run_zero_shot(args):
    """Run zero-shot embedding generation"""
    print("\n[Step 1/4] Initializing data loader...")
    loader = DatasetLoader(dev_mode=args.dev)
    
    # Determine which datasets to process
    if args.dataset.lower() == "all":
        dataset_names = loader.get_available_datasets()
    else:
        dataset_names = [args.dataset]
    
    print(f"Datasets to process: {dataset_names}")
    
    print("\n[Step 2/4] Initializing embedder...")
    embedder = QwenEmbedder(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        dev_mode=args.dev
    )
    
    print("\n[Step 3/4] Initializing embedding manager...")
    manager = EmbeddingManager(
        output_dir=args.output_dir,
        result_dir=args.result_dir,
        model_size=args.model_size,
        exp_dir=args.exp_dir,
        dev_mode=args.dev
    )
    
    print("\n[Step 4/4] Processing datasets...")
    results = {}
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"\n{'='*70}")
        print(f"Processing [{i}/{len(dataset_names)}]: {dataset_name}")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Load dataset
            print(f"\nLoading dataset: {dataset_name}")
            texts, labels, info = loader.load_dataset(
                dataset_name,
                max_samples=args.max_samples,
                shuffle=args.shuffle,
                random_seed=args.seed
            )
            
            print(f"  Samples: {len(texts)}")
            print(f"  Has labels: {info.has_label}")
            print(f"  Language: {info.language}")
            
            if args.dev:
                print(f"[DEV] Text lengths (chars): min={min(len(t) for t in texts)}, "
                      f"max={max(len(t) for t in texts)}, "
                      f"avg={sum(len(t) for t in texts)/len(texts):.1f}")
            
            # Generate embeddings
            print(f"\nGenerating zero-shot embeddings...")
            output = embedder.zero_shot_embed(
                texts=texts,
                labels=labels,
                dataset_name=dataset_name,
                show_progress=True
            )
            
            if args.dev:
                print(f"[DEV] Embedding matrix shape: {output.embeddings.shape}")
                print(f"[DEV] Embedding matrix dtype: {output.embeddings.dtype}")
                print(f"[DEV] Memory usage: {output.embeddings.nbytes / 1024 / 1024:.2f} MB")
            
            # Validate embeddings
            if args.validate:
                print(f"\nValidating embeddings...")
                stats = embedder.validate_embeddings(output.embeddings)
                print(f"  Shape: {stats['shape']}")
                print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                print(f"  Norm mean: {stats['norm_mean']:.4f}")
                print(f"  Has NaN: {stats['has_nan']}, Has Inf: {stats['has_inf']}")
                
                if stats['has_nan'] or stats['has_inf']:
                    print("  [WARNING] Embeddings contain NaN or Inf values!")
            
            # Save embeddings
            print(f"\nSaving embeddings...")
            saved_paths = manager.save_embeddings(output, mode_subdir="zero_shot")
            print(f"  Embeddings: {saved_paths['embeddings']}")
            print(f"  Result (versioned): {saved_paths['result_embeddings']}")
            
            elapsed = time.time() - start_time
            print(f"\nCompleted {dataset_name} in {elapsed:.1f}s")
            
            results[dataset_name] = {
                "status": "success",
                "num_samples": len(texts),
                "embedding_shape": output.embeddings.shape,
                "elapsed_time": elapsed,
                "paths": saved_paths
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[ERROR] Failed to process {dataset_name}: {e}")
            results[dataset_name] = {
                "status": "failed",
                "error": str(e),
                "elapsed_time": elapsed
            }
            if args.dev:
                import traceback
                traceback.print_exc()
    
    return results


def run_supervised(args):
    """Run supervised LoRA training (for labeled datasets)"""
    from trainer import EmbeddingTrainerV2 as EmbeddingTrainer, TrainingConfig
    import numpy as np
    
    print("\n[Step 1/5] Initializing data loader...")
    loader = DatasetLoader(dev_mode=args.dev)
    
    # Determine datasets - only labeled ones for supervised
    labeled_datasets = loader.get_labeled_datasets()
    
    if args.dataset.lower() == "all":
        dataset_names = labeled_datasets
    else:
        if args.dataset not in labeled_datasets:
            print(f"[WARNING] {args.dataset} is not a labeled dataset.")
            print(f"Labeled datasets: {labeled_datasets}")
            print("Proceeding anyway...")
        dataset_names = [args.dataset]
    
    print(f"Datasets to process: {dataset_names}")
    
    print("\n[Step 2/5] Initializing trainer...")
    use_lora = args.use_lora and not args.no_lora
    
    config = TrainingConfig(
        model_path=args.model_path,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=use_lora,
        output_dir=args.output_dir,
        result_dir=args.result_dir,
        model_size=args.model_size,
        exp_dir=args.exp_dir
    )
    
    trainer = EmbeddingTrainer(config=config, dev_mode=args.dev)
    
    print("\n[Step 3/5] Processing datasets...")
    results = {}
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"\n{'='*70}")
        print(f"Processing [{i}/{len(dataset_names)}]: {dataset_name}")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Load dataset
            texts, labels, info = loader.load_dataset(
                dataset_name,
                max_samples=args.max_samples,
                shuffle=args.shuffle,
                random_seed=args.seed
            )
            
            # If no built-in labels, try --label_column override or interactive selection
            if labels is None:
                import pandas as pd
                config_entry = loader.DATASET_CONFIGS[dataset_name]
                full_path = os.path.join(loader.base_path, config_entry["file_path"])
                df = pd.read_csv(full_path, encoding='utf-8')
                text_col = config_entry["text_column"]
                
                # Find candidate label columns (categorical, not text, 2-500 unique values)
                candidate_cols = []
                for col in df.columns:
                    if col == text_col:
                        continue
                    nunique = df[col].nunique()
                    if 2 <= nunique <= 500:
                        candidate_cols.append((col, nunique))
                
                label_column = getattr(args, 'label_column', None)
                
                if label_column and label_column != 'select':
                    # Direct specification
                    if label_column in df.columns:
                        labels = df[label_column].values
                        if args.max_samples and args.max_samples < len(labels):
                            labels = labels[:args.max_samples]
                        print(f"  Using --label_column '{label_column}' ({len(np.unique(labels))} classes)")
                    else:
                        print(f"[ERROR] Column '{label_column}' not found. Available: {df.columns.tolist()}")
                        continue
                elif candidate_cols:
                    # Interactive selection or auto-select
                    if label_column == 'select' or len(candidate_cols) > 1:
                        print(f"\n  Available label columns for {dataset_name}:")
                        print(f"  {'No.':<5} {'Column':<20} {'Unique Values':<15} {'Sample'}")
                        print(f"  {'-'*60}")
                        for idx, (col, nunique) in enumerate(candidate_cols, 1):
                            sample = str(df[col].iloc[0])[:30]
                            print(f"  {idx:<5} {col:<20} {nunique:<15} {sample}")
                        print()
                        try:
                            choice = input(f"  Select label column [1-{len(candidate_cols)}]: ").strip()
                            choice_idx = int(choice) - 1
                            if 0 <= choice_idx < len(candidate_cols):
                                chosen_col = candidate_cols[choice_idx][0]
                            else:
                                print(f"[ERROR] Invalid choice: {choice}")
                                continue
                        except (ValueError, EOFError):
                            print(f"[ERROR] Invalid input, skipping {dataset_name}")
                            continue
                    else:
                        chosen_col = candidate_cols[0][0]
                    
                    labels = df[chosen_col].values
                    if args.max_samples and args.max_samples < len(labels):
                        labels = labels[:args.max_samples]
                    print(f"  Using label column '{chosen_col}' ({len(np.unique(labels))} classes)")
                else:
                    print(f"[ERROR] {dataset_name} has no suitable label columns.")
                    print(f"  Columns: {df.columns.tolist()}")
                    print(f"  Use --label_column <col> to specify one.")
                    continue
            
            # Convert labels to numeric if needed
            if not np.issubdtype(labels.dtype, np.number):
                unique_labels = np.unique(labels)
                label_map = {l: i for i, l in enumerate(unique_labels)}
                labels = np.array([label_map[l] for l in labels])
            
            print(f"  Samples: {len(texts)}")
            print(f"  Unique labels: {len(np.unique(labels))}")
            
            # Train
            train_result = trainer.train_supervised(texts, labels, dataset_name)
            
            # Generate embeddings with trained model
            print("\nGenerating embeddings with trained model...")
            embeddings = trainer.generate_embeddings(texts)
            
            # Save embeddings
            saved_paths = trainer.save_embeddings(
                embeddings, dataset_name, "supervised", labels
            )
            
            elapsed = time.time() - start_time
            print(f"\nCompleted {dataset_name} in {elapsed:.1f}s")
            
            results[dataset_name] = {
                "status": "success",
                "num_samples": len(texts),
                "embedding_shape": embeddings.shape,
                "final_loss": train_result['final_loss'],
                "elapsed_time": elapsed,
                "paths": saved_paths
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[ERROR] Failed to process {dataset_name}: {e}")
            results[dataset_name] = {
                "status": "failed",
                "error": str(e),
                "elapsed_time": elapsed
            }
            if args.dev:
                import traceback
                traceback.print_exc()
    
    return results


def run_unsupervised(args):
    """Run unsupervised SimCSE training (for all datasets)"""
    from trainer import EmbeddingTrainerV2 as EmbeddingTrainer, TrainingConfig
    
    print("\n[Step 1/5] Initializing data loader...")
    loader = DatasetLoader(dev_mode=args.dev)
    
    # Determine datasets
    if args.dataset.lower() == "all":
        dataset_names = loader.get_available_datasets()
    else:
        dataset_names = [args.dataset]
    
    print(f"Datasets to process: {dataset_names}")
    
    print("\n[Step 2/5] Initializing trainer...")
    use_lora = args.use_lora and not args.no_lora
    
    config = TrainingConfig(
        model_path=args.model_path,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=use_lora,
        output_dir=args.output_dir,
        result_dir=args.result_dir,
        model_size=args.model_size,
        exp_dir=args.exp_dir
    )
    
    trainer = EmbeddingTrainer(config=config, dev_mode=args.dev)
    
    print("\n[Step 3/5] Processing datasets...")
    results = {}
    
    for i, dataset_name in enumerate(dataset_names, 1):
        print(f"\n{'='*70}")
        print(f"Processing [{i}/{len(dataset_names)}]: {dataset_name}")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Load dataset
            texts, labels, info = loader.load_dataset(
                dataset_name,
                max_samples=args.max_samples,
                shuffle=args.shuffle,
                random_seed=args.seed
            )
            
            print(f"  Samples: {len(texts)}")
            
            # Train
            train_result = trainer.train_unsupervised(texts, dataset_name)
            
            # Generate embeddings with trained model
            print("\nGenerating embeddings with trained model...")
            embeddings = trainer.generate_embeddings(texts)
            
            # Save embeddings
            saved_paths = trainer.save_embeddings(
                embeddings, dataset_name, "unsupervised", labels
            )
            
            elapsed = time.time() - start_time
            print(f"\nCompleted {dataset_name} in {elapsed:.1f}s")
            
            results[dataset_name] = {
                "status": "success",
                "num_samples": len(texts),
                "embedding_shape": embeddings.shape,
                "final_loss": train_result['final_loss'],
                "elapsed_time": elapsed,
                "paths": saved_paths
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[ERROR] Failed to process {dataset_name}: {e}")
            results[dataset_name] = {
                "status": "failed",
                "error": str(e),
                "elapsed_time": elapsed
            }
            if args.dev:
                import traceback
                traceback.print_exc()
    
    return results


def run_joint_unsupervised(args):
    """
    Run joint unsupervised training on ALL datasets together.
    This trains a single shared embedding function using all datasets.
    """
    from trainer import EmbeddingTrainerV2 as EmbeddingTrainer, TrainingConfig
    import numpy as np
    
    print("\n" + "=" * 70)
    print("JOINT UNSUPERVISED TRAINING")
    print("Training shared embedding on ALL datasets")
    print("=" * 70)
    
    print("\n[Step 1/5] Loading all datasets...")
    loader = DatasetLoader(dev_mode=args.dev)
    dataset_names = loader.get_available_datasets()
    
    # First, collect all datasets
    dataset_info = {}
    for dataset_name in dataset_names:
        texts, labels, info = loader.load_dataset(
            dataset_name,
            max_samples=args.max_samples,
            shuffle=args.shuffle,
            random_seed=args.seed
        )
        # Convert to list
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        else:
            texts = list(texts)
        
        dataset_info[dataset_name] = {
            'texts': texts,
            'labels': labels,
            'count': len(texts)
        }
        print(f"  {dataset_name}: {len(texts):,} samples")
    
    # Data balancing function
    def oversample_small_datasets(dataset_info, target_size=100000):
        """Oversample small datasets to avoid data imbalance"""
        print(f"\n[Data Balancing] Target size: {target_size:,}")
        balanced_info = {}
        
        for dataset_name, info in dataset_info.items():
            texts = info['texts']
            labels = info.get('labels', [])
            original_count = len(texts)
            
            if original_count < target_size:
                # Oversample: repeat small datasets
                repeat_times = int(np.ceil(target_size / original_count))
                balanced_texts = (texts * repeat_times)[:target_size]
                balanced_labels = (labels * repeat_times)[:target_size] if labels is not None else []
                print(f"  {dataset_name}: {original_count:,} → {target_size:,} (repeat {repeat_times}x)")
            else:
                # Keep as is or downsample
                balanced_texts = texts[:target_size]
                balanced_labels = labels[:target_size] if labels is not None else []
                print(f"  {dataset_name}: {original_count:,} → {target_size:,}")
            
            balanced_info[dataset_name] = {
                'texts': balanced_texts,
                'labels': balanced_labels,
                'count': len(balanced_texts),
                'start_idx': 0  # Will be updated later
            }
        
        return balanced_info
    
    # Apply data balancing
    print("\n[Step 2/5] Balancing datasets...")
    dataset_info = oversample_small_datasets(dataset_info, target_size=100000)
    
    # Merge all datasets
    all_texts = []
    current_idx = 0
    
    for dataset_name, info in dataset_info.items():
        info['start_idx'] = current_idx
        all_texts.extend(info['texts'])
        current_idx += info['count']
    
    print(f"\nTotal samples after balancing: {len(all_texts):,}")
    
    print("\n[Step 2/5] Initializing trainer...")
    use_lora = args.use_lora and not args.no_lora
    
    config = TrainingConfig(
        model_path=args.model_path,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=use_lora,
        output_dir=args.output_dir,
        result_dir=args.result_dir,
        model_size=args.model_size,
        exp_dir=args.exp_dir
    )
    
    trainer = EmbeddingTrainer(config=config, dev_mode=args.dev)
    
    print("\n[Step 3/5] Training on joint dataset...")
    start_time = time.time()
    
    train_result = trainer.train_unsupervised(
        np.array(all_texts), 
        dataset_name="joint_all"
    )
    
    training_time = time.time() - start_time
    print(f"\nJoint training completed in {training_time:.1f}s")
    
    print("\n[Step 4/5] Generating embeddings for each dataset...")
    results = {}
    
    # Generate embeddings for all texts at once (pass as list for tokenizer)
    all_embeddings = trainer.generate_embeddings(all_texts)
    
    print("\n[Step 5/5] Saving embeddings per dataset...")
    for dataset_name, info in dataset_info.items():
        start_idx = info['start_idx']
        count = info['count']
        labels = info['labels']
        
        # Extract embeddings for this dataset
        embeddings = all_embeddings[start_idx:start_idx + count]
        
        # Save embeddings
        saved_paths = trainer.save_embeddings(
            embeddings, dataset_name, "joint_unsupervised", labels
        )
        
        results[dataset_name] = {
            "status": "success",
            "num_samples": count,
            "embedding_shape": embeddings.shape,
            "final_loss": train_result['final_loss'],
            "elapsed_time": training_time / len(dataset_names),
            "paths": saved_paths
        }
        print(f"  Saved {dataset_name}: {embeddings.shape}")
    
    return results


def run_joint_supervised(args):
    """
    Run joint supervised training on ALL labeled datasets together.
    This trains a single shared embedding function using all labeled datasets.
    """
    from trainer import EmbeddingTrainerV2 as EmbeddingTrainer, TrainingConfig
    import numpy as np
    
    print("\n" + "=" * 70)
    print("JOINT SUPERVISED TRAINING")
    print("Training shared embedding on ALL labeled datasets")
    print("=" * 70)
    
    print("\n[Step 1/5] Loading all labeled datasets...")
    loader = DatasetLoader(dev_mode=args.dev)
    labeled_datasets = loader.get_labeled_datasets()
    
    label_offset = 0
    dataset_info = {}
    
    for dataset_name in labeled_datasets:
        texts, labels, info = loader.load_dataset(
            dataset_name,
            max_samples=args.max_samples,
            shuffle=args.shuffle,
            random_seed=args.seed
        )
        
        if labels is None:
            print(f"  {dataset_name}: skipped (no labels)")
            continue
        
        # Convert to list
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        else:
            texts = list(texts)
        
        # Convert labels to numeric if needed
        if not np.issubdtype(labels.dtype, np.number):
            unique_labels = np.unique(labels)
            label_map = {l: i for i, l in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])
        
        # Offset labels to make them unique across datasets
        labels = labels + label_offset
        label_offset = labels.max() + 1
        
        dataset_info[dataset_name] = {
            'texts': texts,
            'labels': labels,
            'count': len(texts)
        }
        print(f"  {dataset_name}: {len(texts):,} samples, {len(np.unique(labels))} classes")
    
    # Data balancing function (same as unsupervised)
    def oversample_small_datasets(dataset_info, target_size=100000):
        """Oversample small datasets to avoid data imbalance"""
        print(f"\n[Data Balancing] Target size: {target_size:,}")
        balanced_info = {}
        
        for dataset_name, info in dataset_info.items():
            texts = info['texts']
            labels = info.get('labels', [])
            original_count = len(texts)
            
            if original_count < target_size:
                repeat_times = int(np.ceil(target_size / original_count))
                balanced_texts = (texts * repeat_times)[:target_size]
                balanced_labels = (list(labels) * repeat_times)[:target_size] if labels is not None else []
                print(f"  {dataset_name}: {original_count:,} → {target_size:,} (repeat {repeat_times}x)")
            else:
                balanced_texts = texts[:target_size]
                balanced_labels = list(labels)[:target_size] if labels is not None else []
                print(f"  {dataset_name}: {original_count:,} → {target_size:,}")
            
            balanced_info[dataset_name] = {
                'texts': balanced_texts,
                'labels': balanced_labels,
                'count': len(balanced_texts),
                'start_idx': 0
            }
        
        return balanced_info
    
    # Apply data balancing
    print("\n[Step 2/5] Balancing datasets...")
    dataset_info = oversample_small_datasets(dataset_info, target_size=100000)
    
    # Merge all datasets
    all_texts = []
    all_labels = []
    current_idx = 0
    
    for dataset_name, info in dataset_info.items():
        info['start_idx'] = current_idx
        all_texts.extend(info['texts'])
        all_labels.extend(info['labels'])
        current_idx += info['count']
    
    print(f"\nTotal samples after balancing: {len(all_texts):,}")
    print(f"Total classes: {len(np.unique(all_labels))}")
    
    print("\n[Step 2/5] Initializing trainer...")
    use_lora = args.use_lora and not args.no_lora
    
    config = TrainingConfig(
        model_path=args.model_path,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=use_lora,
        output_dir=args.output_dir,
        result_dir=args.result_dir,
        model_size=args.model_size,
        exp_dir=args.exp_dir
    )
    
    trainer = EmbeddingTrainer(config=config, dev_mode=args.dev)
    
    print("\n[Step 3/5] Training on joint dataset...")
    start_time = time.time()
    
    train_result = trainer.train_supervised(
        np.array(all_texts),
        np.array(all_labels),
        dataset_name="joint_all"
    )
    
    training_time = time.time() - start_time
    print(f"\nJoint training completed in {training_time:.1f}s")
    
    print("\n[Step 4/5] Generating embeddings for each dataset...")
    results = {}
    
    # Generate embeddings for all texts at once (pass as list for tokenizer)
    all_embeddings = trainer.generate_embeddings(all_texts)
    
    print("\n[Step 5/5] Saving embeddings per dataset...")
    for dataset_name, info in dataset_info.items():
        start_idx = info['start_idx']
        count = info['count']
        labels = info['labels']
        
        # Extract embeddings for this dataset
        embeddings = all_embeddings[start_idx:start_idx + count]
        
        # Save embeddings
        saved_paths = trainer.save_embeddings(
            embeddings, dataset_name, "joint_supervised", labels
        )
        
        results[dataset_name] = {
            "status": "success",
            "num_samples": count,
            "embedding_shape": embeddings.shape,
            "final_loss": train_result['final_loss'],
            "elapsed_time": training_time / len(dataset_info),
            "paths": saved_paths
        }
        print(f"  Saved {dataset_name}: {embeddings.shape}")
    
    return results


def run_generate_vocab_embeddings(args):
    """
    Generate embeddings for vocabulary words.
    This is critical for ETM's rho matrix initialization.
    """
    import json
    import numpy as np
    
    print("\n" + "=" * 70)
    print("GENERATE VOCABULARY EMBEDDINGS")
    print("Creating word embeddings for ETM decoder")
    print("=" * 70)
    
    if not args.vocab_file:
        raise ValueError("--vocab_file is required for generate_vocab_embeddings mode")
    
    if not os.path.exists(args.vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found: {args.vocab_file}")
    
    print(f"\n[Step 1/4] Loading vocabulary from {args.vocab_file}...")
    with open(args.vocab_file, 'r') as f:
        vocab_data = json.load(f)
    
    # Extract word list in strict index order
    if isinstance(vocab_data, dict):
        if 'word2idx' in vocab_data:
            word2idx = vocab_data['word2idx']
            # CRITICAL: Sort by index to maintain alignment
            idx2word = {idx: word for word, idx in word2idx.items()}
            words = [idx2word[i] for i in range(len(word2idx))]
            
            print(f"Vocabulary size: {len(words):,}")
            print(f"Index range: 0 to {len(words)-1}")
            
            # Validate continuity
            if set(idx2word.keys()) != set(range(len(word2idx))):
                raise ValueError("Vocabulary indices are not continuous! Check global_vocab.json")
            
            print("✓ Vocabulary indices are continuous")
            
        elif 'words' in vocab_data:
            words = vocab_data['words']
        else:
            raise ValueError("Vocabulary file must contain 'word2idx' or 'words' key")
    elif isinstance(vocab_data, list):
        words = vocab_data
    else:
        raise ValueError("Vocabulary file must be a dict or list")
    
    if 'word2idx' not in vocab_data:
        print(f"Vocabulary size: {len(words):,}")
    
    print("\n[Step 2/4] Initializing embedder...")
    embedder = QwenEmbedder(
        model_path=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        dev_mode=args.dev
    )
    
    print("\n[Step 3/4] Generating embeddings for vocabulary words...")
    # Generate embeddings (order is preserved!)
    vocab_embeddings = embedder.encode(
        words,
        normalize=not args.no_normalize,
        show_progress=True
    )
    
    print(f"\nGenerated embeddings shape: {vocab_embeddings.shape}")
    
    # Validate alignment
    assert vocab_embeddings.shape[0] == len(words), (
        f"Embedding count mismatch! Expected {len(words)}, got {vocab_embeddings.shape[0]}"
    )
    
    # Sample check
    print("\nVocabulary alignment verification (sample check):")
    for i in [0, len(words)//2, len(words)-1]:
        print(f"  Index {i}: '{words[i]}'")
    print("✓ Vocabulary embeddings generated in correct order")
    
    print("\n[Step 4/4] Saving vocabulary embeddings...")
    # Save to output directory
    output_path = os.path.join(args.output_dir, "qwen_vocab_vectors.npy")
    np.save(output_path, vocab_embeddings)
    print(f"Saved to: {output_path}")
    
    # Also save to result directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(
        args.result_dir, 
        f"qwen_vocab_vectors_{timestamp}.npy"
    )
    os.makedirs(args.result_dir, exist_ok=True)
    np.save(result_path, vocab_embeddings)
    print(f"Saved to result: {result_path}")
    
    # Save metadata
    metadata = {
        'vocab_file': args.vocab_file,
        'vocab_size': len(words),
        'embedding_dim': vocab_embeddings.shape[1],
        'model_path': args.model_path,
        'normalized': not args.no_normalize,
        'timestamp': timestamp
    }
    
    metadata_path = os.path.join(args.output_dir, "qwen_vocab_vectors_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("✓ Vocabulary embeddings generated successfully")
    print("=" * 70)
    print(f"Shape: {vocab_embeddings.shape}")
    print(f"Output: {output_path}")
    print("\nYou can now use this file to initialize ETM's rho matrix:")
    print(f"  python ETM/main.py --mode train --vocab_embeddings {output_path}")
    print("=" * 70)
    
    return {
        "vocab_embeddings": {
            "status": "success",
            "vocab_size": len(words),
            "embedding_shape": vocab_embeddings.shape,
            "output_path": output_path,
            "result_path": result_path
        }
    }


def run_generate(args):
    """
    Generate embeddings using a trained model checkpoint.
    This is used after joint training to generate embeddings for each dataset.
    """
    from trainer import EmbeddingTrainerV2 as EmbeddingTrainer, TrainingConfig
    import numpy as np
    
    print("\n" + "=" * 70)
    print("GENERATE EMBEDDINGS")
    print("Using trained model to generate embeddings")
    print("=" * 70)
    
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for generate mode")
    
    print("\n[Step 1/4] Initializing trainer and loading checkpoint...")
    use_lora = args.use_lora and not args.no_lora
    
    config = TrainingConfig(
        model_path=args.model_path,
        max_length=args.max_length,
        epochs=1,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=use_lora,
        output_dir=args.output_dir,
        result_dir=args.result_dir,
        model_size=args.model_size,
        exp_dir=args.exp_dir
    )
    
    trainer = EmbeddingTrainer(config=config, dev_mode=args.dev)
    trainer.load_checkpoint(args.checkpoint)
    
    print("\n[Step 2/4] Loading datasets...")
    loader = DatasetLoader(dev_mode=args.dev)
    
    if args.dataset.lower() == "all":
        dataset_names = loader.get_available_datasets()
    else:
        dataset_names = [args.dataset]
    
    print(f"Datasets to process: {dataset_names}")
    
    print("\n[Step 3/4] Generating embeddings...")
    results = {}
    
    for dataset_name in dataset_names:
        start_time = time.time()
        
        texts, labels, info = loader.load_dataset(
            dataset_name,
            max_samples=args.max_samples,
            shuffle=args.shuffle,
            random_seed=args.seed
        )
        
        print(f"\n{dataset_name}: {len(texts)} samples")
        
        embeddings = trainer.generate_embeddings(texts)
        
        saved_paths = trainer.save_embeddings(
            embeddings, dataset_name, "generated", labels
        )
        
        elapsed = time.time() - start_time
        
        results[dataset_name] = {
            "status": "success",
            "num_samples": len(texts),
            "embedding_shape": embeddings.shape,
            "final_loss": 0.0,
            "elapsed_time": elapsed,
            "paths": saved_paths
        }
    
    return results


def print_summary(results: dict):
    """Print final summary of results"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_samples = 0
    total_time = 0
    success_count = 0
    
    for name, result in results.items():
        status_icon = "OK" if result["status"] == "success" else "FAILED"
        print(f"\n{name}: [{status_icon}]")
        
        if result["status"] == "success":
            print(f"  Samples: {result['num_samples']}")
            print(f"  Shape: {result['embedding_shape']}")
            print(f"  Time: {result['elapsed_time']:.1f}s")
            total_samples += result['num_samples']
            success_count += 1
        else:
            print(f"  Error: {result['error']}")
        
        total_time += result['elapsed_time']
    
    print("\n" + "-" * 70)
    print(f"Total datasets: {len(results)} ({success_count} succeeded)")
    print(f"Total samples: {total_samples}")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 70)


def main():
    """Main entry point"""
    args = parse_args()
    
    # Print header
    print_header(args)
    
    # Show dataset summary in dev mode
    if args.dev:
        print("\n[DEV] Dataset Summary:")
        print(get_dataset_summary())
    
    # Run based on mode
    try:
        if args.mode == "zero_shot":
            results = run_zero_shot(args)
        elif args.mode == "supervised":
            results = run_supervised(args)
        elif args.mode == "unsupervised":
            results = run_unsupervised(args)
        elif args.mode == "joint_unsupervised":
            results = run_joint_unsupervised(args)
        elif args.mode == "joint_supervised":
            results = run_joint_supervised(args)
        elif args.mode == "generate":
            results = run_generate(args)
        elif args.mode == "generate_vocab_embeddings":
            results = run_generate_vocab_embeddings(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Print summary
        print_summary(results)
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if args.dev:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



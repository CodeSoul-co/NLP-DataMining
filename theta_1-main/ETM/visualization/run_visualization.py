#!/usr/bin/env python3
"""
ETM Unified Visualization Runner
统一可视化运行脚本 - 训练完成后一键生成所有可视化

Usage:
    python run_visualization.py --result_dir /path/to/result --dataset socialTwitter --mode zero_shot
    
    # Or use the convenience function:
    from visualization.run_visualization import run_all_visualizations
    run_all_visualizations(result_dir, dataset, mode)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_latest_file(directory, pattern):
    """Find the latest file matching pattern in directory."""
    from glob import glob
    files = glob(str(Path(directory) / pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_visualization_data(result_dir, dataset, mode, model_size=None):
    """
    Load all data needed for visualization from result directory.
    
    Args:
        result_dir: Base result directory (e.g., /root/autodl-tmp/result)
        dataset: Dataset name (e.g., socialTwitter)
        mode: Training mode (zero_shot, supervised, unsupervised)
        model_size: Optional model size subdirectory (e.g., 0.6B)
    
    Returns:
        dict with all visualization data
    """
    from scipy import sparse
    
    result_dir = Path(result_dir)
    
    # Handle model_size subdirectory
    if model_size:
        base_dir = result_dir / model_size / dataset / mode
    else:
        base_dir = result_dir / dataset / mode
    
    # Also check if result_dir already includes model_size/dataset/mode
    if not base_dir.exists():
        # Try direct path
        if (result_dir / 'model').exists():
            base_dir = result_dir
        else:
            raise FileNotFoundError(f"Result directory not found: {base_dir}")
    
    model_dir = base_dir / 'model'
    evaluation_dir = base_dir / 'evaluation'
    topic_words_dir = base_dir / 'topic_words'
    
    # Resolve BOW directory
    # 1. Check if base_dir has a config.json with data_exp reference
    bow_dir = None
    config_file = base_dir / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                train_config = json.load(f)
            data_exp = train_config.get('data_exp', '')
            ms = train_config.get('model_size', model_size or '')
            ds = train_config.get('dataset', dataset)
            if data_exp and data_exp != 'legacy':
                # New exp structure: find bow in data exp dir
                candidate = result_dir.parent.parent / 'data' / data_exp / 'bow'
                if not candidate.exists() and ms:
                    candidate = Path('/root/autodl-tmp/result') / ms / ds / 'data' / data_exp / 'bow'
                if candidate.exists():
                    bow_dir = candidate
        except Exception:
            pass
    
    # 2. Fallback to legacy paths
    if bow_dir is None:
        if model_size:
            bow_dir = result_dir / model_size / dataset / 'bow'
        else:
            # Try parent directories
            candidate = base_dir.parent / 'bow'
            if candidate.exists():
                bow_dir = candidate
            else:
                bow_dir = result_dir / dataset / 'bow'
    
    print(f"\n{'='*60}")
    print(f"Loading visualization data")
    print(f"{'='*60}")
    print(f"Base directory: {base_dir}")
    
    data = {}
    
    # Load theta (document-topic distribution)
    theta_file = find_latest_file(model_dir, "theta_*.npy")
    if theta_file:
        data['theta'] = np.load(theta_file)
        print(f"✓ Loaded theta: {data['theta'].shape}")
    else:
        raise FileNotFoundError(f"theta not found in {model_dir}")
    
    # Load beta (topic-word distribution)
    beta_file = find_latest_file(model_dir, "beta_*.npy")
    if beta_file:
        data['beta'] = np.load(beta_file)
        print(f"✓ Loaded beta: {data['beta'].shape}")
    else:
        raise FileNotFoundError(f"beta not found in {model_dir}")
    
    # Load topic embeddings
    emb_file = find_latest_file(model_dir, "topic_embeddings_*.npy")
    if emb_file:
        data['topic_embeddings'] = np.load(emb_file)
        print(f"✓ Loaded topic_embeddings: {data['topic_embeddings'].shape}")
    
    # Load topic words - try topic_words_dir first, then model_dir
    words_file = find_latest_file(topic_words_dir, "topic_words_*.json")
    if not words_file:
        words_file = find_latest_file(model_dir, "topic_words_*.json")
    
    if words_file:
        with open(words_file, 'r', encoding='utf-8') as f:
            topic_words_raw = json.load(f)
        
        # Convert to standard format: [(topic_id, [(word, weight), ...]), ...]
        if isinstance(topic_words_raw, list):
            # Format: [[topic_id, [[word, weight], ...]], ...]
            data['topic_words'] = [
                (item[0], [(w[0], w[1]) for w in item[1]])
                for item in topic_words_raw
            ]
        elif isinstance(topic_words_raw, dict):
            # Format: {"0": [[word, weight], ...], ...}
            data['topic_words'] = [
                (int(k), [(w[0], w[1]) for w in v])
                for k, v in sorted(topic_words_raw.items(), key=lambda x: int(x[0]))
            ]
        print(f"✓ Loaded topic_words: {len(data['topic_words'])} topics")
    else:
        # Generate from beta
        n_topics = data['beta'].shape[0]
        data['topic_words'] = []
        for i in range(n_topics):
            top_indices = np.argsort(data['beta'][i])[-20:][::-1]
            words = [(f"word_{idx}", float(data['beta'][i, idx])) for idx in top_indices]
            data['topic_words'].append((i, words))
        print(f"⚠ Generated topic_words from beta: {len(data['topic_words'])} topics")
    
    # Load training history
    history_file = find_latest_file(model_dir, "training_history_*.json")
    if history_file:
        with open(history_file, 'r', encoding='utf-8') as f:
            data['training_history'] = json.load(f)
        print(f"✓ Loaded training_history: {len(data['training_history'].get('train_loss', []))} epochs")
    
    # Load evaluation metrics
    metrics_file = find_latest_file(evaluation_dir, "metrics_*.json")
    if metrics_file:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data['metrics'] = json.load(f)
        print(f"✓ Loaded metrics")
    
    # Load vocab
    vocab_file = bow_dir / 'vocab.txt'
    if vocab_file.exists():
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data['vocab'] = [line.strip() for line in f.readlines()]
        print(f"✓ Loaded vocab: {len(data['vocab'])} words")
    else:
        # Generate placeholder vocab
        data['vocab'] = [f"word_{i}" for i in range(data['beta'].shape[1])]
        print(f"⚠ Generated placeholder vocab: {len(data['vocab'])} words")
    
    # Load BOW matrix (optional)
    bow_file = bow_dir / 'bow_matrix.npy'
    if bow_file.exists():
        data['bow_matrix'] = np.load(bow_file)
        print(f"✓ Loaded bow_matrix: {data['bow_matrix'].shape}")
    
    # Load timestamps (optional)
    ts_file = base_dir / 'timestamps.npy'
    if ts_file.exists():
        data['timestamps'] = np.load(ts_file, allow_pickle=True)
        print(f"✓ Loaded timestamps: {len(data['timestamps'])}")
    
    # Load config
    config_file = find_latest_file(model_dir, "config_*.json")
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as f:
            data['config'] = json.load(f)
        print(f"✓ Loaded config")
    
    print(f"{'='*60}\n")
    
    return data


def run_all_visualizations(
    result_dir,
    dataset,
    mode,
    model_size=None,
    output_dir=None,
    language='en',
    dpi=300
):
    """
    Run all visualizations for ETM results.
    
    Args:
        result_dir: Base result directory
        dataset: Dataset name
        mode: Training mode
        model_size: Optional model size subdirectory
        output_dir: Output directory for visualizations (default: result_dir/.../visualization)
        language: Language for labels ('en' or 'zh')
        dpi: DPI for saved figures
    
    Returns:
        Path to output directory
    """
    # Load data
    data = load_visualization_data(result_dir, dataset, mode, model_size)
    
    # Determine output directory
    if output_dir is None:
        result_dir = Path(result_dir)
        if model_size:
            output_dir = result_dir / model_size / dataset / mode / 'visualization'
        else:
            output_dir = result_dir / dataset / mode / 'visualization'
    
    output_dir = Path(output_dir)
    
    # Add timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"viz_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running ETM Visualizations")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Language: {language}")
    print(f"DPI: {dpi}")
    print(f"{'='*60}\n")
    
    # Import visualization generator
    from visualization.visualization_generator import VisualizationGenerator
    
    # Create generator
    generator = VisualizationGenerator(
        theta=data['theta'],
        beta=data['beta'],
        vocab=data['vocab'],
        topic_words=data['topic_words'],
        topic_embeddings=data.get('topic_embeddings'),
        timestamps=data.get('timestamps'),
        bow_matrix=data.get('bow_matrix'),
        training_history=data.get('training_history'),
        metrics=data.get('metrics'),
        output_dir=str(output_dir),
        language=language,
        dpi=dpi
    )
    
    # Generate all visualizations
    generator.generate_all()
    
    # Also run topic visualizer for additional charts
    try:
        from visualization.topic_visualizer import TopicVisualizer
        
        print(f"\n[Additional Visualizations]")
        
        viz = TopicVisualizer(output_dir=str(output_dir / 'global'), dpi=dpi, language=language)
        
        # Topic word bars
        viz.visualize_topic_words(
            data['topic_words'],
            num_words=10,
            filename='topic_words_bars.png'
        )
        print(f"  ✓ topic_words_bars.png")
        
        # Topic similarity heatmap
        viz.visualize_topic_similarity(
            data['beta'],
            data['topic_words'],
            filename='topic_similarity.png'
        )
        print(f"  ✓ topic_similarity.png")
        
        # Document-topic distribution
        viz.visualize_document_topics(
            data['theta'],
            method='umap',
            max_docs=5000,
            filename='doc_topic_umap.png'
        )
        print(f"  ✓ doc_topic_umap.png")
        
        # Training history
        if data.get('training_history'):
            viz.visualize_training_history(
                data['training_history'],
                filename='training_history.png'
            )
            print(f"  ✓ training_history.png")
        
        # Metrics
        if data.get('metrics'):
            viz.visualize_metrics(
                data['metrics'],
                filename='metrics.png'
            )
            print(f"  ✓ metrics.png")
        
        # Word clouds (if wordcloud package available)
        try:
            viz.visualize_all_wordclouds(
                data['topic_words'],
                num_words=30,
                filename='topic_wordclouds.png'
            )
            print(f"  ✓ topic_wordclouds.png")
        except Exception as e:
            print(f"  ⚠ topic_wordclouds skipped: {e}")
        
        # pyLDAvis-style visualization (Intertopic Distance Map)
        try:
            viz.visualize_pyldavis_style(
                data['theta'],
                data['beta'],
                data['topic_words'],
                selected_topic=0,
                n_words=30,
                filename='pyldavis_intertopic.png'
            )
            print(f"  ✓ pyldavis_intertopic.png")
        except Exception as e:
            print(f"  ⚠ pyldavis_intertopic skipped: {e}")
        
        # Also generate interactive HTML version if pyLDAvis is available
        try:
            from visualization.topic_visualizer import generate_pyldavis_visualization
            html_path = generate_pyldavis_visualization(
                theta=data['theta'],
                beta=data['beta'],
                bow_matrix=data.get('bow_matrix'),
                vocab=data['vocab'],
                output_path=str(output_dir / 'global' / 'pyldavis_interactive.html')
            )
            if html_path:
                print(f"  ✓ pyldavis_interactive.html")
        except Exception as e:
            print(f"  ⚠ pyldavis_interactive.html skipped: {e}")
        
    except Exception as e:
        print(f"  ⚠ Additional visualizations error: {e}")
    
    # Generate summary report
    generate_summary_report(data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir


def generate_summary_report(data, output_dir):
    """Generate a summary report of the visualization."""
    output_dir = Path(output_dir)
    
    report = []
    report.append("# ETM Visualization Summary Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data summary
    report.append("## Data Summary")
    report.append(f"- Documents: {data['theta'].shape[0]:,}")
    report.append(f"- Topics: {data['theta'].shape[1]}")
    report.append(f"- Vocabulary size: {len(data['vocab']):,}")
    report.append(f"- Has timestamps: {'Yes' if data.get('timestamps') is not None else 'No'}")
    report.append(f"- Has training history: {'Yes' if data.get('training_history') is not None else 'No'}")
    report.append(f"- Has metrics: {'Yes' if data.get('metrics') is not None else 'No'}")
    report.append("")
    
    # Topic summary
    report.append("## Topic Summary")
    report.append("")
    for topic_id, words in data['topic_words']:
        top_words = [w[0] for w in words[:10]]
        strength = data['theta'][:, topic_id].mean()
        report.append(f"### Topic {topic_id + 1}")
        report.append(f"- **Strength**: {strength:.6f}")
        report.append(f"- **Top words**: {', '.join(top_words)}")
        report.append("")
    
    # Metrics summary
    if data.get('metrics'):
        report.append("## Evaluation Metrics")
        metrics = data['metrics']
        if 'topic_diversity_td' in metrics:
            report.append(f"- Topic Diversity (TD): {metrics['topic_diversity_td']:.4f}")
        if 'topic_diversity_irbo' in metrics:
            report.append(f"- Topic Diversity (iRBO): {metrics['topic_diversity_irbo']:.4f}")
        if 'topic_coherence_npmi_avg' in metrics:
            report.append(f"- Coherence (NPMI): {metrics['topic_coherence_npmi_avg']:.4f}")
        if 'topic_coherence_cv_avg' in metrics:
            report.append(f"- Coherence (C_V): {metrics['topic_coherence_cv_avg']:.4f}")
        if 'perplexity' in metrics and metrics['perplexity'] is not None:
            report.append(f"- Perplexity: {metrics['perplexity']:.2f}")
        report.append("")
    
    # Training summary
    if data.get('training_history'):
        history = data['training_history']
        report.append("## Training Summary")
        if 'epochs_trained' in history:
            report.append(f"- Epochs trained: {history['epochs_trained']}")
        if 'best_val_loss' in history:
            report.append(f"- Best validation loss: {history['best_val_loss']:.4f}")
        if 'test_loss' in history:
            report.append(f"- Test loss: {history['test_loss']:.4f}")
        report.append("")
    
    # Generated files
    report.append("## Generated Visualizations")
    report.append("")
    report.append("### Global Charts")
    global_dir = output_dir / 'global'
    if global_dir.exists():
        for f in sorted(global_dir.glob('*.png')):
            report.append(f"- `{f.name}`")
    report.append("")
    
    report.append("### Per-Topic Charts")
    topics_dir = output_dir / 'topics'
    if topics_dir.exists():
        topic_dirs = sorted(topics_dir.glob('topic_*'))
        if topic_dirs:
            report.append(f"- {len(topic_dirs)} topic directories with individual charts")
    report.append("")
    
    # Write report
    report_path = output_dir / 'README.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"  ✓ README.md (summary report)")


def load_baseline_data(result_dir, dataset, model, num_topics=20):
    """
    Load baseline model data (LDA, ETM, CTM) for visualization.
    
    Args:
        result_dir: Result directory - can be:
            - New structure: full experiment path (e.g., .../models/lda/exp_xxx)
            - Old structure: base result directory (e.g., /root/autodl-tmp/result/baseline)
        dataset: Dataset name (e.g., socialTwitter)
        model: Model name (lda, etm, ctm_zeroshot)
        num_topics: Number of topics
    
    Returns:
        dict with all visualization data
    """
    from scipy import sparse
    
    result_dir = Path(result_dir)
    
    # Check if result_dir is already the experiment directory (new structure)
    # New structure: result_dir contains model subdirectory (e.g., lda/) directly
    data_exp_dir = None  # For loading vocab from data experiment
    
    if (result_dir / model).exists():
        model_dir = result_dir / model
        dataset_dir = result_dir
        # Try to load data_exp from config.json
        config_path = result_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            data_exp = config.get('data_exp')
            if data_exp:
                # Construct data experiment directory path
                data_exp_dir = result_dir.parent.parent.parent / 'data' / data_exp
    elif result_dir.name.startswith('exp_'):
        # result_dir is the experiment directory itself
        model_dir = result_dir / model
        dataset_dir = result_dir
        if not model_dir.exists():
            # Try looking for files directly in result_dir
            model_dir = result_dir
        # Try to load data_exp from config.json
        config_path = result_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            data_exp = config.get('data_exp')
            if data_exp:
                data_exp_dir = result_dir.parent.parent.parent / 'data' / data_exp
    else:
        # Old structure: result_dir / dataset / model
        dataset_dir = result_dir / dataset
        model_dir = dataset_dir / model
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"\n{'='*60}")
    print(f"Loading baseline data: {dataset} / {model}")
    print(f"{'='*60}")
    
    data = {}
    
    # Load theta from various possible subdirectories
    theta_path = None
    beta_path = None
    
    # Check paths in priority order
    possible_paths = [
        model_dir / 'model' / f'theta_k{num_topics}.npy',  # HDP/neural
        model_dir / f'theta_k{num_topics}.npy',  # direct
        model_dir / model / f'theta_k{num_topics}.npy',  # model subdir
        model_dir / model / 'model' / f'theta_k{num_topics}.npy',  # model/model subdir
    ]
    
    # CTM saves to ctm_zeroshot/ or ctm_combined/
    if model == 'ctm':
        possible_paths.insert(0, model_dir / 'ctm_zeroshot' / f'theta_k{num_topics}.npy')
        possible_paths.insert(1, model_dir / 'ctm_combined' / f'theta_k{num_topics}.npy')
    
    for p in possible_paths:
        if p.exists():
            theta_path = p
            beta_path = p.parent / f'beta_k{num_topics}.npy'
            break
    
    if theta_path and theta_path.exists():
        data['theta'] = np.load(theta_path)
        print(f"✓ Loaded theta: {data['theta'].shape}")
    else:
        raise FileNotFoundError(f"theta not found in any of: {[str(p) for p in possible_paths]}")
    
    if beta_path and beta_path.exists():
        data['beta'] = np.load(beta_path)
        print(f"✓ Loaded beta: {data['beta'].shape}")
    else:
        raise FileNotFoundError(f"beta not found: {beta_path}")
    
    # Load vocab - try multiple locations
    vocab_path = None
    # 1. Try data_exp_dir first (new structure)
    if data_exp_dir and (data_exp_dir / 'vocab.json').exists():
        vocab_path = data_exp_dir / 'vocab.json'
    # 2. Try model_dir / bow / vocab.json (old structure)
    elif (model_dir / 'bow' / 'vocab.json').exists():
        vocab_path = model_dir / 'bow' / 'vocab.json'
    # 3. Try dataset_dir / vocab.json
    elif (dataset_dir / 'vocab.json').exists():
        vocab_path = dataset_dir / 'vocab.json'
    
    if vocab_path and vocab_path.exists():
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data['vocab'] = json.load(f)
        print(f"✓ Loaded vocab: {len(data['vocab'])} words")
    else:
        data['vocab'] = [f"word_{i}" for i in range(data['beta'].shape[1])]
        print(f"⚠ Generated placeholder vocab: {len(data['vocab'])} words")
    
    # Load topic_words from topicwords/ subdirectory
    topic_words_path = model_dir / 'topicwords' / f'topic_words_k{num_topics}.json'
    if not topic_words_path.exists():
        topic_words_path = model_dir / f'topic_words_k{num_topics}.json'
    if topic_words_path.exists():
        with open(topic_words_path, 'r', encoding='utf-8') as f:
            topic_words_raw = json.load(f)
        
        topic_words = []
        if isinstance(topic_words_raw, dict):
            # Format: {"topic_0": ["word1", ...], ...}
            sorted_items = sorted(topic_words_raw.items(), 
                                 key=lambda x: int(x[0].replace('topic_', '')) if 'topic_' in x[0] else int(x[0]))
            for key, words in sorted_items:
                topic_id = int(key.replace('topic_', '')) if 'topic_' in key else int(key)
                if isinstance(words, list) and len(words) > 0 and isinstance(words[0], str):
                    word_weights = []
                    for w in words:
                        idx = data['vocab'].index(w) if w in data['vocab'] else -1
                        weight = float(data['beta'][topic_id, idx]) if idx >= 0 else 0.01
                        word_weights.append((w, weight))
                    topic_words.append((topic_id, word_weights))
                else:
                    topic_words.append((topic_id, []))
        else:
            # Generate from beta
            for i in range(data['beta'].shape[0]):
                top_indices = np.argsort(-data['beta'][i])[:20]
                words = [(data['vocab'][idx], float(data['beta'][i, idx])) for idx in top_indices]
                topic_words.append((i, words))
        
        data['topic_words'] = topic_words
        print(f"✓ Loaded topic_words: {len(data['topic_words'])} topics")
    else:
        # Generate from beta
        topic_words = []
        for i in range(data['beta'].shape[0]):
            top_indices = np.argsort(-data['beta'][i])[:20]
            words = [(data['vocab'][idx], float(data['beta'][i, idx])) for idx in top_indices]
            topic_words.append((i, words))
        data['topic_words'] = topic_words
        print(f"⚠ Generated topic_words from beta")
    
    # Load BOW matrix - try multiple locations
    bow_path = None
    # 1. Try data_exp_dir first (new structure)
    if data_exp_dir and (data_exp_dir / 'bow_matrix.npy').exists():
        bow_path = data_exp_dir / 'bow_matrix.npy'
    # 2. Try model_dir / bow / bow_matrix.npy (old structure)
    elif (model_dir / 'bow' / 'bow_matrix.npy').exists():
        bow_path = model_dir / 'bow' / 'bow_matrix.npy'
    # 3. Try dataset_dir / bow_matrix.npy
    elif (dataset_dir / 'bow_matrix.npy').exists():
        bow_path = dataset_dir / 'bow_matrix.npy'
    
    if bow_path and bow_path.exists():
        data['bow_matrix'] = np.load(bow_path)
        print(f"✓ Loaded bow_matrix: {data['bow_matrix'].shape}")
    
    # Load metrics - try multiple locations
    metrics_path = None
    # 1. Try result_dir (experiment directory) first
    if (result_dir / f'metrics_k{num_topics}.json').exists():
        metrics_path = result_dir / f'metrics_k{num_topics}.json'
    # 2. Try model_dir / evaluation / metrics_k{num_topics}.json
    elif (model_dir / 'evaluation' / f'metrics_k{num_topics}.json').exists():
        metrics_path = model_dir / 'evaluation' / f'metrics_k{num_topics}.json'
    # 3. Try model_dir / metrics_k{num_topics}.json
    elif (model_dir / f'metrics_k{num_topics}.json').exists():
        metrics_path = model_dir / f'metrics_k{num_topics}.json'
    
    if metrics_path and metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data['metrics'] = json.load(f)
        print(f"✓ Loaded metrics from {metrics_path.name}")
    
    # Load timestamps for DTM (time_slices.json and time_indices.npy)
    data['topic_embeddings'] = None
    data['training_history'] = None
    data['timestamps'] = None
    
    # Try to load timestamp data (required for DTM)
    time_slices_path = dataset_dir / 'time_slices.json'
    time_indices_path = dataset_dir / 'time_indices.npy'
    
    if time_slices_path.exists() and time_indices_path.exists():
        with open(time_slices_path, 'r', encoding='utf-8') as f:
            time_slices_info = json.load(f)
        time_indices = np.load(time_indices_path)
        
        # 将time_indices转换为datetime对象列表
        from datetime import datetime
        unique_times = time_slices_info.get('unique_times', [])
        
        # 创建时间戳数组（每个文档对应一个时间戳）
        timestamps = []
        for idx in time_indices:
            if idx < len(unique_times):
                year = unique_times[idx]
                timestamps.append(datetime(year, 1, 1))
            else:
                timestamps.append(datetime(2020, 1, 1))
        
        data['timestamps'] = np.array(timestamps)
        data['time_slices_info'] = time_slices_info
        print(f"✓ Loaded timestamps: {len(data['timestamps'])} dates ({len(unique_times)} unique years)")
    
    # 尝试加载训练历史（DTM需要）
    training_history_path = model_dir / f'training_history_k{num_topics}.json'
    if training_history_path.exists():
        with open(training_history_path, 'r', encoding='utf-8') as f:
            data['training_history'] = json.load(f)
        print(f"✓ Loaded training_history")
    
    print(f"{'='*60}\n")
    return data


def _run_dtm_specific_visualizations(data, output_dir, language='en', dpi=300):
    """
    Run DTM-specific visualizations using visualization_generator2.
    
    DTM-specific visualizations:
    - Topic evolution sankey diagram (topic_sankey.png)
    - Topic strength temporal changes (topic_similarity_evolution.png)
    - All topics strength table (all_topics_strength_table.png)
    - High-frequency word evolution (vocab_evolution.png)
    - Topic independence visualization (topic_independence.png)
    - Global word cloud (wordcloud_global.png)
    - Topic proportion pie chart (topic_proportion.png)
    """
    from pathlib import Path
    import numpy as np
    
    output_dir = Path(output_dir)
    global_dir = output_dir / 'global'
    global_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load DTM-specific data (beta_over_time, topic_evolution)
    dtm_dir = output_dir.parent
    
    # Load beta_over_time from model/ subdirectory
    beta_over_time = None
    beta_over_time_file = dtm_dir / 'model' / f"beta_over_time_k{data['theta'].shape[1]}.npy"
    if not beta_over_time_file.exists():
        beta_over_time_file = dtm_dir / f"beta_over_time_k{data['theta'].shape[1]}.npy"
    if beta_over_time_file.exists():
        beta_over_time = np.load(beta_over_time_file)
        print(f"  Loaded beta_over_time: {beta_over_time.shape}")
    
    # Load topic_evolution from topicwords/ subdirectory
    topic_evolution = None
    topic_evolution_file = dtm_dir / 'topicwords' / f"topic_evolution_k{data['theta'].shape[1]}.json"
    if not topic_evolution_file.exists():
        topic_evolution_file = dtm_dir / f"topic_evolution_k{data['theta'].shape[1]}.json"
    if topic_evolution_file.exists():
        import json
        with open(topic_evolution_file, 'r', encoding='utf-8') as f:
            topic_evolution = json.load(f)
        print(f"  Loaded topic_evolution: {len(topic_evolution)} topics")
    
    # 使用visualization_generator2中的功能
    try:
        from visualization.visualization_generator2 import VisualizationGenerator as VG2
        
        # 创建生成器
        gen2 = VG2(
            theta=data['theta'],
            beta=data['beta'],
            vocab=data['vocab'],
            topic_words=data['topic_words'],
            topic_embeddings=data.get('topic_embeddings'),
            timestamps=data.get('timestamps'),
            bow_matrix=data.get('bow_matrix'),
            training_history=data.get('training_history'),
            metrics=data.get('metrics'),
            output_dir=str(output_dir),
            language=language,
            dpi=dpi
        )
        
        # 生成DTM特有的可视化
        # 1. 主题独立性可视化 (pyLDAvis风格)
        try:
            gen2.generate_pyldavis()
        except Exception as e:
            print(f"  ⚠ topic_independence.png skipped: {e}")
        
        # 2. 全局词云图
        try:
            gen2.generate_global_wordcloud()
        except Exception as e:
            print(f"  ⚠ wordcloud_global.png skipped: {e}")
        
        # 3. 主题占比饼图
        try:
            gen2.generate_topic_proportion_pie()
        except Exception as e:
            print(f"  ⚠ topic_proportion.png skipped: {e}")
        
        # 4. 如果有时间戳，生成时序相关图表
        if data.get('timestamps') is not None:
            # 桑基图
            try:
                gen2.generate_sankey_diagram()
            except Exception as e:
                print(f"  ⚠ topic_sankey.png skipped: {e}")
            
            # 主题强度时序变化
            try:
                gen2.generate_topic_similarity_evolution()
            except Exception as e:
                print(f"  ⚠ topic_similarity_evolution.png skipped: {e}")
            
            # 所有主题强度表
            try:
                gen2.generate_all_topics_strength_table()
            except Exception as e:
                print(f"  ⚠ all_topics_strength_table.png skipped: {e}")
        
        # 5. 如果有训练历史，生成训练相关图表
        if data.get('training_history') is not None:
            try:
                gen2.generate_training_convergence()
            except Exception as e:
                print(f"  ⚠ training_convergence skipped: {e}")
        
    except ImportError as e:
        print(f"  ⚠ visualization_generator2 not available: {e}")
    except Exception as e:
        print(f"  ⚠ DTM visualizations error: {e}")
    
    # 生成DTM主题词演化可视化（如果有topic_evolution数据）
    if topic_evolution is not None:
        try:
            _generate_topic_word_evolution(topic_evolution, global_dir, language, dpi)
        except Exception as e:
            print(f"  ⚠ topic_word_evolution.png skipped: {e}")


def _generate_topic_word_evolution(topic_evolution, output_dir, language='en', dpi=300):
    """生成DTM主题词演化可视化"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    output_dir = Path(output_dir)
    n_topics = len(topic_evolution)
    
    # 选择前6个主题展示
    n_show = min(6, n_topics)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (topic_id, time_data) in enumerate(list(topic_evolution.items())[:n_show]):
        ax = axes[idx]
        
        # 获取时间点和词
        times = sorted(time_data.keys())
        
        if len(times) < 2:
            ax.text(0.5, 0.5, f'Topic {topic_id}\n(insufficient data)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # 获取所有时间点的top词
        all_words = set()
        for t in times:
            words = time_data[t][:5]  # top 5 words
            for w, _ in words:
                all_words.add(w)
        
        # 绘制词权重随时间变化
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_words)))
        
        for i, word in enumerate(list(all_words)[:5]):
            weights = []
            for t in times:
                weight = 0
                for w, wt in time_data[t]:
                    if w == word:
                        weight = wt
                        break
                weights.append(weight)
            
            ax.plot(range(len(times)), weights, 'o-', color=colors[i], 
                   label=word, linewidth=2, markersize=4)
        
        ax.set_title(f'Topic {topic_id}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time' if language == 'en' else '时间', fontsize=9)
        ax.set_ylabel('Weight' if language == 'en' else '权重', fontsize=9)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')
    
    title = 'DTM Topic Word Evolution' if language == 'en' else 'DTM主题词演化'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_word_evolution.png', dpi=dpi, 
               bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ topic_word_evolution.png")


def run_baseline_visualization(
    result_dir,
    dataset,
    model,
    num_topics=20,
    output_dir=None,
    language='en',
    dpi=300
):
    """
    Run visualizations for baseline models (LDA, ETM, CTM, DTM).
    
    Args:
        result_dir: Base result directory for baseline models
        dataset: Dataset name
        model: Model name (lda, etm, ctm_zeroshot, dtm)
        num_topics: Number of topics
        output_dir: Output directory (default: result_dir/dataset/model/visualization_k{num_topics}_{timestamp})
        language: Language for labels ('en' or 'zh')
        dpi: DPI for saved figures
    
    Returns:
        Path to output directory
    """
    # Load data
    data = load_baseline_data(result_dir, dataset, model, num_topics)
    
    # Determine output directory with k and language in folder name
    if output_dir is None:
        result_path = Path(result_dir)
        lang_suffix = 'zh' if language == 'zh' else 'en'
        viz_folder = f'visualization_k{num_topics}_{lang_suffix}'
        
        # Check if result_dir is already an experiment directory (new structure)
        if result_path.name.startswith('exp_') or (result_path / model).exists():
            # New structure: output to result_dir/visualization_k{num_topics}_{language}/
            output_dir = result_path / viz_folder
        else:
            # Old structure: result_dir/dataset/model/visualization_k{num_topics}_{language}
            output_dir = result_path / dataset / model / viz_folder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running Visualizations for {model.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Use VisualizationGenerator
    from visualization.visualization_generator import VisualizationGenerator
    
    generator = VisualizationGenerator(
        theta=data['theta'],
        beta=data['beta'],
        vocab=data['vocab'],
        topic_words=data['topic_words'],
        topic_embeddings=data.get('topic_embeddings'),
        timestamps=data.get('timestamps'),
        bow_matrix=data.get('bow_matrix'),
        training_history=data.get('training_history'),
        metrics=data.get('metrics'),
        output_dir=str(output_dir),
        language=language,
        dpi=dpi
    )
    
    generator.generate_all()
    
    # Additional visualizations using TopicVisualizer
    try:
        from visualization.topic_visualizer import TopicVisualizer
        
        print(f"\n[Additional Visualizations]")
        global_dir = output_dir / 'global'
        viz = TopicVisualizer(output_dir=str(global_dir), dpi=dpi, language=language)
        
        # Filename mapping for zh/en
        filenames = {
            'topic_words_bars': '主题词条形图.png' if language == 'zh' else 'topic_words_bars.png',
            'topic_similarity': '主题相似度.png' if language == 'zh' else 'topic_similarity.png',
            'doc_topic_umap': '文档主题UMAP.png' if language == 'zh' else 'doc_topic_umap.png',
            'metrics': '评估指标图.png' if language == 'zh' else 'metrics.png',
            'topic_wordclouds': '主题词云.png' if language == 'zh' else 'topic_wordclouds.png',
            'pyldavis_intertopic': '主题距离图.png' if language == 'zh' else 'pyldavis_intertopic.png',
            'pyldavis_interactive': '交互式主题可视化.html' if language == 'zh' else 'pyldavis_interactive.html',
        }
        
        viz.visualize_topic_words(data['topic_words'], num_words=10, filename=filenames['topic_words_bars'])
        print(f"  ✓ {filenames['topic_words_bars']}")
        
        viz.visualize_topic_similarity(data['beta'], data['topic_words'], filename=filenames['topic_similarity'])
        print(f"  ✓ {filenames['topic_similarity']}")
        
        viz.visualize_document_topics(data['theta'], method='umap', max_docs=5000, filename=filenames['doc_topic_umap'])
        print(f"  ✓ {filenames['doc_topic_umap']}")
        
        # Note: metrics chart is already generated by VisualizationGenerator (评估指标.png)
        
        try:
            viz.visualize_all_wordclouds(data['topic_words'], num_words=30, filename=filenames['topic_wordclouds'])
            print(f"  ✓ {filenames['topic_wordclouds']}")
        except Exception as e:
            print(f"  ⚠ {filenames['topic_wordclouds']} skipped: {e}")
        
        try:
            viz.visualize_pyldavis_style(data['theta'], data['beta'], data['topic_words'], 
                                        selected_topic=0, n_words=30, filename=filenames['pyldavis_intertopic'])
            print(f"  ✓ {filenames['pyldavis_intertopic']}")
        except Exception as e:
            print(f"  ⚠ {filenames['pyldavis_intertopic']} skipped: {e}")
        
        try:
            from visualization.topic_visualizer import generate_pyldavis_visualization
            html_path = generate_pyldavis_visualization(
                theta=data['theta'], beta=data['beta'], bow_matrix=data.get('bow_matrix'),
                vocab=data['vocab'], output_path=str(global_dir / filenames['pyldavis_interactive'])
            )
            if html_path:
                print(f"  ✓ {filenames['pyldavis_interactive']}")
        except Exception as e:
            print(f"  ⚠ pyldavis_interactive.html skipped: {e}")
            
    except Exception as e:
        print(f"  ⚠ Additional visualizations error: {e}")
    
    # DTM专用可视化（使用visualization_generator2中的额外功能）
    if model == 'dtm':
        try:
            print(f"\n[DTM-Specific Visualizations]")
            _run_dtm_specific_visualizations(data, output_dir, language, dpi)
        except Exception as e:
            print(f"  ⚠ DTM-specific visualizations error: {e}")
    
    generate_summary_report(data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir


def run_all_baseline_visualizations(
    result_dir='/root/autodl-tmp/result/baseline',
    datasets=None,
    models=None,
    num_topics=20,
    language='en',
    dpi=300
):
    """
    Run visualizations for all baseline models.
    
    Args:
        result_dir: Base result directory
        datasets: List of datasets (default: all)
        models: List of models (default: all)
        num_topics: Number of topics
        language: Language for labels
        dpi: DPI for figures
    """
    if datasets is None:
        datasets = ['socialTwitter', 'hatespeech', 'mental_health', 'FCPB', 'germanCoal']
    if models is None:
        models = ['lda', 'etm', 'ctm_zeroshot']
    
    print("="*70)
    print("Running Visualizations for All Baseline Models")
    print("="*70)
    
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for model in models:
            print(f"\n>>> {dataset} / {model}")
            try:
                run_baseline_visualization(result_dir, dataset, model, num_topics, language=language, dpi=dpi)
                results[dataset][model] = 'SUCCESS'
            except Exception as e:
                print(f"  [ERROR] {e}")
                results[dataset][model] = f'FAILED: {e}'
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for dataset, models_result in results.items():
        print(f"\n{dataset}:")
        for model, status in models_result.items():
            print(f"  {model}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description='ETM Unified Visualization Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # THETA model visualization
    python run_visualization.py --result_dir /root/autodl-tmp/result --dataset socialTwitter --mode zero_shot
    
    # Baseline model visualization
    python run_visualization.py --baseline --result_dir /root/autodl-tmp/result/baseline --dataset FCPB --model lda
    
    # All baseline models
    python run_visualization.py --baseline --all
        """
    )
    
    # Baseline mode arguments
    parser.add_argument('--baseline', action='store_true',
                        help='Run visualization for baseline models (LDA, ETM, CTM)')
    parser.add_argument('--all', action='store_true',
                        help='Run for all datasets and models (baseline mode only)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['lda', 'hdp', 'stm', 'btm', 'etm', 'ctm', 'ctm_zeroshot', 'dtm', 'nvdm', 'gsm', 'prodlda', 'bertopic'],
                        help='Model name (baseline mode only)')
    parser.add_argument('--num_topics', type=int, default=20,
                        help='Number of topics (baseline mode only)')
    
    # Common arguments
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Base result directory')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['zero_shot', 'supervised', 'unsupervised'],
                        help='Training mode (THETA mode only)')
    parser.add_argument('--model_size', type=str, default=None,
                        help='Model size subdirectory (e.g., 0.6B)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto)')
    parser.add_argument('--language', type=str, default='en',
                        choices=['en', 'zh'],
                        help='Language for labels')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    
    args = parser.parse_args()
    
    if args.baseline:
        # Baseline model visualization
        if args.all:
            run_all_baseline_visualizations(
                result_dir=args.result_dir or '/root/autodl-tmp/result/baseline',
                num_topics=args.num_topics,
                language=args.language,
                dpi=args.dpi
            )
        elif args.dataset and args.model:
            run_baseline_visualization(
                result_dir=args.result_dir or '/root/autodl-tmp/result/baseline',
                dataset=args.dataset,
                model=args.model,
                num_topics=args.num_topics,
                output_dir=args.output_dir,
                language=args.language,
                dpi=args.dpi
            )
        else:
            parser.error("Baseline mode requires --all or both --dataset and --model")
    else:
        # THETA model visualization
        if not args.result_dir or not args.dataset or not args.mode:
            parser.error("THETA mode requires --result_dir, --dataset, and --mode")
        run_all_visualizations(
            result_dir=args.result_dir,
            dataset=args.dataset,
            mode=args.mode,
            model_size=args.model_size,
            output_dir=args.output_dir,
            language=args.language,
            dpi=args.dpi
        )


if __name__ == '__main__':
    main()

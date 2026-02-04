#!/usr/bin/env python
"""
Data preprocessing script - Generate required preprocessing files for new datasets

Supported preprocessing types:
1. THETA: Qwen embedding + BOW + vocab_embeddings
2. Baseline (LDA/ETM/CTM): BOW + SBERT embeddings (CTM specific)
3. DTM: BOW + time slice information (requires timestamp column)

Usage:
    # Prepare data for THETA (generate Qwen embedding and BOW)
    python prepare_data.py --dataset new_dataset --model theta --model_size 0.6B --mode zero_shot
    
    # Prepare data for baseline models (generate BOW)
    python prepare_data.py --dataset new_dataset --model baseline
    
    # Prepare data for DTM (requires timestamps, generate BOW + time slices)
    python prepare_data.py --dataset edu_data --model dtm --time_column year
    
    # Check data file locations
    python prepare_data.py --dataset socialTwitter --model theta --model_size 0.6B --check-only
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATASET_CONFIGS, RESULT_DIR, DATA_DIR, QWEN_MODEL_PATH,
    get_qwen_model_path, get_embedding_dim, QWEN_MODEL_PATHS, EMBEDDING_DIMS,
    get_sbert_model_path, get_job_result_dir, get_job_data_dir
)


def parse_args():
    parser = argparse.ArgumentParser(description='Data preprocessing script')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--job_id', type=str, default=None, 
                        help='Unique job ID for multi-user support')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['theta', 'baseline', 'dtm'],
                        help='Target model type: theta, baseline or dtm')
    parser.add_argument('--model_size', type=str, default='0.6B',
                        choices=['0.6B', '4B', '8B'],
                        help='Qwen model size (THETA specific)')
    parser.add_argument('--mode', type=str, default='zero_shot',
                        choices=['zero_shot', 'supervised', 'unsupervised'],
                        help='THETA mode')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--bow-only', action='store_true', help='Only generate BOW')
    parser.add_argument('--check-only', action='store_true', help='Only check files')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--clean', action='store_true', 
                        help='Perform data cleaning first (generate cleaned CSV from raw text)')
    parser.add_argument('--raw-input', type=str, default=None,
                        help='Raw data input path (use with --clean)')
    parser.add_argument('--language', type=str, default='english',
                        choices=['english', 'chinese'],
                        help='Language for data cleaning (use with --clean)')
    # DTM specific parameters
    parser.add_argument('--time_column', type=str, default='year',
                        help='Time column name (DTM specific)')
    parser.add_argument('--time_slices', type=int, default=None,
                        help='Number of time slices, auto-detect by default (DTM specific)')
    return parser.parse_args()


def process_docx_directory(input_dir: str, dataset: str, language: str = 'chinese', job_id: Optional[str] = None) -> Path:
    """
    Process a docx directory and convert all docx files to a CSV with timestamps.
    Uses the dataclean module for text extraction and cleaning.
    
    Supports a directory structure organized by year:
    input_dir/
    ├── province1/
    │   ├── 2020/
    │   │   └── xxx.docx
    │   └── 2021/
    │       └── xxx.docx
    """
    import re
    
    # Use dataclean module
    sys.path.insert(0, str(Path(__file__).parent / 'dataclean'))
    from dataclean.src.converter import TextConverter
    from dataclean.src.cleaner import TextCleaner
    
    print(f"\n[Processing DOCX directory] {input_dir}")
    print(f"  Using dataclean module for text extraction and cleaning")
    
    converter = TextConverter()
    cleaner = TextCleaner(language=language)
    
    input_path = Path(input_dir)
    output_dir = Path(DATA_DIR) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all supported files
    all_files = [f for f in input_path.rglob('*') if f.is_file() and converter.is_supported(str(f))]
    print(f"  Found {len(all_files)} supported files")
    
    if not all_files:
        raise ValueError(f"No supported files found in {input_dir}")
    
    records = []
    year_counts = {}
    
    for file_path in tqdm(all_files, desc="Processing files"):
        try:
            # Extract text using dataclean
            text = converter.extract_text(str(file_path))
            
            if not text or len(text) < 50:
                continue
            
            # Clean text using dataclean
            cleaned_text = cleaner.clean_text(text)
            
            if len(cleaned_text) < 30:
                continue
            
            # Extract year
            year_matches = re.findall(r'20\d{2}', str(file_path))
            year = int(year_matches[-1]) if year_matches else 2020
            
            # Extract province (directory name)
            parts = file_path.relative_to(input_path).parts
            province = parts[0] if parts else "Unknown"
            
            records.append({
                'cleaned_content': cleaned_text,
                'text': text,
                'year': year,
                'timestamp': f"{year}-01-01",
                'province': province,
                'title': file_path.stem,
                'source_file': str(file_path.relative_to(input_path))
            })
            
            year_counts[year] = year_counts.get(year, 0) + 1
            
        except Exception as e:
            print(f"  [Warning] Cannot process {file_path}: {e}")
            continue
    
    print(f"\n  Successfully processed {len(records)} documents")
    print(f"  Year distribution:")
    for year in sorted(year_counts.keys())[:10]:
        print(f"    {year}: {year_counts[year]} docs")
    if len(year_counts) > 10:
        print(f"    ... (total {len(year_counts)} years)")
    
    # Save CSV
    df = pd.DataFrame(records)
    output_csv = output_dir / f'{dataset}_cleaned.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n  Saved to: {output_csv}")
    
    return output_csv


def run_dataclean(raw_input: str, dataset: str, language: str = 'english', job_id: Optional[str] = None) -> Path:
    """
    Run data cleaning, convert raw text to cleaned CSV
    
    Args:
        raw_input: Raw data path (file or directory)
        dataset: Dataset name
        language: Language ('english' or 'chinese')
    
    Returns:
        Path to cleaned CSV file
    """
    print(f"\n[Data Cleaning] Input: {raw_input}, Language: {language}")
    
    # Import dataclean module
    sys.path.insert(0, str(Path(__file__).parent / 'dataclean'))
    from dataclean.src.converter import TextConverter
    from dataclean.src.cleaner import TextCleaner
    from dataclean.src.consolidator import DataConsolidator
    
    # Initialize components
    converter = TextConverter()
    cleaner = TextCleaner(language=language)
    consolidator = DataConsolidator()
    
    # Output path - with job_id support
    if job_id:
        output_dir = Path(DATA_DIR) / f"job_{job_id}"
    else:
        output_dir = Path(DATA_DIR) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename
    if job_id:
        output_csv = output_dir / 'data.csv'
    else:
        output_csv = output_dir / f'{dataset}_cleaned.csv'
    
    # Get files to process
    raw_path = Path(raw_input)
    if raw_path.is_dir():
        files = []
        for root, _, filenames in os.walk(raw_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if converter.is_supported(file_path):
                    files.append(file_path)
        print(f"  Found {len(files)} supported files")
    elif raw_path.suffix == '.csv':
        # If input is CSV, clean directly
        print(f"  Input is CSV file, performing text cleaning...")
        df = pd.read_csv(raw_path)
        
        # Find text column
        text_col = None
        for col in ['text', 'content', 'Text', 'Content', 'cleaned_content', 'raw_text']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            # Use first string column
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
        
        if text_col is None:
            raise ValueError(f"Cannot find text column, columns: {df.columns.tolist()}")
        
        print(f"  Using text column: {text_col}")
        
        # Clean text
        cleaned_texts = []
        for text in tqdm(df[text_col].fillna('').astype(str), desc="Cleaning text"):
            cleaned = cleaner.clean_text(text)
            cleaned_texts.append(cleaned)
        
        # Save
        result_df = pd.DataFrame({
            'cleaned_content': cleaned_texts
        })
        
        # Keep other columns
        for col in df.columns:
            if col != text_col and col not in result_df.columns:
                result_df[col] = df[col]
        
        result_df.to_csv(output_csv, index=False)
        print(f"  Cleaning completed: {output_csv}")
        return output_csv
    else:
        files = [str(raw_path)]
    
    # Process file list
    csv_path = consolidator.create_oneline_csv(
        files,
        str(output_csv),
        converter.extract_text,
        lambda text: cleaner.clean_text(text)
    )
    
    print(f"  Cleaning completed: {csv_path}")
    return Path(csv_path)


def find_data_file(dataset: str, job_id: Optional[str] = None) -> Optional[Path]:
    """Find CSV file for dataset"""
    # If job_id is provided, look in job-specific directory first
    if job_id:
        job_data_dir = Path(DATA_DIR) / f"job_{job_id}"
        
        # Possible filenames in job directory
        job_possible_names = [
            'data.csv',
            f'{dataset}.csv',
            f'{dataset}_cleaned.csv',
            'cleaned.csv',
            'train.csv',
        ]
        
        for name in job_possible_names:
            path = job_data_dir / name
            if path.exists():
                return path
        
        # Search for any CSV file in job directory
        job_csv_files = list(job_data_dir.glob('*.csv'))
        if job_csv_files:
            return job_csv_files[0]
    
    # Fall back to original dataset directory
    data_dir = Path(DATA_DIR) / dataset
    
    # Possible filenames
    possible_names = [
        f'{dataset}_cleaned.csv',
        f'{dataset}.csv',
        'cleaned.csv',
        'data.csv',
        'train.csv',
    ]
    
    for name in possible_names:
        path = data_dir / name
        if path.exists():
            return path
    
    # Search for any CSV file
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        return csv_files[0]
    
    return None


def load_texts(data_path: Path) -> Tuple[List[str], Optional[np.ndarray]]:
    """Load text data"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Find text column
    text_col = None
    for col in ['cleaned_content', 'clean_text', 'cleaned_text', 'text', 'content', 'Text']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")
    
    texts = df[text_col].fillna('').astype(str).tolist()
    
    # Find label column
    labels = None
    for col in ['label', 'Label', 'labels', 'category']:
        if col in df.columns:
            labels = df[col].values
            break
    
    print(f"Loaded {len(texts)} documents, text_col={text_col}")
    return texts, labels


def generate_bow(texts: List[str], vocab_size: int, output_dir: Path) -> Tuple[sp.csr_matrix, List[str]]:
    """Generate BOW matrix and vocabulary"""
    from bow.vocab_builder import VocabBuilder, VocabConfig
    from bow.bow_generator import BOWGenerator
    
    print(f"\n[Generating BOW] vocab_size={vocab_size}")
    
    vocab_config = VocabConfig(
        max_vocab_size=vocab_size,
        min_df=5,
        max_df_ratio=0.7
    )
    vocab_builder = VocabBuilder(config=vocab_config)
    vocab_builder.add_documents(texts, dataset_name="dataset")
    vocab_builder.build_vocab()
    
    bow_generator = BOWGenerator(vocab_builder)
    bow_output = bow_generator.generate_bow(texts, dataset_name="dataset")
    
    vocab = vocab_builder.get_vocab_list()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(output_dir / 'bow_matrix.npz', bow_output.bow_matrix)
    
    with open(output_dir / 'vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))
    
    with open(output_dir / 'vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)
    
    print(f"  ✓ BOW shape: {bow_output.bow_matrix.shape}")
    print(f"  ✓ Saved to {output_dir}")
    
    return bow_output.bow_matrix, vocab


def generate_qwen_embeddings(
    texts: List[str],
    labels: Optional[np.ndarray],
    model_size: str,
    mode: str,
    output_dir: Path,
    batch_size: int = 32
) -> np.ndarray:
    """Generate Qwen document embeddings"""
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    model_path = get_qwen_model_path(model_size)
    if not Path(model_path).exists():
        raise ValueError(f"Qwen model not found: {model_path}")
    
    print(f"\n[Generating Qwen Embedding] model={model_size}, mode={mode}")
    print(f"  Model path: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    # Generate embeddings
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            
            # Use CLS token or mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                batch_emb = outputs[0][:, 0, :].cpu().numpy()
            
            embeddings.append(batch_emb)
    
    embeddings = np.vstack(embeddings)
    
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    emb_filename = f'{mode}_embeddings_{timestamp}.npy' if model_size != '0.6B' else f'embeddings.npy'
    np.save(output_dir / emb_filename, embeddings)
    
    if labels is not None:
        label_filename = f'{mode}_labels_{timestamp}.npy' if model_size != '0.6B' else f'labels.npy'
        np.save(output_dir / label_filename, labels)
    
    # Save metadata
    metadata = {
        'num_documents': len(texts),
        'embedding_dim': embeddings.shape[1],
        'model_size': model_size,
        'mode': mode,
        'timestamp': timestamp
    }
    meta_filename = f'{mode}_metadata_{timestamp}.json' if model_size != '0.6B' else 'metadata.json'
    with open(output_dir / meta_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    print(f"  ✓ Saved to {output_dir}")
    
    # Clean up GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return embeddings


def generate_vocab_embeddings(
    vocab: List[str],
    model_size: str,
    output_dir: Path,
    batch_size: int = 64
) -> np.ndarray:
    """Generate vocabulary embeddings"""
    from model.vocab_embedder import VocabEmbedder
    
    model_path = get_qwen_model_path(model_size)
    
    print(f"\n[Generating Vocab Embedding] vocab_size={len(vocab)}")
    
    embedder = VocabEmbedder(
        model_path=model_path,
        batch_size=batch_size,
        normalize=True
    )
    
    embeddings = embedder.embed_vocab(vocab)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'vocab_embeddings.npy', embeddings)
    
    print(f"  ✓ Vocab embeddings shape: {embeddings.shape}")
    print(f"  ✓ Saved to {output_dir}")
    
    return embeddings


def generate_sbert_embeddings(texts: List[str], output_dir: Path, batch_size: int = 32) -> np.ndarray:
    """Generate SBERT embeddings (CTM/DTM specific)"""
    from sentence_transformers import SentenceTransformer
    
    print(f"\n[Generating SBERT Embedding]")
    
    # Use local SBERT model with environment variable support
    local_sbert_path = get_sbert_model_path()
    if Path(local_sbert_path).exists():
        print(f"  Using local model: {local_sbert_path}")
        model = SentenceTransformer(local_sbert_path)
    else:
        print(f"  Downloading online model: all-MiniLM-L6-v2")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'sbert_embeddings.npy', embeddings)
    
    print(f"  ✓ SBERT embeddings shape: {embeddings.shape}")
    print(f"  ✓ Saved to {output_dir}")
    
    return embeddings


def prepare_theta_data(args):
    """Prepare data required for THETA model"""
    dataset = args.dataset
    model_size = args.model_size
    mode = args.mode
    job_id = args.job_id
    
    print(f"\n{'='*70}")
    if job_id:
        print(f"[THETA] Preparing data: {dataset}, model={model_size}, mode={mode}, job_id={job_id}")
    else:
        print(f"[THETA] Preparing data: {dataset}, model={model_size}, mode={mode}")
    print(f"{'='*70}")
    
    # Find data file
    data_path = find_data_file(dataset, job_id)
    if data_path is None:
        if job_id:
            print(f"  Data file not found: {DATA_DIR}/job_{job_id}/ or {DATA_DIR}/{dataset}/")
        else:
            print(f"  Data file not found: {DATA_DIR}/{dataset}/")
        return False
    
    # Load texts
    texts, labels = load_texts(data_path)
    
    # Output directory - with job_id support
    if job_id:
        # Use job_id in the path structure
        result_base = Path(RESULT_DIR) / f"job_{job_id}"
    else:
        # Original path structure
        result_base = Path(RESULT_DIR) / model_size / dataset
    
    # Create directories for results
    bow_dir = result_base / model_size / 'bow'
    
    if model_size == '0.6B':
        emb_dir = result_base / model_size / mode / 'embeddings'
    else:
        emb_dir = result_base / model_size / 'embedding'
    
    # 1. Generate BOW
    if not (bow_dir / 'bow_matrix.npz').exists() or not args.bow_only:
        bow_matrix, vocab = generate_bow(texts, args.vocab_size, bow_dir)
    else:
        print(f"\n[Skip] BOW already exists: {bow_dir}")
        with open(bow_dir / 'vocab.txt', 'r') as f:
            vocab = f.read().strip().split('\n')
    
    if args.bow_only:
        print("\n[Done] Only generated BOW")
        return True
    
    # 2. Generate document embeddings
    generate_qwen_embeddings(texts, labels, model_size, mode, emb_dir, args.batch_size)
    
    # 3. Generate vocabulary embeddings
    generate_vocab_embeddings(vocab, model_size, bow_dir, args.batch_size)
    
    print(f"\n{'='*70}")
    print(f"[Done] THETA data preparation completed")
    print(f"  - BOW: {bow_dir}")
    print(f"  - Embeddings: {emb_dir}")
    print(f"{'='*70}")
    
    return True


def prepare_baseline_data(args):
    """Prepare data required for baseline models"""
    dataset = args.dataset
    job_id = args.job_id
    
    print(f"\n{'='*70}")
    if job_id:
        print(f"[Baseline] Preparing data: {dataset}, job_id={job_id}")
    else:
        print(f"[Baseline] Preparing data: {dataset}")
    print(f"{'='*70}")
    
    # Find data file
    data_path = find_data_file(dataset, job_id)
    if data_path is None:
        if job_id:
            print(f"  Data file not found: {DATA_DIR}/job_{job_id}/ or {DATA_DIR}/{dataset}/")
        else:
            print(f"  Data file not found: {DATA_DIR}/{dataset}/")
        return False
    
    # Load texts
    texts, labels = load_texts(data_path)
    
    # Output directory - with job_id support
    if job_id:
        result_dir = Path(RESULT_DIR) / f"job_{job_id}" / 'baseline'
    else:
        result_dir = Path(RESULT_DIR) / 'baseline' / dataset
    
    # 1. Generate BOW
    bow_matrix, vocab = generate_bow(texts, args.vocab_size, result_dir)
    
    if args.bow_only:
        print("\n[Done] Only generated BOW")
        return True
    
    # 2. Generate SBERT embeddings (CTM specific)
    try:
        generate_sbert_embeddings(texts, result_dir, args.batch_size)
    except Exception as e:
        print(f"  [Warning] SBERT generation failed: {e}")
        print(f"  CTM may not work, but LDA and ETM can run normally")
    
    print(f"\n{'='*70}")
    print(f"[Done] Baseline data preparation completed")
    print(f"  - Output directory: {result_dir}")
    print(f"{'='*70}")
    
    return True


def prepare_dtm_data(args):
    """Prepare data required for DTM model (with time slice information)"""
    dataset = args.dataset
    time_column = args.time_column
    job_id = args.job_id
    
    print(f"\n{'='*70}")
    if job_id:
        print(f"[DTM] Preparing data: {dataset}, job_id={job_id}")
    else:
        print(f"[DTM] Preparing data: {dataset}")
    print(f"{'='*70}")
    
    # Find data file
    data_path = find_data_file(dataset, job_id)
    if data_path is None:
        if job_id:
            print(f"  Data file not found: {DATA_DIR}/job_{job_id}/ or {DATA_DIR}/{dataset}/")
        else:
            print(f"  Data file not found: {DATA_DIR}/{dataset}/")
        return False
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Find text column
    text_col = None
    for col in ['cleaned_content', 'clean_text', 'cleaned_text', 'text', 'content', 'Text']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")
    
    texts = df[text_col].fillna('').astype(str).tolist()
    
    # Find time column
    if time_column not in df.columns:
        # Try other possible time column names
        for col in ['year', 'timestamp', 'date', 'time', 'Year', 'Date']:
            if col in df.columns:
                time_column = col
                break
    
    if time_column not in df.columns:
        print(f"  Time column '{time_column}' not found")
        print(f"  Available columns: {df.columns.tolist()}")
        print(f"  DTM requires time information. Please ensure the CSV contains a time column")
        return False
    
    # Extract time information
    time_values = df[time_column].values
    
    # Convert to year (if date format)
    try:
        if df[time_column].dtype == 'object':
            # Try to parse date
            parsed_dates = pd.to_datetime(df[time_column], errors='coerce')
            if parsed_dates.notna().sum() > len(df) * 0.5:
                time_values = parsed_dates.dt.year.fillna(2020).astype(int).values
            else:
                # May already be year
                time_values = pd.to_numeric(df[time_column], errors='coerce').fillna(2020).astype(int).values
        else:
            time_values = df[time_column].fillna(2020).astype(int).values
    except Exception as e:
        print(f"  [Warning] Time parsing failed: {e}")
        time_values = np.zeros(len(df), dtype=int)
    
    # Calculate time slices
    unique_times = sorted(set(time_values))
    num_time_slices = args.time_slices if args.time_slices else len(unique_times)
    
    # Create time-to-index mapping
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    time_indices = np.array([time_to_idx.get(t, 0) for t in time_values])
    
    print(f"  Documents: {len(texts)}")
    print(f"  Time range: {min(unique_times)} - {max(unique_times)}")
    print(f"  Time slices: {num_time_slices}")
    print(f"  Time distribution:")
    for t in unique_times[:10]:  # Only show first 10
        count = (time_values == t).sum()
        print(f"    {t}: {count} docs")
    if len(unique_times) > 10:
        print(f"    ... (total {len(unique_times)} time points)")
    
    # Output directory - with job_id support
    if job_id:
        result_dir = Path(RESULT_DIR) / f"job_{job_id}" / 'baseline'
    else:
        result_dir = Path(RESULT_DIR) / 'baseline' / dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate BOW
    bow_matrix, vocab = generate_bow(texts, args.vocab_size, result_dir)
    
    # 2. Save time slice information
    time_info = {
        'time_column': time_column,
        'unique_times': [int(t) for t in unique_times],
        'num_time_slices': num_time_slices,
        'time_to_idx': {str(k): v for k, v in time_to_idx.items()},
        'documents_per_time': {str(t): int((time_values == t).sum()) for t in unique_times}
    }
    
    with open(result_dir / 'time_slices.json', 'w', encoding='utf-8') as f:
        json.dump(time_info, f, ensure_ascii=False, indent=2)
    
    np.save(result_dir / 'time_indices.npy', time_indices)
    
    print(f"\n  Time slice info saved to: {result_dir / 'time_slices.json'}")
    print(f"  Time indices saved to: {result_dir / 'time_indices.npy'}")
    
    if args.bow_only:
        print("\n[Done] Only generated BOW and time slice info")
        return True
    
    # 3. Generate SBERT embeddings (optional, for initialization)
    try:
        generate_sbert_embeddings(texts, result_dir, args.batch_size)
    except Exception as e:
        print(f"  [Warning] SBERT generation failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"[Done] DTM data preparation completed")
    print(f"  - Output directory: {result_dir}")
    print(f"  - Time slices: {num_time_slices}")
    print(f"{'='*70}")
    
    return True


def check_files(args):
    """Check data file status"""
    dataset = args.dataset
    job_id = args.job_id
    
    print(f"\n{'='*70}")
    if job_id:
        print(f"Data file check: {dataset}, job_id={job_id}")
    else:
        print(f"Data file check: {dataset}")
    print(f"{'='*70}")
    
    # Check raw data
    data_path = find_data_file(dataset, job_id)
    print(f"\n[Raw Data]")
    if data_path:
        size_mb = data_path.stat().st_size / 1024 / 1024
        print(f"  ✓ {data_path} ({size_mb:.2f} MB)")
    else:
        if job_id:
            print(f"  Data file not found: {DATA_DIR}/job_{job_id}/ or {DATA_DIR}/{dataset}/")
        else:
            print(f"  Data file not found: {DATA_DIR}/{dataset}/")
    
    # Check THETA data
    if args.model == 'theta':
        model_size = args.model_size
        mode = args.mode
        
        # Output directory - with job_id support
        if job_id:
            result_base = Path(RESULT_DIR) / f"job_{job_id}" / model_size
        else:
            result_base = Path(RESULT_DIR) / model_size / dataset
        
        print(f"\n[THETA {model_size} - {mode}]")
        
        files = {
            'bow_matrix': result_base / 'bow' / 'bow_matrix.npz',
            'vocab': result_base / 'bow' / 'vocab.txt',
            'vocab_embeddings': result_base / 'bow' / 'vocab_embeddings.npy',
        }
        
        # Embedding path
        if model_size == '0.6B':
            emb_path = result_base / mode / 'embeddings' / f'{dataset}_{mode}_embeddings.npy'
        else:
            emb_dir = result_base / 'embedding'
            emb_path = None
            if emb_dir.exists():
                for f in emb_dir.glob(f'{mode}_embeddings_*.npy'):
                    emb_path = f
                    break
        files['embeddings'] = emb_path if emb_path else result_base / mode / 'embeddings' / 'embeddings.npy'
        
        for name, path in files.items():
            if path and path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"  ✓ {name}: {size_mb:.2f} MB")
            else:
                print(f"  {name}: missing")
                if path:
                    print(f"      Path: {path}")
    
    # Check Baseline/DTM data
    elif args.model in ['baseline', 'dtm']:
        # Output directory - with job_id support
        if job_id:
            result_dir = Path(RESULT_DIR) / f"job_{job_id}" / 'baseline'
        else:
            result_dir = Path(RESULT_DIR) / 'baseline' / dataset
        
        model_label = "DTM" if args.model == 'dtm' else "Baseline"
        print(f"\n[{model_label}]")
        
        files = {
            'bow_matrix': result_dir / 'bow_matrix.npz',
            'vocab': result_dir / 'vocab.json',
            'sbert_embeddings': result_dir / 'sbert_embeddings.npy',
        }
        
        # DTM additional check for time slice info
        if args.model == 'dtm':
            files['time_slices'] = result_dir / 'time_slices.json'
            files['time_indices'] = result_dir / 'time_indices.npy'
        
        for name, path in files.items():
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                print(f"  ✓ {name}: {size_mb:.2f} MB")
            else:
                print(f"  {name}: missing")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Generate job_id if not provided
    if args.job_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(2).hex()
        args.job_id = f"job_{timestamp}_{random_suffix}"
        print(f"\n[Info] Auto-generated job_id: {args.job_id}")
    
    if args.check_only:
        check_files(args)
        return
    
    # If data cleaning is needed first
    if args.clean:
        if args.raw_input is None:
            print("[Error] --raw-input parameter is required when using --clean")
            print("Example: python prepare_data.py --dataset my_data --model theta --clean --raw-input /path/to/raw_data.csv")
            return
        
        raw_path = Path(args.raw_input)
        # Check if it's a docx directory
        if raw_path.is_dir() and list(raw_path.rglob('*.docx')):
            process_docx_directory(args.raw_input, args.dataset, args.language, args.job_id)
        else:
            run_dataclean(args.raw_input, args.dataset, args.language, args.job_id)
    
    if args.model == 'theta':
        prepare_theta_data(args)
    elif args.model == 'dtm':
        prepare_dtm_data(args)
    else:
        prepare_baseline_data(args)


if __name__ == '__main__':
    main()

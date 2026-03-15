"""
Phase 3: The Pipeline - Matrix Generation

Transforms mapped columns into training matrices:
1. Dropna: Remove rows with null values in mapped columns
2. Sort: Sort by time column (ascending) if available
3. Transform:
   - Text → bow_matrix.npy
   - Time → time_slices.json & time_indices.npy
   - Covariates → One-hot Encoding → covariates.npy
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import Counter

from .column_mapper import MappingConfig


class MatrixPipeline:
    """Generates training matrices from mapped CSV data."""
    
    def __init__(
        self,
        mapping_config: MappingConfig,
        output_dir: str,
        vocab_size: int = 5000,
        min_doc_freq: int = 3,
        max_doc_freq_ratio: float = 0.95,
        language: str = 'auto'
    ):
        """
        Initialize pipeline.
        
        Args:
            mapping_config: Column mapping configuration
            output_dir: Output directory for matrices
            vocab_size: Maximum vocabulary size
            min_doc_freq: Minimum document frequency for words
            max_doc_freq_ratio: Maximum document frequency ratio
            language: Language for text processing ('auto', 'chinese', 'english')
        """
        self.mapping_config = mapping_config
        self.output_dir = Path(output_dir)
        self.vocab_size = vocab_size
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq_ratio = max_doc_freq_ratio
        self.language = language
        
        # Results
        self.df = None
        self.vocab = None
        self.bow_matrix = None
        self.time_slices = None
        self.time_indices = None
        self.covariates = None
        self.covariate_names = None
        
        # Statistics
        self.stats = {
            'original_rows': 0,
            'after_dropna': 0,
            'vocab_size': 0,
            'time_slices': 0,
            'covariate_dims': 0,
        }
    
    def load_data(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """Load CSV data."""
        encodings = [encoding, 'utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for enc in encodings:
            try:
                self.df = pd.read_csv(file_path, encoding=enc)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        if self.df is None:
            raise ValueError("Could not read CSV file")
        
        self.stats['original_rows'] = len(self.df)
        return self.df
    
    def step1_dropna(self) -> pd.DataFrame:
        """
        Step 1: Remove rows with null values in mapped columns.
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        # Columns to check for nulls
        columns_to_check = [self.mapping_config.text_column]
        
        if self.mapping_config.time_column:
            columns_to_check.append(self.mapping_config.time_column)
        
        columns_to_check.extend(self.mapping_config.covariate_columns)
        
        # Filter out columns that don't exist
        columns_to_check = [c for c in columns_to_check if c in self.df.columns]
        
        # Drop rows with nulls
        before_count = len(self.df)
        self.df = self.df.dropna(subset=columns_to_check)
        after_count = len(self.df)
        
        self.stats['after_dropna'] = after_count
        print(f"  Dropna: {before_count} → {after_count} rows ({before_count - after_count} removed)")
        
        return self.df
    
    def step2_sort(self) -> pd.DataFrame:
        """
        Step 2: Sort by time column (ascending) if available.
        
        Returns:
            Sorted DataFrame
        """
        if self.df is None:
            raise RuntimeError("No data loaded.")
        
        if self.mapping_config.time_column:
            time_col = self.mapping_config.time_column
            
            # Try to convert to datetime or numeric for proper sorting
            try:
                # Try datetime first
                self.df[time_col] = pd.to_datetime(self.df[time_col])
            except:
                try:
                    # Try numeric
                    self.df[time_col] = pd.to_numeric(self.df[time_col])
                except:
                    pass  # Keep as-is
            
            self.df = self.df.sort_values(by=time_col, ascending=True)
            self.df = self.df.reset_index(drop=True)
            print(f"  Sort: Sorted by '{time_col}' ascending")
        else:
            print("  Sort: No time column, skipping sort")
        
        return self.df
    
    def step3_transform_text(self) -> Tuple[np.ndarray, List[str]]:
        """
        Step 3a: Transform text to BOW matrix.
        
        Returns:
            (bow_matrix, vocab)
        """
        if self.df is None:
            raise RuntimeError("No data loaded.")
        
        text_col = self.mapping_config.text_column
        texts = self.df[text_col].astype(str).tolist()
        
        # Detect language if auto
        if self.language == 'auto':
            self.language = self._detect_language(texts[:100])
            print(f"  Detected language: {self.language}")
        
        # Tokenize
        tokenized = self._tokenize_texts(texts)
        
        # Build vocabulary
        self.vocab = self._build_vocab(tokenized)
        self.stats['vocab_size'] = len(self.vocab)
        
        # Build BOW matrix
        word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        rows, cols, data = [], [], []
        for doc_idx, tokens in enumerate(tokenized):
            word_counts = Counter(tokens)
            for word, count in word_counts.items():
                if word in word_to_idx:
                    rows.append(doc_idx)
                    cols.append(word_to_idx[word])
                    data.append(count)
        
        self.bow_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(texts), len(self.vocab)),
            dtype=np.float32
        )
        
        print(f"  BOW matrix: {self.bow_matrix.shape}")
        
        return self.bow_matrix, self.vocab
    
    def _detect_language(self, texts: List[str]) -> str:
        """Detect language from sample texts."""
        chinese_chars = 0
        total_chars = 0
        
        for text in texts:
            for char in str(text):
                if '\u4e00' <= char <= '\u9fff':
                    chinese_chars += 1
                total_chars += 1
        
        if total_chars > 0 and chinese_chars / total_chars > 0.1:
            return 'chinese'
        return 'english'
    
    def _tokenize_texts(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts based on language."""
        if self.language == 'chinese':
            try:
                import jieba
                return [list(jieba.cut(str(text))) for text in texts]
            except ImportError:
                print("  Warning: jieba not installed, using character-level tokenization")
                return [[c for c in str(text) if c.strip()] for text in texts]
        else:
            # English: simple whitespace tokenization with lowercasing
            import re
            tokenized = []
            for text in texts:
                tokens = re.findall(r'\b[a-zA-Z]{2,}\b', str(text).lower())
                tokenized.append(tokens)
            return tokenized
    
    def _build_vocab(self, tokenized: List[List[str]]) -> List[str]:
        """Build vocabulary from tokenized texts."""
        # Count document frequency
        doc_freq = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))
        
        num_docs = len(tokenized)
        max_doc_freq = int(num_docs * self.max_doc_freq_ratio)
        
        # Filter by frequency
        valid_words = [
            word for word, freq in doc_freq.items()
            if self.min_doc_freq <= freq <= max_doc_freq
        ]
        
        # Sort by frequency and take top vocab_size
        valid_words.sort(key=lambda w: doc_freq[w], reverse=True)
        vocab = valid_words[:self.vocab_size]
        
        print(f"  Vocabulary: {len(vocab)} words (from {len(doc_freq)} unique)")
        
        return vocab
    
    def step3_transform_time(self) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """
        Step 3b: Transform time column to time slices.
        
        Returns:
            (time_slices_info, time_indices)
        """
        if not self.mapping_config.time_column:
            print("  Time: No time column, skipping")
            return None, None
        
        time_col = self.mapping_config.time_column
        time_values = self.df[time_col].values
        
        # Extract time periods (year or date)
        time_periods = self._extract_time_periods(time_values)
        
        # Get unique time periods
        unique_times = sorted(set(time_periods))
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        
        # Create time indices
        self.time_indices = np.array([time_to_idx[t] for t in time_periods], dtype=np.int32)
        
        # Count documents per time slice
        docs_per_time = Counter(time_periods)
        
        self.time_slices = {
            'time_column': time_col,
            'unique_times': [str(t) for t in unique_times],
            'num_time_slices': len(unique_times),
            'time_to_idx': {str(k): v for k, v in time_to_idx.items()},
            'documents_per_time': {str(k): v for k, v in docs_per_time.items()},
        }
        
        self.stats['time_slices'] = len(unique_times)
        print(f"  Time slices: {len(unique_times)} periods")
        
        return self.time_slices, self.time_indices
    
    def _extract_time_periods(self, time_values: np.ndarray) -> List:
        """Extract time periods from values."""
        periods = []
        
        for val in time_values:
            if pd.isna(val):
                periods.append(None)
                continue
            
            # Try to extract year
            if isinstance(val, (int, float)):
                periods.append(int(val))
            elif hasattr(val, 'year'):
                periods.append(val.year)
            else:
                # Try to parse as date string
                try:
                    import re
                    match = re.search(r'(\d{4})', str(val))
                    if match:
                        periods.append(int(match.group(1)))
                    else:
                        periods.append(str(val))
                except:
                    periods.append(str(val))
        
        return periods
    
    def step3_transform_covariates(self) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        Step 3c: Transform covariate columns to one-hot encoding.
        
        Returns:
            (covariates_matrix, covariate_names)
        """
        if not self.mapping_config.covariate_columns:
            print("  Covariates: No covariate columns, skipping")
            return None, None
        
        covariate_cols = self.mapping_config.covariate_columns
        
        # One-hot encode each covariate column
        encoded_parts = []
        self.covariate_names = []
        
        for col in covariate_cols:
            if col not in self.df.columns:
                continue
            
            values = self.df[col].astype(str)
            unique_values = sorted(values.unique())
            
            # Create one-hot encoding
            for val in unique_values:
                encoded_parts.append((values == val).astype(np.float32).values)
                self.covariate_names.append(f"{col}_{val}")
        
        if encoded_parts:
            self.covariates = np.column_stack(encoded_parts)
            self.stats['covariate_dims'] = self.covariates.shape[1]
            print(f"  Covariates: {self.covariates.shape[1]} dimensions from {len(covariate_cols)} columns")
        else:
            self.covariates = None
        
        return self.covariates, self.covariate_names
    
    def run(self, file_path: str) -> Dict[str, Any]:
        """
        Run complete pipeline.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Pipeline result with file paths
        """
        print(f"\n{'='*60}")
        print("Matrix Pipeline")
        print(f"{'='*60}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("\n[Loading Data]")
        self.load_data(file_path)
        print(f"  Loaded {self.stats['original_rows']} rows")
        
        # Step 1: Dropna
        print("\n[Step 1: Dropna]")
        self.step1_dropna()
        
        # Step 2: Sort
        print("\n[Step 2: Sort]")
        self.step2_sort()
        
        # Step 3: Transform
        print("\n[Step 3: Transform]")
        
        # 3a: Text → BOW
        print("  [3a: Text → BOW]")
        self.step3_transform_text()
        
        # 3b: Time → Time slices
        print("  [3b: Time → Time slices]")
        self.step3_transform_time()
        
        # 3c: Covariates → One-hot
        print("  [3c: Covariates → One-hot]")
        self.step3_transform_covariates()
        
        # Save all matrices
        print("\n[Saving Matrices]")
        result = self.save_all()
        
        print(f"\n{'='*60}")
        print("Pipeline Complete")
        print(f"  Output: {self.output_dir}")
        print(f"  Documents: {self.stats['after_dropna']}")
        print(f"  Vocabulary: {self.stats['vocab_size']}")
        if self.stats['time_slices'] > 0:
            print(f"  Time slices: {self.stats['time_slices']}")
        if self.stats['covariate_dims'] > 0:
            print(f"  Covariate dims: {self.stats['covariate_dims']}")
        print(f"{'='*60}")
        
        return result
    
    def save_all(self) -> Dict[str, str]:
        """Save all matrices and return file paths."""
        result = {
            'output_dir': str(self.output_dir),
            'files': {},
        }
        
        # Save BOW matrix
        bow_path = self.output_dir / 'bow_matrix.npy'
        sp.save_npz(self.output_dir / 'bow_matrix.npz', self.bow_matrix)
        result['files']['bow_matrix'] = str(self.output_dir / 'bow_matrix.npz')
        print(f"  ✓ bow_matrix.npz")
        
        # Save vocabulary
        vocab_path = self.output_dir / 'vocab.json'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        result['files']['vocab'] = str(vocab_path)
        print(f"  ✓ vocab.json")
        
        # Save time slices
        if self.time_slices:
            ts_path = self.output_dir / 'time_slices.json'
            with open(ts_path, 'w', encoding='utf-8') as f:
                json.dump(self.time_slices, f, ensure_ascii=False, indent=2)
            result['files']['time_slices'] = str(ts_path)
            print(f"  ✓ time_slices.json")
            
            ti_path = self.output_dir / 'time_indices.npy'
            np.save(ti_path, self.time_indices)
            result['files']['time_indices'] = str(ti_path)
            print(f"  ✓ time_indices.npy")
        
        # Save covariates
        if self.covariates is not None:
            cov_path = self.output_dir / 'covariates.npy'
            np.save(cov_path, self.covariates)
            result['files']['covariates'] = str(cov_path)
            print(f"  ✓ covariates.npy")
            
            cov_names_path = self.output_dir / 'covariate_names.json'
            with open(cov_names_path, 'w', encoding='utf-8') as f:
                json.dump(self.covariate_names, f, ensure_ascii=False, indent=2)
            result['files']['covariate_names'] = str(cov_names_path)
            print(f"  ✓ covariate_names.json")
        
        # Save config
        config = {
            'mapping': self.mapping_config.to_dict(),
            'stats': self.stats,
            'vocab_size': len(self.vocab),
            'language': self.language,
            'created': datetime.now().isoformat(),
        }
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        result['files']['config'] = str(config_path)
        print(f"  ✓ config.json")
        
        result['stats'] = self.stats
        result['available_models'] = self.mapping_config.get_available_models()
        
        return result


def run_pipeline(
    csv_path: str,
    mapping_config: MappingConfig,
    output_dir: str,
    vocab_size: int = 5000,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run the pipeline.
    
    Args:
        csv_path: Path to CSV file
        mapping_config: Column mapping configuration
        output_dir: Output directory
        vocab_size: Maximum vocabulary size
        
    Returns:
        Pipeline result
    """
    pipeline = MatrixPipeline(
        mapping_config=mapping_config,
        output_dir=output_dir,
        vocab_size=vocab_size,
        **kwargs
    )
    return pipeline.run(csv_path)

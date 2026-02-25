"""
BOW (Bag-of-Words) Generator for Engine A.

Generates BOW matrices for each dataset using the global vocabulary.
BOW is used ONLY as the reconstruction target for ETM, NOT as input.
"""

import os
import json
import re
import numpy as np
from scipy import sparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .vocab_builder import VocabBuilder, VocabConfig


@dataclass
class BOWOutput:
    """Container for BOW generation results"""
    bow_matrix: sparse.csr_matrix  # Shape: (num_docs, vocab_size)
    dataset_name: str
    num_docs: int
    vocab_size: int
    total_tokens: int
    avg_doc_length: float
    sparsity: float


class BOWGenerator:
    """
    Generates BOW matrices for datasets using a shared vocabulary.
    
    The BOW matrix serves as the reconstruction target for ETM:
    - Input to ETM: Qwen document embeddings
    - Output/Target: BOW distribution
    - Loss: Reconstruction of BOW from topic distribution
    """
    
    def __init__(
        self,
        vocab_builder: VocabBuilder,
        dev_mode: bool = False
    ):
        """
        Initialize BOW generator.
        
        Args:
            vocab_builder: VocabBuilder with built vocabulary
            dev_mode: Print debug information
        """
        self.vocab = vocab_builder
        self.word2idx = vocab_builder.get_word2idx()
        self.vocab_size = vocab_builder.get_vocab_size()
        self.dev_mode = dev_mode
        
        # Use same tokenization config
        self.config = vocab_builder.config
        self.stopwords = vocab_builder.stopwords
        
        if self.dev_mode:
            print(f"[DEV] BOWGenerator initialized with vocab_size={self.vocab_size}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text (delegates to VocabBuilder's tokenizer for consistency).
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.vocab._tokenize(text)
    
    def generate_bow(
        self,
        texts: List[str],
        dataset_name: str,
        show_progress: bool = True,
        normalize: bool = False
    ) -> BOWOutput:
        """
        Generate BOW matrix for a dataset.
        
        Args:
            texts: List of text documents
            dataset_name: Name of the dataset
            show_progress: Show progress bar
            normalize: Whether to L1-normalize rows
            
        Returns:
            BOWOutput with BOW matrix and statistics
        """
        num_docs = len(texts)
        
        if self.dev_mode:
            print(f"[DEV] Generating BOW for {dataset_name}: {num_docs} documents")
        
        # Build sparse matrix using COO format for efficiency
        rows = []
        cols = []
        data = []
        total_tokens = 0
        
        iterator = enumerate(texts)
        if show_progress:
            iterator = tqdm(enumerate(texts), total=num_docs, desc=f"BOW {dataset_name}")
        
        for doc_idx, text in iterator:
            tokens = self._tokenize(text)
            
            # Count tokens in vocabulary
            token_counts = {}
            for token in tokens:
                if token in self.word2idx:
                    idx = self.word2idx[token]
                    token_counts[idx] = token_counts.get(idx, 0) + 1
                    total_tokens += 1
            
            # Add to sparse matrix data
            for word_idx, count in token_counts.items():
                rows.append(doc_idx)
                cols.append(word_idx)
                data.append(count)
        
        # Create sparse matrix
        bow_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_docs, self.vocab_size),
            dtype=np.float32
        )
        
        # Normalize if requested
        if normalize:
            row_sums = np.array(bow_matrix.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            bow_matrix = bow_matrix.multiply(1.0 / row_sums[:, np.newaxis])
        
        # Calculate statistics
        non_zero = bow_matrix.nnz
        total_elements = num_docs * self.vocab_size
        sparsity = 1.0 - (non_zero / total_elements)
        avg_doc_length = total_tokens / num_docs if num_docs > 0 else 0
        
        output = BOWOutput(
            bow_matrix=bow_matrix,
            dataset_name=dataset_name,
            num_docs=num_docs,
            vocab_size=self.vocab_size,
            total_tokens=total_tokens,
            avg_doc_length=avg_doc_length,
            sparsity=sparsity
        )
        
        if self.dev_mode:
            print(f"[DEV] BOW matrix shape: {bow_matrix.shape}")
            print(f"[DEV] Non-zero elements: {non_zero}")
            print(f"[DEV] Sparsity: {sparsity:.4f}")
            print(f"[DEV] Avg doc length: {avg_doc_length:.1f}")
        
        return output
    
    def save_bow(
        self,
        output: BOWOutput,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Save BOW matrix and metadata.
        
        Args:
            output: BOWOutput object
            output_dir: Output directory
            
        Returns:
            Dictionary with saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sparse matrix
        bow_path = os.path.join(output_dir, f"{output.dataset_name}_bow.npz")
        sparse.save_npz(bow_path, output.bow_matrix)
        
        # Save metadata
        metadata = {
            "dataset_name": output.dataset_name,
            "num_docs": output.num_docs,
            "vocab_size": output.vocab_size,
            "total_tokens": output.total_tokens,
            "avg_doc_length": output.avg_doc_length,
            "sparsity": output.sparsity
        }
        meta_path = os.path.join(output_dir, f"{output.dataset_name}_bow_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        paths = {
            "bow": bow_path,
            "metadata": meta_path
        }
        
        if self.dev_mode:
            print(f"[DEV] Saved BOW files:")
            for key, path in paths.items():
                print(f"[DEV]   {key}: {path}")
        
        return paths
    
    @staticmethod
    def load_bow(bow_path: str) -> Tuple[sparse.csr_matrix, Dict]:
        """
        Load BOW matrix from file.
        
        Args:
            bow_path: Path to .npz file
            
        Returns:
            (bow_matrix, metadata)
        """
        bow_matrix = sparse.load_npz(bow_path)
        
        # Load metadata if exists
        meta_path = bow_path.replace('_bow.npz', '_bow_meta.json')
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        return bow_matrix, metadata

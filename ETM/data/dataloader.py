"""
ETM Dataset and DataLoader utilities
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
from typing import Optional


class ETMDataset(Dataset):
    """
    Dataset for ETM training
    
    Args:
        doc_embeddings: Document embeddings (N x D)
        bow_matrix: Bag-of-words matrix (N x V), can be sparse or dense
        normalize_bow: Whether to normalize BOW to sum to 1
        dev_mode: Enable debug logging
    """
    
    def __init__(
        self,
        doc_embeddings: np.ndarray,
        bow_matrix,
        normalize_bow: bool = True,
        dev_mode: bool = False
    ):
        self.dev_mode = dev_mode
        
        # Store document embeddings
        self.doc_embeddings = torch.tensor(doc_embeddings, dtype=torch.float32)
        
        # Handle sparse or dense BOW matrix
        if sparse.issparse(bow_matrix):
            self.bow_matrix = bow_matrix.toarray()
        else:
            self.bow_matrix = bow_matrix
        
        self.bow_matrix = self.bow_matrix.astype(np.float32)
        
        # Normalize BOW if requested
        if normalize_bow:
            row_sums = self.bow_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            self.bow_matrix = self.bow_matrix / row_sums
        
        self.bow_matrix = torch.tensor(self.bow_matrix, dtype=torch.float32)
        
        assert len(self.doc_embeddings) == len(self.bow_matrix), \
            f"Mismatch: {len(self.doc_embeddings)} embeddings vs {len(self.bow_matrix)} BOW rows"
        
        if dev_mode:
            print(f"[ETMDataset] Loaded {len(self)} samples")
            print(f"[ETMDataset] Doc embedding dim: {self.doc_embeddings.shape[1]}")
            print(f"[ETMDataset] Vocab size: {self.bow_matrix.shape[1]}")
    
    def __len__(self):
        return len(self.doc_embeddings)
    
    def __getitem__(self, idx):
        return {
            'doc_embedding': self.doc_embeddings[idx],
            'bow': self.bow_matrix[idx]
        }

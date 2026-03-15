"""
ETM Dataset and DataLoader utilities
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from typing import Optional, Union


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
        keep_sparse: bool = False,
        dev_mode: bool = False
    ):
        self.dev_mode = dev_mode
        self.keep_sparse = keep_sparse
        
        # Store document embeddings
        self.doc_embeddings = torch.tensor(doc_embeddings, dtype=torch.float32)
        
        # Handle sparse or dense BOW matrix
        if sparse.issparse(bow_matrix):
            if keep_sparse:
                self.bow_matrix = bow_matrix
            else:
                self.bow_matrix = bow_matrix.toarray()
        else:
            self.bow_matrix = bow_matrix
        
        if not keep_sparse:
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


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    sampler = None
) -> DataLoader:
    """
    Create a DataLoader for ETM training.
    
    Args:
        dataset: ETMDataset or Subset of ETMDataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for CUDA
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        sampler: Optional sampler for distributed training
        
    Returns:
        DataLoader for ETM training
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # persistent_workers requires num_workers > 0
    if num_workers == 0:
        persistent_workers = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        sampler=sampler
    )

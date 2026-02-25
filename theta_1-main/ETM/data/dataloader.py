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
    Dataset for ETM training - 支持稀疏格式延迟转换
    
    Args:
        doc_embeddings: Document embeddings (N x D)
        bow_matrix: Bag-of-words matrix (N x V), can be sparse or dense
        normalize_bow: Whether to normalize BOW to sum to 1
        dev_mode: Enable debug logging
        keep_sparse: Whether to keep sparse format and convert on-the-fly (saves memory)
    """
    
    def __init__(
        self,
        doc_embeddings: np.ndarray,
        bow_matrix,
        normalize_bow: bool = True,
        dev_mode: bool = False,
        keep_sparse: bool = False
    ):
        self.dev_mode = dev_mode
        self.keep_sparse = keep_sparse
        self.normalize_bow = normalize_bow
        
        # Store document embeddings
        self.doc_embeddings = torch.tensor(doc_embeddings, dtype=torch.float32)
        
        # 保持稀疏格式，只在需要时转换单行
        if keep_sparse and sparse.issparse(bow_matrix):
            self.bow_matrix = bow_matrix.tocsr()  # 确保是CSR格式，行访问更快
            self.is_sparse = True
            self._vocab_size = bow_matrix.shape[1]
            if dev_mode:
                print(f"[ETMDataset] Keeping sparse format (CSR), shape: {bow_matrix.shape}")
        else:
            # 原有逻辑：转为dense
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
            self.is_sparse = False
            self._vocab_size = self.bow_matrix.shape[1]
        
        # Validate dimensions
        n_embeddings = len(self.doc_embeddings)
        n_bow = self.bow_matrix.shape[0]
        assert n_embeddings == n_bow, \
            f"Mismatch: {n_embeddings} embeddings vs {n_bow} BOW rows"
        
        if dev_mode:
            print(f"[ETMDataset] Loaded {len(self)} samples")
            print(f"[ETMDataset] Doc embedding dim: {self.doc_embeddings.shape[1]}")
            print(f"[ETMDataset] Vocab size: {self._vocab_size}")
            print(f"[ETMDataset] Sparse mode: {self.is_sparse}")
    
    def __len__(self):
        return len(self.doc_embeddings)
    
    def __getitem__(self, idx):
        if self.is_sparse:
            # 延迟转换：只在需要时转换单行
            bow_row = self.bow_matrix[idx].toarray().flatten().astype(np.float32)
            if self.normalize_bow:
                row_sum = bow_row.sum()
                if row_sum > 0:
                    bow_row = bow_row / row_sum
            bow = torch.tensor(bow_row, dtype=torch.float32)
        else:
            bow = self.bow_matrix[idx]
        
        return {
            'doc_embedding': self.doc_embeddings[idx],
            'bow': bow
        }
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return self._vocab_size


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    sampler=None
) -> DataLoader:
    """
    创建优化的DataLoader
    
    Args:
        dataset: PyTorch Dataset
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数（0表示主进程加载）
        pin_memory: 是否使用pinned memory加速GPU传输
        persistent_workers: 是否保持worker进程存活
        prefetch_factor: 每个worker预取的batch数
        drop_last: 是否丢弃最后不完整的batch
        sampler: 可选的采样器（用于DDP等）
        
    Returns:
        配置好的DataLoader实例
    """
    # num_workers > 0 时才能使用 persistent_workers 和 prefetch_factor
    use_persistent = persistent_workers and num_workers > 0
    use_prefetch = prefetch_factor if num_workers > 0 else None
    
    # If sampler is provided, shuffle must be False
    if sampler is not None:
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=use_prefetch,
        drop_last=drop_last,
        sampler=sampler
    )

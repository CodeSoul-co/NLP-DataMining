"""
Balanced Dataset Sampler for Joint Training

Solves data imbalance problem:
- mental_health: 1,000,000 samples
- germanCoal: 5,000 samples
- Ratio 200:1

Strategy: Ensure balanced sampling of each dataset within each epoch
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Dict
from collections import Counter


class BalancedDatasetSampler(Sampler):
    """
    Balanced dataset sampler
    
    Ensures balanced sample proportions from each dataset in every batch,
    preventing large datasets from "overwhelming" small datasets.
    
    Strategy:
    1. Calculate sampling weight for each dataset (log scaling)
    2. Oversample small datasets, downsample large datasets
    3. Ensure all datasets are adequately sampled in each epoch
    """
    
    def __init__(
        self,
        dataset_labels: List[int],
        strategy: str = "oversample",
        temperature: float = 0.5,
        seed: int = 42
    ):
        """
        Args:
            dataset_labels: Dataset ID for each sample (0, 1, 2, ...)
            strategy: Balancing strategy
                - "oversample": Oversample small datasets to the size of the largest dataset
                - "downsample": Downsample large datasets to the size of the smallest dataset
                - "weighted": Weighted sampling (recommended)
            temperature: Temperature parameter controlling balance degree (0=fully balanced, 1=original distribution)
            seed: Random seed
        """
        self.dataset_labels = np.array(dataset_labels)
        self.strategy = strategy
        self.temperature = temperature
        self.seed = seed
        
        # Count samples in each dataset
        self.dataset_counts = Counter(dataset_labels)
        self.num_datasets = len(self.dataset_counts)
        self.dataset_ids = sorted(self.dataset_counts.keys())
        
        # Create indices for each dataset
        self.dataset_indices = {}
        for dataset_id in self.dataset_ids:
            self.dataset_indices[dataset_id] = np.where(
                self.dataset_labels == dataset_id
            )[0]
        
        # Compute sampling strategy
        self._compute_sampling_strategy()
        
        print(f"\n[BalancedDatasetSampler] Initialization complete")
        print(f"  Strategy: {strategy}")
        print(f"  Number of datasets: {self.num_datasets}")
        print(f"  Original distribution:")
        for dataset_id in self.dataset_ids:
            count = self.dataset_counts[dataset_id]
            print(f"    Dataset {dataset_id}: {count:,} samples")
        print(f"  Samples per epoch after balancing: {len(self):,}")
    
    def _compute_sampling_strategy(self):
        """Compute sampling strategy"""
        if self.strategy == "oversample":
            # Oversample: Sample all datasets to the size of the largest dataset
            max_count = max(self.dataset_counts.values())
            self.samples_per_dataset = {
                dataset_id: max_count 
                for dataset_id in self.dataset_ids
            }
            self.epoch_length = max_count * self.num_datasets
            
        elif self.strategy == "downsample":
            # Downsample: Sample all datasets to the size of the smallest dataset
            min_count = min(self.dataset_counts.values())
            self.samples_per_dataset = {
                dataset_id: min_count 
                for dataset_id in self.dataset_ids
            }
            self.epoch_length = min_count * self.num_datasets
            
        elif self.strategy == "weighted":
            # Weighted sampling: Use log scaling for balance
            # Calculate sampling weight for each dataset
            counts = np.array([self.dataset_counts[i] for i in self.dataset_ids])
            
            # Log scaling: log(count + 1)
            log_counts = np.log(counts + 1)
            
            # Apply temperature parameter
            # temperature=0: Fully balanced (all datasets have same weight)
            # temperature=1: Keep original distribution
            if self.temperature == 0:
                weights = np.ones_like(log_counts)
            else:
                weights = log_counts ** (1 / self.temperature)
            
            # Normalize
            weights = weights / weights.sum()
            
            # Calculate number of samples for each dataset
            # Target: Total samples approximately equal to average of all datasets
            target_total = int(np.mean(counts) * self.num_datasets)
            self.samples_per_dataset = {
                dataset_id: int(weights[i] * target_total)
                for i, dataset_id in enumerate(self.dataset_ids)
            }
            self.epoch_length = sum(self.samples_per_dataset.values())
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        print(f"  Balanced distribution:")
        for dataset_id in self.dataset_ids:
            samples = self.samples_per_dataset[dataset_id]
            original = self.dataset_counts[dataset_id]
            ratio = samples / original
            print(f"    Dataset {dataset_id}: {samples:,} samples "
                  f"(original: {original:,}, sampling rate: {ratio:.2f}x)")
    
    def __iter__(self):
        """Generate sampling indices"""
        np.random.seed(self.seed)
        
        # Generate sampling indices for each dataset
        all_indices = []
        
        for dataset_id in self.dataset_ids:
            dataset_idx = self.dataset_indices[dataset_id]
            num_samples = self.samples_per_dataset[dataset_id]
            
            if num_samples <= len(dataset_idx):
                # Downsample: Random selection
                sampled = np.random.choice(
                    dataset_idx, 
                    size=num_samples, 
                    replace=False
                )
            else:
                # Oversample: Repeated sampling
                sampled = np.random.choice(
                    dataset_idx, 
                    size=num_samples, 
                    replace=True
                )
            
            all_indices.extend(sampled.tolist())
        
        # Shuffle all indices
        np.random.shuffle(all_indices)
        
        # Update random seed (different for each epoch)
        self.seed += 1
        
        return iter(all_indices)
    
    def __len__(self):
        """Return number of samples per epoch"""
        return self.epoch_length


class SimpleOversamplingDataset:
    """
    Simple oversampling dataset wrapper
    
    Directly duplicates samples from small datasets during data loading phase
    """
    
    def __init__(
        self,
        texts_by_dataset: Dict[str, List[str]],
        labels_by_dataset: Dict[str, List] = None,
        target_size: int = None,
        strategy: str = "max"
    ):
        """
        Args:
            texts_by_dataset: {dataset_name: [text1, text2, ...]}
            labels_by_dataset: {dataset_name: [label1, label2, ...]}
            target_size: Target size, None for automatic calculation
            strategy: "max" (largest dataset), "mean" (average), "median" (median)
        """
        self.texts_by_dataset = texts_by_dataset
        self.labels_by_dataset = labels_by_dataset or {}
        
        # Calculate target size
        sizes = [len(texts) for texts in texts_by_dataset.values()]
        if target_size is None:
            if strategy == "max":
                target_size = max(sizes)
            elif strategy == "mean":
                target_size = int(np.mean(sizes))
            elif strategy == "median":
                target_size = int(np.median(sizes))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        self.target_size = target_size
        
        print(f"\n[SimpleOversamplingDataset] Oversampling initialization")
        print(f"  Target size: {target_size:,}")
        print(f"  Original distribution:")
        
        # Oversample
        self.all_texts = []
        self.all_labels = []
        self.all_dataset_ids = []
        
        for dataset_id, (dataset_name, texts) in enumerate(texts_by_dataset.items()):
            original_size = len(texts)
            labels = self.labels_by_dataset.get(dataset_name, [None] * original_size)
            
            # Calculate number of times to duplicate
            repeat_times = int(np.ceil(target_size / original_size))
            
            # Oversample
            oversampled_texts = (texts * repeat_times)[:target_size]
            oversampled_labels = (labels * repeat_times)[:target_size]
            
            self.all_texts.extend(oversampled_texts)
            self.all_labels.extend(oversampled_labels)
            self.all_dataset_ids.extend([dataset_id] * target_size)
            
            print(f"    {dataset_name}: {original_size:,} -> {target_size:,} "
                  f"(duplicated {repeat_times}x)")
        
        print(f"  Total samples: {len(self.all_texts):,}")
    
    def get_data(self):
        """Return balanced data"""
        return {
            'texts': self.all_texts,
            'labels': self.all_labels if any(l is not None for l in self.all_labels) else None,
            'dataset_ids': self.all_dataset_ids
        }


def create_balanced_dataloader(
    texts_by_dataset: Dict[str, List[str]],
    labels_by_dataset: Dict[str, List] = None,
    batch_size: int = 32,
    strategy: str = "weighted",
    temperature: float = 0.3,
    num_workers: int = 4,
    seed: int = 42
):
    """
    Create balanced data loader
    
    Args:
        texts_by_dataset: {dataset_name: [text1, text2, ...]}
        labels_by_dataset: {dataset_name: [label1, label2, ...]}
        batch_size: Batch size
        strategy: Balancing strategy ("oversample", "downsample", "weighted")
        temperature: Temperature parameter (only for weighted strategy)
        num_workers: Number of data loading threads
        seed: Random seed
    
    Returns:
        DataLoader with balanced sampling
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Merge all datasets
    all_texts = []
    all_labels = []
    dataset_ids = []
    
    for dataset_id, (dataset_name, texts) in enumerate(texts_by_dataset.items()):
        all_texts.extend(texts)
        dataset_ids.extend([dataset_id] * len(texts))
        
        if labels_by_dataset and dataset_name in labels_by_dataset:
            all_labels.extend(labels_by_dataset[dataset_name])
        else:
            all_labels.extend([None] * len(texts))
    
    # Create balanced sampler
    sampler = BalancedDatasetSampler(
        dataset_labels=dataset_ids,
        strategy=strategy,
        temperature=temperature,
        seed=seed
    )
    
    # Note: This returns indices and dataset_ids
    # Actual text encoding needs to be done in the training loop
    return {
        'texts': all_texts,
        'labels': all_labels if any(l is not None for l in all_labels) else None,
        'dataset_ids': dataset_ids,
        'sampler': sampler
    }


if __name__ == '__main__':
    # Test code
    print("Testing balanced sampler\n")
    
    # Simulate data distribution
    dataset_labels = (
        [0] * 5000 +      # germanCoal: 5k
        [1] * 10000 +     # FCPB: 10k
        [2] * 40000 +     # socialTwitter: 40k
        [3] * 200000 +    # FCPB: 200k
        [4] * 1000000     # mental_health: 1M
    )
    
    print("=" * 70)
    print("Strategy 1: Oversample")
    print("=" * 70)
    sampler1 = BalancedDatasetSampler(dataset_labels, strategy="oversample")
    
    print("\n" + "=" * 70)
    print("Strategy 2: Weighted (temperature=0.3)")
    print("=" * 70)
    sampler2 = BalancedDatasetSampler(
        dataset_labels, 
        strategy="weighted", 
        temperature=0.3
    )
    
    print("\n" + "=" * 70)
    print("Strategy 3: Weighted (temperature=0.5)")
    print("=" * 70)
    sampler3 = BalancedDatasetSampler(
        dataset_labels, 
        strategy="weighted", 
        temperature=0.5
    )

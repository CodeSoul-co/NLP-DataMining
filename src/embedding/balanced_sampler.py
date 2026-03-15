"""
Balanced Dataset Sampler for Joint Training

解决数据不平衡问题：
- mental_health: 1,000,000 条
- germanCoal: 5,000 条
- 比例 200:1

策略：确保每个epoch中各数据集被采样的次数平衡
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Dict
from collections import Counter


class BalancedDatasetSampler(Sampler):
    """
    平衡数据集采样器
    
    确保每个batch中各数据集的样本比例均衡，避免大数据集"吞噬"小数据集。
    
    策略：
    1. 计算每个数据集的采样权重（对数缩放）
    2. 过采样小数据集，下采样大数据集
    3. 每个epoch确保所有数据集被充分采样
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
            dataset_labels: 每个样本所属的数据集ID (0, 1, 2, ...)
            strategy: 平衡策略
                - "oversample": 过采样小数据集到最大数据集的规模
                - "downsample": 下采样大数据集到最小数据集的规模
                - "weighted": 加权采样（推荐）
            temperature: 温度参数，控制平衡程度 (0=完全平衡, 1=原始分布)
            seed: 随机种子
        """
        self.dataset_labels = np.array(dataset_labels)
        self.strategy = strategy
        self.temperature = temperature
        self.seed = seed
        
        # 统计每个数据集的样本数
        self.dataset_counts = Counter(dataset_labels)
        self.num_datasets = len(self.dataset_counts)
        self.dataset_ids = sorted(self.dataset_counts.keys())
        
        # 为每个数据集创建索引
        self.dataset_indices = {}
        for dataset_id in self.dataset_ids:
            self.dataset_indices[dataset_id] = np.where(
                self.dataset_labels == dataset_id
            )[0]
        
        # 计算采样策略
        self._compute_sampling_strategy()
        
        print(f"\n[BalancedDatasetSampler] 初始化完成")
        print(f"  策略: {strategy}")
        print(f"  数据集数量: {self.num_datasets}")
        print(f"  原始分布:")
        for dataset_id in self.dataset_ids:
            count = self.dataset_counts[dataset_id]
            print(f"    Dataset {dataset_id}: {count:,} 样本")
        print(f"  平衡后每个epoch采样数: {len(self):,}")
    
    def _compute_sampling_strategy(self):
        """计算采样策略"""
        if self.strategy == "oversample":
            # 过采样：所有数据集都采样到最大数据集的规模
            max_count = max(self.dataset_counts.values())
            self.samples_per_dataset = {
                dataset_id: max_count 
                for dataset_id in self.dataset_ids
            }
            self.epoch_length = max_count * self.num_datasets
            
        elif self.strategy == "downsample":
            # 下采样：所有数据集都采样到最小数据集的规模
            min_count = min(self.dataset_counts.values())
            self.samples_per_dataset = {
                dataset_id: min_count 
                for dataset_id in self.dataset_ids
            }
            self.epoch_length = min_count * self.num_datasets
            
        elif self.strategy == "weighted":
            # 加权采样：使用对数缩放平衡
            # 计算每个数据集的采样权重
            counts = np.array([self.dataset_counts[i] for i in self.dataset_ids])
            
            # 对数缩放：log(count + 1)
            log_counts = np.log(counts + 1)
            
            # 应用温度参数
            # temperature=0: 完全平衡 (所有数据集相同权重)
            # temperature=1: 保持原始分布
            if self.temperature == 0:
                weights = np.ones_like(log_counts)
            else:
                weights = log_counts ** (1 / self.temperature)
            
            # 归一化
            weights = weights / weights.sum()
            
            # 计算每个数据集的采样数
            # 目标：总采样数约为所有数据集的平均值
            target_total = int(np.mean(counts) * self.num_datasets)
            self.samples_per_dataset = {
                dataset_id: int(weights[i] * target_total)
                for i, dataset_id in enumerate(self.dataset_ids)
            }
            self.epoch_length = sum(self.samples_per_dataset.values())
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        print(f"  平衡后分布:")
        for dataset_id in self.dataset_ids:
            samples = self.samples_per_dataset[dataset_id]
            original = self.dataset_counts[dataset_id]
            ratio = samples / original
            print(f"    Dataset {dataset_id}: {samples:,} 样本 "
                  f"(原始: {original:,}, 采样率: {ratio:.2f}x)")
    
    def __iter__(self):
        """生成采样索引"""
        np.random.seed(self.seed)
        
        # 为每个数据集生成采样索引
        all_indices = []
        
        for dataset_id in self.dataset_ids:
            dataset_idx = self.dataset_indices[dataset_id]
            num_samples = self.samples_per_dataset[dataset_id]
            
            if num_samples <= len(dataset_idx):
                # 下采样：随机选择
                sampled = np.random.choice(
                    dataset_idx, 
                    size=num_samples, 
                    replace=False
                )
            else:
                # 过采样：重复采样
                sampled = np.random.choice(
                    dataset_idx, 
                    size=num_samples, 
                    replace=True
                )
            
            all_indices.extend(sampled.tolist())
        
        # 打乱所有索引
        np.random.shuffle(all_indices)
        
        # 更新随机种子（每个epoch不同）
        self.seed += 1
        
        return iter(all_indices)
    
    def __len__(self):
        """返回每个epoch的样本数"""
        return self.epoch_length


class SimpleOversamplingDataset:
    """
    简单过采样数据集包装器
    
    在数据加载阶段直接复制小数据集的样本
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
            target_size: 目标大小，None则自动计算
            strategy: "max" (最大数据集), "mean" (平均), "median" (中位数)
        """
        self.texts_by_dataset = texts_by_dataset
        self.labels_by_dataset = labels_by_dataset or {}
        
        # 计算目标大小
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
        
        print(f"\n[SimpleOversamplingDataset] 过采样初始化")
        print(f"  目标大小: {target_size:,}")
        print(f"  原始分布:")
        
        # 过采样
        self.all_texts = []
        self.all_labels = []
        self.all_dataset_ids = []
        
        for dataset_id, (dataset_name, texts) in enumerate(texts_by_dataset.items()):
            original_size = len(texts)
            labels = self.labels_by_dataset.get(dataset_name, [None] * original_size)
            
            # 计算需要复制的次数
            repeat_times = int(np.ceil(target_size / original_size))
            
            # 过采样
            oversampled_texts = (texts * repeat_times)[:target_size]
            oversampled_labels = (labels * repeat_times)[:target_size]
            
            self.all_texts.extend(oversampled_texts)
            self.all_labels.extend(oversampled_labels)
            self.all_dataset_ids.extend([dataset_id] * target_size)
            
            print(f"    {dataset_name}: {original_size:,} → {target_size:,} "
                  f"(复制 {repeat_times}x)")
        
        print(f"  总样本数: {len(self.all_texts):,}")
    
    def get_data(self):
        """返回平衡后的数据"""
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
    创建平衡的数据加载器
    
    Args:
        texts_by_dataset: {dataset_name: [text1, text2, ...]}
        labels_by_dataset: {dataset_name: [label1, label2, ...]}
        batch_size: batch大小
        strategy: 平衡策略 ("oversample", "downsample", "weighted")
        temperature: 温度参数 (仅用于weighted策略)
        num_workers: 数据加载线程数
        seed: 随机种子
    
    Returns:
        DataLoader with balanced sampling
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # 合并所有数据集
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
    
    # 创建平衡采样器
    sampler = BalancedDatasetSampler(
        dataset_labels=dataset_ids,
        strategy=strategy,
        temperature=temperature,
        seed=seed
    )
    
    # 注意：这里返回的是索引和dataset_ids
    # 实际的文本编码需要在训练循环中进行
    return {
        'texts': all_texts,
        'labels': all_labels if any(l is not None for l in all_labels) else None,
        'dataset_ids': dataset_ids,
        'sampler': sampler
    }


if __name__ == '__main__':
    # 测试代码
    print("测试平衡采样器\n")
    
    # 模拟数据分布
    dataset_labels = (
        [0] * 5000 +      # germanCoal: 5k
        [1] * 10000 +     # FCPB: 10k
        [2] * 40000 +     # socialTwitter: 40k
        [3] * 200000 +    # FCPB: 200k
        [4] * 1000000     # mental_health: 1M
    )
    
    print("=" * 70)
    print("策略1: Oversample (过采样)")
    print("=" * 70)
    sampler1 = BalancedDatasetSampler(dataset_labels, strategy="oversample")
    
    print("\n" + "=" * 70)
    print("策略2: Weighted (加权采样, temperature=0.3)")
    print("=" * 70)
    sampler2 = BalancedDatasetSampler(
        dataset_labels, 
        strategy="weighted", 
        temperature=0.3
    )
    
    print("\n" + "=" * 70)
    print("策略3: Weighted (加权采样, temperature=0.5)")
    print("=" * 70)
    sampler3 = BalancedDatasetSampler(
        dataset_labels, 
        strategy="weighted", 
        temperature=0.5
    )

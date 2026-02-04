"""
Data loader for all 5 sociology text datasets.
Provides unified interface for loading and preprocessing text data.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Dataset metadata container"""
    name: str
    file_path: str
    text_column: str
    label_column: Optional[str]
    language: str
    num_samples: int
    has_label: bool


class DatasetLoader:
    """
    Unified data loader for all 5 datasets.
    
    Datasets:
        - germanCoal: German political debates (unlabeled)
        - FCPB: Financial consumer complaints (unlabeled)
        - hatespeech: Hate speech detection (binary label)
        - mental_health: Mental health posts (28 categories)
        - socialTwitter: Twitter account classification (binary label)
    """
    
    # Dataset configurations
    DATASET_CONFIGS = {
        "germanCoal": {
            "file_path": "data/germanCoal/german_coal_text_only.csv",
            "text_column": "cleaned_text",
            "label_column": None,
            "language": "de",
            "has_label": False
        },
        "FCPB": {
            "file_path": "data/FCPB/complaints_text_only.csv",
            "text_column": "Consumer complaint narrative",
            "label_column": None,
            "language": "en",
            "has_label": False
        },
        "hatespeech": {
            "file_path": "data/hatespeech/hatespeech_text_only.csv",
            "text_column": "cleaned_content",
            "label_column": "Label",
            "language": "en",
            "has_label": True
        },
        "mental_health": {
            "file_path": "data/mental_health/mental_health_text_only.csv",
            "text_column": "clean_text",
            "label_column": "subreddit_id",
            "language": "en",
            "has_label": True
        },
        "socialTwitter": {
            "file_path": "data/socialTwitter/socialTwitter_text_only.csv",
            "text_column": "clean_text",
            "label_column": "label",
            "language": "multi",
            "has_label": True
        }
    }
    
    def __init__(self, base_path: str = "/root/autodl-tmp", dev_mode: bool = False):
        """
        Initialize data loader.
        
        Args:
            base_path: Root directory of the project
            dev_mode: If True, print debug information
        """
        self.base_path = base_path
        self.dev_mode = dev_mode
        self._validate_datasets()
    
    def _validate_datasets(self):
        """Validate that all dataset files exist"""
        for name, config in self.DATASET_CONFIGS.items():
            full_path = os.path.join(self.base_path, config["file_path"])
            if not os.path.exists(full_path):
                print(f"[WARNING] Dataset {name} not found at {full_path}")
            elif self.dev_mode:
                print(f"[DEV] Found dataset {name} at {full_path}")
    
    def get_available_datasets(self) -> List[str]:
        """Return list of available dataset names"""
        return list(self.DATASET_CONFIGS.keys())
    
    def get_labeled_datasets(self) -> List[str]:
        """Return list of datasets with labels"""
        return [name for name, config in self.DATASET_CONFIGS.items() if config["has_label"]]
    
    def get_unlabeled_datasets(self) -> List[str]:
        """Return list of datasets without labels"""
        return [name for name, config in self.DATASET_CONFIGS.items() if not config["has_label"]]
    
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """
        Get metadata for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DatasetInfo object with dataset metadata
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {self.get_available_datasets()}")
        
        config = self.DATASET_CONFIGS[dataset_name]
        full_path = os.path.join(self.base_path, config["file_path"])
        
        # Count lines without loading full file
        with open(full_path, 'r', encoding='utf-8') as f:
            num_samples = sum(1 for _ in f) - 1  # Subtract header
        
        return DatasetInfo(
            name=dataset_name,
            file_path=full_path,
            text_column=config["text_column"],
            label_column=config["label_column"],
            language=config["language"],
            num_samples=num_samples,
            has_label=config["has_label"]
        )
    
    def load_dataset(
        self, 
        dataset_name: str, 
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        random_seed: int = 42
    ) -> Tuple[List[str], Optional[np.ndarray], DatasetInfo]:
        """
        Load a dataset and return texts and labels.
        
        Args:
            dataset_name: Name of the dataset to load
            max_samples: Maximum number of samples to load (None for all)
            shuffle: Whether to shuffle the data
            random_seed: Random seed for shuffling
            
        Returns:
            texts: List of text strings
            labels: Numpy array of labels (None if unlabeled)
            info: DatasetInfo object
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {self.get_available_datasets()}")
        
        config = self.DATASET_CONFIGS[dataset_name]
        full_path = os.path.join(self.base_path, config["file_path"])
        
        if self.dev_mode:
            print(f"[DEV] Loading dataset: {dataset_name}")
            print(f"[DEV] File path: {full_path}")
        
        # Load CSV
        df = pd.read_csv(full_path, encoding='utf-8')
        
        if self.dev_mode:
            print(f"[DEV] Loaded {len(df)} rows")
            print(f"[DEV] Columns: {df.columns.tolist()}")
        
        # Shuffle if requested
        if shuffle:
            df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(df):
            df = df.head(max_samples)
            if self.dev_mode:
                print(f"[DEV] Limited to {max_samples} samples")
        
        # Extract texts
        text_col = config["text_column"]
        texts = df[text_col].fillna("").astype(str).tolist()
        
        # Extract labels if available
        labels = None
        if config["has_label"] and config["label_column"] is not None:
            label_col = config["label_column"]
            labels = df[label_col].values
            
            if self.dev_mode:
                unique_labels = np.unique(labels)
                print(f"[DEV] Labels column: {label_col}")
                print(f"[DEV] Unique labels: {len(unique_labels)}")
                if len(unique_labels) <= 10:
                    print(f"[DEV] Label values: {unique_labels}")
        
        # Create info object
        info = DatasetInfo(
            name=dataset_name,
            file_path=full_path,
            text_column=text_col,
            label_column=config["label_column"],
            language=config["language"],
            num_samples=len(texts),
            has_label=config["has_label"]
        )
        
        if self.dev_mode:
            print(f"[DEV] Text sample: {texts[0][:100]}...")
        
        return texts, labels, info
    
    def load_all_datasets(
        self, 
        max_samples_per_dataset: Optional[int] = None
    ) -> Dict[str, Tuple[List[str], Optional[np.ndarray], DatasetInfo]]:
        """
        Load all available datasets.
        
        Args:
            max_samples_per_dataset: Maximum samples per dataset
            
        Returns:
            Dictionary mapping dataset name to (texts, labels, info)
        """
        all_data = {}
        for name in self.get_available_datasets():
            try:
                texts, labels, info = self.load_dataset(name, max_samples=max_samples_per_dataset)
                all_data[name] = (texts, labels, info)
                print(f"Loaded {name}: {len(texts)} samples, labeled: {info.has_label}")
            except Exception as e:
                print(f"[ERROR] Failed to load {name}: {e}")
        
        return all_data


def get_dataset_summary(base_path: str = "/root/autodl-tmp") -> pd.DataFrame:
    """
    Get summary statistics for all datasets.
    
    Returns:
        DataFrame with dataset statistics
    """
    loader = DatasetLoader(base_path)
    
    summaries = []
    for name in loader.get_available_datasets():
        try:
            info = loader.get_dataset_info(name)
            config = loader.DATASET_CONFIGS[name]
            summaries.append({
                "Dataset": name,
                "Samples": info.num_samples,
                "Language": info.language,
                "Has Label": info.has_label,
                "Label Column": config["label_column"] or "N/A"
            })
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
    
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    # Test data loader
    print("=" * 60)
    print("Testing DatasetLoader")
    print("=" * 60)
    
    loader = DatasetLoader(dev_mode=True)
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(get_dataset_summary())
    
    # Test loading hatespeech dataset
    print("\n" + "=" * 60)
    print("Testing hatespeech dataset loading")
    print("=" * 60)
    
    texts, labels, info = loader.load_dataset("hatespeech", max_samples=100)
    print(f"\nLoaded {len(texts)} texts")
    print(f"Labels shape: {labels.shape if labels is not None else 'N/A'}")
    print(f"Sample text: {texts[0][:100]}...")




"""
Baseline Data Processor - Baseline模型数据处理器

为LDA、CTM等Baseline模型提供数据处理功能：
1. 从CSV文件加载文本数据
2. 生成BOW矩阵（使用sklearn的CountVectorizer）
3. 为CTM生成SBERT embedding（不使用Qwen）

这样Baseline模型使用独立的数据处理流程，与Qwen-based ETM分开。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BaselineDataProcessor:
    """
    Baseline模型数据处理器
    
    从原始CSV文件处理数据，生成BOW矩阵和（可选的）SBERT embedding。
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 5,
        max_df: float = 0.95,
        stop_words: str = 'english',
        use_tfidf: bool = False
    ):
        """
        初始化数据处理器
        
        Args:
            max_features: 最大词汇量
            min_df: 最小文档频率
            max_df: 最大文档频率
            stop_words: 停用词
            use_tfidf: 是否使用TF-IDF而不是词频
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.use_tfidf = use_tfidf
        
        # 向量化器
        VectorizerClass = TfidfVectorizer if use_tfidf else CountVectorizer
        self.vectorizer = VectorizerClass(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words
        )
        
        # 存储处理后的数据
        self.texts = None
        self.labels = None
        self.bow_matrix = None
        self.vocab = None
        self.vocab_to_idx = None
    
    # 常见的文本列名（按优先级排序）
    TEXT_COLUMN_CANDIDATES = [
        'cleaned_content', 'cleaned_text', 'clean_text', 'text', 'content',
        'Consumer complaint narrative',  # FCPB
        'narrative', 'document', 'body', 'message', 'post'
    ]
    
    # 常见的标签列名
    LABEL_COLUMN_CANDIDATES = [
        'Label', 'label', 'labels', 'category', 'class', 'target',
        'subreddit_id', 'subreddit'  # mental_health
    ]
    
    def load_csv(
        self,
        csv_path: str,
        text_column: str = None,
        label_column: str = None
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        从CSV文件加载数据，自动检测列名
        
        Args:
            csv_path: CSV文件路径
            text_column: 文本列名（None则自动检测）
            label_column: 标签列名（None则自动检测）
            
        Returns:
            (texts, labels)
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Available columns: {df.columns.tolist()}")
        
        # 自动检测文本列
        if text_column is None or text_column not in df.columns:
            for col in self.TEXT_COLUMN_CANDIDATES:
                if col in df.columns:
                    text_column = col
                    break
            else:
                # 如果还没找到，尝试找包含text/content的列
                for col in df.columns:
                    if 'text' in col.lower() or 'content' in col.lower():
                        text_column = col
                        break
                else:
                    # 最后尝试：使用第一个字符串类型的列
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            text_column = col
                            break
                    else:
                        raise ValueError(f"Text column not found. Available: {df.columns.tolist()}")
        
        print(f"Using text column: '{text_column}'")
        self.texts = df[text_column].fillna('').astype(str).tolist()
        
        # 自动检测标签列
        if label_column is None:
            for col in self.LABEL_COLUMN_CANDIDATES:
                if col in df.columns:
                    label_column = col
                    break
        
        if label_column and label_column in df.columns:
            self.labels = df[label_column].values
            print(f"Using label column: '{label_column}'")
        else:
            self.labels = None
            print("No label column found (this is OK for unsupervised training)")
        
        print(f"Loaded {len(self.texts)} documents")
        return self.texts, self.labels
    
    def build_bow(
        self,
        texts: List[str] = None
    ) -> Tuple[sp.csr_matrix, List[str]]:
        """
        构建BOW矩阵
        
        Args:
            texts: 文本列表，如果为None则使用已加载的文本
            
        Returns:
            (bow_matrix, vocab)
        """
        if texts is None:
            texts = self.texts
        if texts is None:
            raise ValueError("No texts available. Call load_csv first.")
        
        print(f"Building BOW matrix with max_features={self.max_features}...")
        
        # 构建BOW
        self.bow_matrix = self.vectorizer.fit_transform(texts)
        self.vocab = self.vectorizer.get_feature_names_out().tolist()
        self.vocab_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        print(f"BOW matrix shape: {self.bow_matrix.shape}")
        print(f"Vocabulary size: {len(self.vocab)}")
        
        return self.bow_matrix, self.vocab
    
    def get_sbert_embeddings(
        self,
        texts: List[str] = None,
        model_name: str = '/root/autodl-tmp/ETM/model/sbert/sentence-transformers/all-MiniLM-L6-v2',
        batch_size: int = 32,
        device: str = 'auto'
    ) -> np.ndarray:
        """
        使用Sentence-BERT生成文档embedding
        
        用于CTM模型，不使用Qwen embedding。
        
        Args:
            texts: 文本列表
            model_name: SBERT模型名称
            batch_size: 批次大小
            device: 设备
            
        Returns:
            embeddings: (num_docs, embedding_dim)
        """
        if texts is None:
            texts = self.texts
        if texts is None:
            raise ValueError("No texts available. Call load_csv first.")
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        print(f"Loading SBERT model: {model_name}...")
        
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = SentenceTransformer(model_name, device=device)
        
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def save(self, save_dir: str):
        """
        保存处理后的数据
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存BOW矩阵
        if self.bow_matrix is not None:
            sp.save_npz(os.path.join(save_dir, 'bow_matrix.npz'), self.bow_matrix)
        
        # 保存词汇表
        if self.vocab is not None:
            with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False)
        
        # 保存配置
        config = {
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_tfidf': self.use_tfidf,
            'num_docs': len(self.texts) if self.texts else 0,
            'vocab_size': len(self.vocab) if self.vocab else 0
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Data saved to {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str) -> 'BaselineDataProcessor':
        """
        加载已保存的数据
        
        Args:
            save_dir: 保存目录
            
        Returns:
            BaselineDataProcessor实例
        """
        # 加载配置
        with open(os.path.join(save_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        processor = cls(
            max_features=config['max_features'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            use_tfidf=config['use_tfidf']
        )
        
        # 加载BOW矩阵
        bow_path = os.path.join(save_dir, 'bow_matrix.npz')
        if os.path.exists(bow_path):
            processor.bow_matrix = sp.load_npz(bow_path)
        
        # 加载词汇表
        vocab_path = os.path.join(save_dir, 'vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                processor.vocab = json.load(f)
            processor.vocab_to_idx = {word: idx for idx, word in enumerate(processor.vocab)}
        
        return processor


def prepare_baseline_data(
    dataset: str,
    vocab_size: int = 5000,
    data_dir: str = '/root/autodl-tmp/data',
    save_dir: str = '/root/autodl-tmp/result/baseline',
    generate_sbert: bool = True,
    sbert_model: str = '/root/autodl-tmp/ETM/model/sbert/sentence-transformers/all-MiniLM-L6-v2'
) -> Dict[str, Any]:
    """
    准备Baseline模型所需的数据
    
    Args:
        dataset: 数据集名称
        vocab_size: 词汇表大小
        data_dir: 数据目录
        save_dir: 保存目录
        generate_sbert: 是否生成SBERT embedding（用于CTM）
        sbert_model: SBERT模型名称
        
    Returns:
        包含所有数据的字典
    """
    # 查找CSV文件
    dataset_dir = os.path.join(data_dir, dataset)
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    
    csv_path = os.path.join(dataset_dir, csv_files[0])
    print(f"Using CSV file: {csv_path}")
    
    # 创建处理器
    processor = BaselineDataProcessor(max_features=vocab_size)
    
    # 加载数据
    texts, labels = processor.load_csv(csv_path)
    
    # 构建BOW
    bow_matrix, vocab = processor.build_bow()
    
    # 保存目录
    output_dir = os.path.join(save_dir, dataset)
    processor.save(output_dir)
    
    result = {
        'texts': texts,
        'labels': labels,
        'bow_matrix': bow_matrix,
        'vocab': vocab,
        'save_dir': output_dir
    }
    
    # 生成SBERT embedding（用于CTM）
    if generate_sbert:
        try:
            embeddings = processor.get_sbert_embeddings(
                model_name=sbert_model
            )
            np.save(os.path.join(output_dir, 'sbert_embeddings.npy'), embeddings)
            result['sbert_embeddings'] = embeddings
        except ImportError as e:
            print(f"Warning: {e}")
            print("CTM will not be available without SBERT embeddings.")
    
    return result

"""
Baseline Data Processor

Provides data processing for baseline models (LDA, CTM, etc.):
1. Load text data from CSV files
2. Generate BOW matrix (using sklearn CountVectorizer)
3. Generate SBERT embeddings for CTM (without Qwen)

Baseline models use an independent data processing pipeline, separate from Qwen-based ETM.
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
    Baseline Model Data Processor
    
    Process data from raw CSV files, generate BOW matrix and (optional) SBERT embeddings.
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
        Initialize data processor
        
        Args:
            max_features: Maximum vocabulary size
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            stop_words: Stop words
            use_tfidf: Whether to use TF-IDF instead of term frequency
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.use_tfidf = use_tfidf
        
        VectorizerClass = TfidfVectorizer if use_tfidf else CountVectorizer
        self.vectorizer = VectorizerClass(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words
        )
        
        self.texts = None
        self.labels = None
        self.bow_matrix = None
        self.vocab = None
        self.vocab_to_idx = None
    
    TEXT_COLUMN_CANDIDATES = [
        'cleaned_content', 'cleaned_text', 'clean_text', 'text', 'content',
        'Consumer complaint narrative',  # FCPB
        'narrative', 'document', 'body', 'message', 'post'
    ]
    
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
        Load data from CSV file with automatic column detection
        
        Args:
            csv_path: CSV file path
            text_column: Text column name (None for auto-detection)
            label_column: Label column name (None for auto-detection)
            
        Returns:
            (texts, labels)
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Available columns: {df.columns.tolist()}")
        
        if text_column is None or text_column not in df.columns:
            for col in self.TEXT_COLUMN_CANDIDATES:
                if col in df.columns:
                    text_column = col
                    break
            else:
                for col in df.columns:
                    if 'text' in col.lower() or 'content' in col.lower():
                        text_column = col
                        break
                else:
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            text_column = col
                            break
                    else:
                        raise ValueError(f"Text column not found. Available: {df.columns.tolist()}")
        
        print(f"Using text column: '{text_column}'")
        self.texts = df[text_column].fillna('').astype(str).tolist()
        
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
        Build BOW matrix
        
        Args:
            texts: Text list, if None use loaded texts
            
        Returns:
            (bow_matrix, vocab)
        """
        if texts is None:
            texts = self.texts
        if texts is None:
            raise ValueError("No texts available. Call load_csv first.")
        
        print(f"Building BOW matrix with max_features={self.max_features}...")
        
        self.bow_matrix = self.vectorizer.fit_transform(texts)
        self.vocab = self.vectorizer.get_feature_names_out().tolist()
        self.vocab_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        print(f"BOW matrix shape: {self.bow_matrix.shape}")
        print(f"Vocabulary size: {len(self.vocab)}")
        
        return self.bow_matrix, self.vocab
    
    def get_sbert_embeddings(
        self,
        texts: List[str] = None,
        model_name: str = None,
        batch_size: int = 32,
        device: str = 'auto'
    ) -> np.ndarray:
        # Default SBERT model path from environment or config
        if model_name is None:
            model_name = os.environ.get('SBERT_MODEL_PATH', 'sentence-transformers/all-MiniLM-L6-v2')
        """
        Generate document embeddings using Sentence-BERT
        
        Used for CTM model, without Qwen embeddings.
        
        Args:
            texts: Text list
            model_name: SBERT model name
            batch_size: Batch size
            device: Device
            
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
        Save processed data
        
        Args:
            save_dir: Save directory
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.bow_matrix is not None:
            # Save as dense npy format
            bow_dense = self.bow_matrix.toarray() if sp.issparse(self.bow_matrix) else self.bow_matrix
            np.save(os.path.join(save_dir, 'bow_matrix.npy'), bow_dense)
        
        if self.vocab is not None:
            with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False)
        
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
        Load saved data
        
        Args:
            save_dir: Save directory
            
        Returns:
            BaselineDataProcessor instance
        """
        with open(os.path.join(save_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        processor = cls(
            max_features=config['max_features'],
            min_df=config['min_df'],
            max_df=config['max_df'],
            use_tfidf=config['use_tfidf']
        )
        
        bow_path = os.path.join(save_dir, 'bow_matrix.npy')
        if os.path.exists(bow_path):
            processor.bow_matrix = np.load(bow_path)
        
        vocab_path = os.path.join(save_dir, 'vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                processor.vocab = json.load(f)
            processor.vocab_to_idx = {word: idx for idx, word in enumerate(processor.vocab)}
        
        return processor


def prepare_baseline_data(
    dataset: str,
    vocab_size: int = 5000,
    data_dir: str = None,
    save_dir: str = None,
    generate_sbert: bool = True,
    sbert_model: str = None
) -> Dict[str, Any]:
    # Default paths from config
    if data_dir is None:
        from config import DATA_DIR
        data_dir = str(DATA_DIR)
    if save_dir is None:
        from config import RESULT_DIR
        save_dir = str(RESULT_DIR / 'baseline')
    if sbert_model is None:
        sbert_model = os.environ.get('SBERT_MODEL_PATH', 'sentence-transformers/all-MiniLM-L6-v2')
    """
    Prepare data for baseline models
    
    Args:
        dataset: Dataset name
        vocab_size: Vocabulary size
        data_dir: Data directory
        save_dir: Save directory
        generate_sbert: Whether to generate SBERT embeddings (for CTM)
        sbert_model: SBERT model name
        
    Returns:
        Dictionary containing all data
    """
    dataset_dir = os.path.join(data_dir, dataset)
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    
    csv_path = os.path.join(dataset_dir, csv_files[0])
    print(f"Using CSV file: {csv_path}")
    
    processor = BaselineDataProcessor(max_features=vocab_size)
    
    texts, labels = processor.load_csv(csv_path)
    
    bow_matrix, vocab = processor.build_bow()
    
    output_dir = os.path.join(save_dir, dataset)
    processor.save(output_dir)
    
    result = {
        'texts': texts,
        'labels': labels,
        'bow_matrix': bow_matrix,
        'vocab': vocab,
        'save_dir': output_dir
    }
    
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

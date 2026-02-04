"""
Embedding & Pre-processing Module for THETA ETM Pipeline.

This module sits between Data Cleaning and ETM Model Training:
    Raw CSV -> Data Cleaning -> [Embedding Module] -> ETM Model Training

It generates TWO required artifacts for ETM:
1. BOW (Bag-of-Words) matrix - Sparse matrix (N x V)
2. Dense vector embeddings - Dense matrix (N x D)

Author: THETA Project
"""

import os
import json
import logging

# Import cloud-native configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_qwen_model_path, THETA_MODEL_SIZE
import gc
import numpy as np
from scipy import sparse
from typing import List, Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import pandas as pd
from tqdm import tqdm

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


class ProcessingStatus(Enum):
    """Status of the preprocessing pipeline"""
    IDLE = "idle"
    BOW_GENERATING = "bow_generating"
    BOW_COMPLETED = "bow_completed"
    EMBEDDING_GENERATING = "embedding_generating"
    EMBEDDING_COMPLETED = "embedding_completed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingConfig:
    """Configuration for the embedding processor"""
    # Model settings
    embedding_model: str = "Qwen-Embedding-0.6B"
    embedding_model_path: Optional[str] = None  # Auto-detect if None
    
    # BOW settings
    min_df: int = 5                      # Minimum document frequency
    max_df_ratio: float = 0.7            # Maximum document frequency ratio
    max_vocab_size: int = 50000          # Maximum vocabulary size
    min_word_length: int = 2             # Minimum word length (English)
    min_chinese_length: int = 1          # Minimum word length (Chinese)
    remove_stopwords: bool = True        # Remove stopwords
    language: str = "multi"              # Language: 'en', 'zh', 'multi'
    
    # Embedding settings
    max_sequence_length: int = 512       # Max tokens per document
    batch_size: int = 32                 # Batch size for embedding
    normalize_embeddings: bool = True    # L2 normalize embeddings
    
    # Processing settings
    chunk_size: int = 500                # Chunk size for large datasets
    device: str = "auto"                 # 'cuda', 'cpu', or 'auto'
    
    # Output settings
    save_format: str = "npz"             # 'npz' for BOW, 'npy' for embeddings


@dataclass
class ProcessingResult:
    """Result of preprocessing pipeline"""
    dataset_name: str
    status: ProcessingStatus
    bow_path: Optional[str] = None
    bow_metadata_path: Optional[str] = None
    embedding_path: Optional[str] = None
    embedding_metadata_path: Optional[str] = None
    vocab_path: Optional[str] = None
    
    # Statistics
    num_documents: int = 0
    vocab_size: int = 0
    embedding_dim: int = 0
    bow_sparsity: float = 0.0
    
    # Timing
    bow_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    
    # Error info
    error_message: Optional[str] = None
    
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        return result


class EmbeddingProcessor:
    """
    Main processor class for generating BOW and dense embeddings.
    
    Usage:
        processor = EmbeddingProcessor(config)
        result = processor.process(csv_path, text_column, output_dir)
    
    The processor generates:
    1. BOW matrix (.npz): Sparse matrix of shape (N, V)
    2. Embeddings (.npy): Dense matrix of shape (N, D)
    3. Vocabulary (.json): Word to index mapping
    4. Metadata (.json): Processing statistics and config
    """
    
    # Stopwords
    ENGLISH_STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'where', 'why',
        'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'would', 'could', 'ought', 'im', 'youre',
        'hes', 'shes', 'its', 'were', 'theyre', 'ive', 'youve', 'weve',
        'theyve', 'id', 'youd', 'hed', 'shed', 'wed', 'theyd', 'ill', 'youll',
        'hell', 'shell', 'well', 'theyll', 'isnt', 'arent', 'wasnt', 'werent',
        'hasnt', 'havent', 'hadnt', 'doesnt', 'dont', 'didnt', 'wont', 'wouldnt',
        'shant', 'shouldnt', 'cant', 'cannot', 'couldnt', 'mustnt', 'lets',
        'url_link', 'user_mention', 'http', 'https', 'www', 'com'
    }
    
    CHINESE_STOPWORDS = {
        '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '那', '她', '他', '它', '们', '这个',
        '那个', '什么', '怎么', '为什么', '哪', '哪里', '谁', '多少', '几',
        '能', '可以', '应该', '必须', '得', '把', '被', '让', '给', '对',
        '从', '向', '往', '跟', '比', '在于', '关于', '对于', '由于', '因为',
        '所以', '但是', '然而', '而且', '并且', '或者', '如果', '虽然', '即使',
    }
    
    # Available embedding models
    AVAILABLE_MODELS = {
        "Qwen-Embedding-0.6B": {
            "dim": 1024,
            "local_path": "qwen3_embedding_0.6B",
            "description": "Qwen3 Embedding 0.6B - Fast, good quality"
        },
        "Qwen-Embedding-1.8B": {
            "dim": 2048,
            "local_path": "qwen3_embedding_1.8B",
            "description": "Qwen3 Embedding 1.8B - Better quality, slower"
        },
        "Qwen-Embedding-7B": {
            "dim": 4096,
            "local_path": "qwen3_embedding_7B",
            "description": "Qwen3 Embedding 7B - Best quality, requires GPU"
        }
    }
    
    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        dev_mode: bool = False
    ):
        """
        Initialize the embedding processor.
        
        Args:
            config: Processing configuration
            progress_callback: Callback function(stage, progress, message)
            dev_mode: Enable debug output
        """
        self.config = config or ProcessingConfig()
        self.progress_callback = progress_callback
        self.dev_mode = dev_mode
        
        # Build stopwords set
        self.stopwords = self.ENGLISH_STOPWORDS.copy()
        if self.config.language in ['zh', 'multi']:
            self.stopwords.update(self.CHINESE_STOPWORDS)
        
        # Model will be loaded lazily
        self._model = None
        self._tokenizer = None
        self._embedding_dim = None
        
        # Current status
        self.status = ProcessingStatus.IDLE
        self.current_progress = 0.0
        
        if self.dev_mode:
            print(f"[DEV] EmbeddingProcessor initialized")
            print(f"[DEV] Config: {asdict(self.config)}")
    
    def _report_progress(self, stage: str, progress: float, message: str):
        """Report progress to callback"""
        self.current_progress = progress
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
        if self.dev_mode:
            print(f"[DEV] {stage}: {progress:.1%} - {message}")
    
    def _detect_model_path(self) -> str:
        """Auto-detect the embedding model path"""
        if self.config.embedding_model_path:
            return self.config.embedding_model_path
        
        # Try common locations
        model_info = self.AVAILABLE_MODELS.get(self.config.embedding_model, {})
        local_name = model_info.get("local_path", "qwen3_embedding_0.6B")
        
        # Use environment variable for base path
        theta_base = os.environ.get("THETA_BASE", "/root/autodl-tmp")
        
        possible_paths = [
            # Project directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), local_name),
            # THETA_wait directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "THETA_wait", local_name),
            # Environment variable path
            os.path.join(theta_base, local_name),
            # Current directory
            os.path.join(os.getcwd(), local_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                if self.dev_mode:
                    print(f"[DEV] Found model at: {path}")
                return path
        
        # Return first path as default (will fail gracefully if not found)
        return possible_paths[0]
    
    def _load_embedding_model(self):
        """Load the embedding model (lazy loading)"""
        if self._model is not None:
            return
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for embedding generation. Install with: pip install torch")
        
        from transformers import AutoModel, AutoTokenizer
        
        model_path = self._detect_model_path()
        self._report_progress("embedding", 0.0, f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding model not found at: {model_path}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Determine device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        self._device = torch.device(device)
        
        # Load model
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        
        # Get embedding dimension
        self._embedding_dim = self._model.config.hidden_size
        
        self._report_progress("embedding", 0.05, f"Model loaded (dim={self._embedding_dim}, device={device})")
    
    def _is_chinese(self, char: str) -> bool:
        """Check if a character is Chinese"""
        return '\u4e00' <= char <= '\u9fff'
    
    def _has_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        return any(self._is_chinese(c) for c in text)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        import re
        
        text = text.lower()
        tokens = []
        
        # Chinese tokenization
        if self._has_chinese(text) and self.config.language in ['zh', 'multi']:
            if JIEBA_AVAILABLE:
                chinese_tokens = list(jieba.cut(text))
                tokens.extend(chinese_tokens)
            else:
                for char in text:
                    if self._is_chinese(char):
                        tokens.append(char)
        
        # English/other tokenization
        english_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        tokens.extend(english_tokens)
        
        # Filter tokens
        filtered = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            is_chinese = self._has_chinese(token)
            
            # Length filter
            if is_chinese:
                if len(token) < self.config.min_chinese_length:
                    continue
            else:
                if len(token) < self.config.min_word_length:
                    continue
            
            # Stopword filter
            if self.config.remove_stopwords and token in self.stopwords:
                continue
            
            filtered.append(token)
        
        return filtered
    
    def _build_vocabulary(self, texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from texts"""
        from collections import Counter
        
        word_counts = Counter()
        doc_counts = Counter()
        
        for text in tqdm(texts, desc="Building vocabulary"):
            tokens = self._tokenize(text)
            word_counts.update(tokens)
            doc_counts.update(set(tokens))
        
        # Filter by document frequency
        num_docs = len(texts)
        min_df = self.config.min_df
        max_df = int(num_docs * self.config.max_df_ratio)
        
        filtered_words = []
        for word, count in word_counts.items():
            doc_freq = doc_counts[word]
            if min_df <= doc_freq <= max_df:
                filtered_words.append((word, count))
        
        # Sort by frequency and limit size
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        if len(filtered_words) > self.config.max_vocab_size:
            filtered_words = filtered_words[:self.config.max_vocab_size]
        
        # Build mappings
        word2idx = {word: idx for idx, (word, _) in enumerate(filtered_words)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        return word2idx, idx2word
    
    def generate_bow(
        self,
        texts: List[str],
        word2idx: Dict[str, int],
        dataset_name: str = "dataset"
    ) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
        """
        Generate Bag-of-Words matrix.
        
        Args:
            texts: List of text documents
            word2idx: Word to index mapping
            dataset_name: Name of the dataset
            
        Returns:
            (bow_matrix, metadata)
        """
        self.status = ProcessingStatus.BOW_GENERATING
        self._report_progress("bow", 0.0, "Starting BOW generation")
        
        num_docs = len(texts)
        vocab_size = len(word2idx)
        
        rows, cols, data = [], [], []
        total_tokens = 0
        
        for doc_idx, text in enumerate(tqdm(texts, desc="Generating BOW")):
            tokens = self._tokenize(text)
            
            token_counts = {}
            for token in tokens:
                if token in word2idx:
                    idx = word2idx[token]
                    token_counts[idx] = token_counts.get(idx, 0) + 1
                    total_tokens += 1
            
            for word_idx, count in token_counts.items():
                rows.append(doc_idx)
                cols.append(word_idx)
                data.append(count)
            
            # Report progress
            if (doc_idx + 1) % 1000 == 0:
                progress = (doc_idx + 1) / num_docs
                self._report_progress("bow", progress, f"Processed {doc_idx + 1}/{num_docs} documents")
        
        # Create sparse matrix
        bow_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_docs, vocab_size),
            dtype=np.float32
        )
        
        # Calculate statistics
        non_zero = bow_matrix.nnz
        total_elements = num_docs * vocab_size
        sparsity = 1.0 - (non_zero / total_elements) if total_elements > 0 else 0
        avg_doc_length = total_tokens / num_docs if num_docs > 0 else 0
        
        metadata = {
            "dataset_name": dataset_name,
            "num_docs": num_docs,
            "vocab_size": vocab_size,
            "total_tokens": total_tokens,
            "avg_doc_length": avg_doc_length,
            "sparsity": sparsity,
            "non_zero_elements": non_zero
        }
        
        self.status = ProcessingStatus.BOW_COMPLETED
        self._report_progress("bow", 1.0, f"BOW complete: {num_docs}x{vocab_size}, sparsity={sparsity:.4f}")
        
        return bow_matrix, metadata
    
    def generate_embeddings(
        self,
        texts: List[str],
        dataset_name: str = "dataset"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate dense vector embeddings using the Qwen model.
        
        Args:
            texts: List of text documents
            dataset_name: Name of the dataset
            
        Returns:
            (embeddings, metadata)
        """
        self.status = ProcessingStatus.EMBEDDING_GENERATING
        self._report_progress("embedding", 0.0, "Starting embedding generation")
        
        # Load model if not loaded
        self._load_embedding_model()
        
        num_texts = len(texts)
        embeddings = np.zeros((num_texts, self._embedding_dim), dtype=np.float32)
        
        # Process in batches
        batch_size = self.config.batch_size
        num_batches = (num_texts + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating embeddings"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_texts)
                batch_texts = texts[start_idx:end_idx]
                
                # Tokenize
                encoded = self._tokenizer(
                    batch_texts,
                    max_length=self.config.max_sequence_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self._device)
                attention_mask = encoded['attention_mask'].to(self._device)
                
                # Forward pass
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Mean pooling
                last_hidden = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Normalize if requested
                if self.config.normalize_embeddings:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                # Store results
                embeddings[start_idx:end_idx] = batch_embeddings.cpu().float().numpy()
                
                # Report progress
                progress = (batch_idx + 1) / num_batches
                self._report_progress("embedding", 0.1 + 0.9 * progress, 
                                     f"Processed {end_idx}/{num_texts} documents")
                
                # Clear GPU memory
                del input_ids, attention_mask, outputs, last_hidden
                del mask_expanded, sum_embeddings, sum_mask, batch_embeddings
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Validate embeddings
        norms = np.linalg.norm(embeddings, axis=1)
        
        metadata = {
            "dataset_name": dataset_name,
            "model_name": self.config.embedding_model,
            "embedding_dim": self._embedding_dim,
            "num_docs": num_texts,
            "max_sequence_length": self.config.max_sequence_length,
            "normalized": self.config.normalize_embeddings,
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "has_nan": bool(np.isnan(embeddings).any()),
            "has_inf": bool(np.isinf(embeddings).any())
        }
        
        self.status = ProcessingStatus.EMBEDDING_COMPLETED
        self._report_progress("embedding", 1.0, f"Embeddings complete: {num_texts}x{self._embedding_dim}")
        
        return embeddings, metadata
    
    def process(
        self,
        csv_path: str,
        text_column: str,
        output_dir: str,
        dataset_name: Optional[str] = None
    ) -> ProcessingResult:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            csv_path: Path to the cleaned CSV file
            text_column: Name of the column containing text
            output_dir: Directory to save outputs
            dataset_name: Name for the dataset (auto-detected if None)
            
        Returns:
            ProcessingResult with paths and statistics
        """
        import time
        start_time = time.time()
        
        # Auto-detect dataset name
        if dataset_name is None:
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
            dataset_name = dataset_name.replace("_cleaned", "").replace("_text_only", "")
        
        result = ProcessingResult(
            dataset_name=dataset_name,
            status=ProcessingStatus.IDLE
        )
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load data
            self._report_progress("load", 0.0, f"Loading data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")
            
            texts = df[text_column].fillna("").astype(str).tolist()
            result.num_documents = len(texts)
            
            self._report_progress("load", 1.0, f"Loaded {len(texts)} documents")
            
            # ==========================================
            # Task A: BOW Generation
            # ==========================================
            bow_start = time.time()
            
            # Build vocabulary
            self._report_progress("vocab", 0.0, "Building vocabulary")
            word2idx, idx2word = self._build_vocabulary(texts)
            result.vocab_size = len(word2idx)
            
            # Save vocabulary
            vocab_path = os.path.join(output_dir, f"{dataset_name}_vocab.json")
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(word2idx, f, ensure_ascii=False, indent=2)
            result.vocab_path = vocab_path
            
            # Generate BOW
            bow_matrix, bow_meta = self.generate_bow(texts, word2idx, dataset_name)
            
            # Save BOW matrix
            bow_path = os.path.join(output_dir, f"{dataset_name}_bow.npz")
            sparse.save_npz(bow_path, bow_matrix)
            result.bow_path = bow_path
            
            # Save BOW metadata
            bow_meta_path = os.path.join(output_dir, f"{dataset_name}_bow_meta.json")
            with open(bow_meta_path, 'w') as f:
                json.dump(bow_meta, f, indent=2)
            result.bow_metadata_path = bow_meta_path
            result.bow_sparsity = bow_meta['sparsity']
            
            result.bow_time_seconds = time.time() - bow_start
            
            # ==========================================
            # Task B: Dense Embedding Generation
            # ==========================================
            emb_start = time.time()
            
            embeddings, emb_meta = self.generate_embeddings(texts, dataset_name)
            result.embedding_dim = emb_meta['embedding_dim']
            
            # Save embeddings
            emb_path = os.path.join(output_dir, f"{dataset_name}_embeddings.npy")
            np.save(emb_path, embeddings)
            result.embedding_path = emb_path
            
            # Save embedding metadata
            emb_meta_path = os.path.join(output_dir, f"{dataset_name}_embeddings_meta.json")
            with open(emb_meta_path, 'w') as f:
                json.dump(emb_meta, f, indent=2)
            result.embedding_metadata_path = emb_meta_path
            
            result.embedding_time_seconds = time.time() - emb_start
            
            # Complete
            result.status = ProcessingStatus.COMPLETED
            result.total_time_seconds = time.time() - start_time
            
            self._report_progress("complete", 1.0, 
                f"Processing complete in {result.total_time_seconds:.1f}s")
            
        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            result.total_time_seconds = time.time() - start_time
            self._report_progress("error", 0.0, f"Error: {e}")
            
            if self.dev_mode:
                import traceback
                traceback.print_exc()
        
        return result
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available embedding models"""
        return self.AVAILABLE_MODELS.copy()
    
    def validate_outputs(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Validate the generated outputs.
        
        Args:
            result: ProcessingResult from process()
            
        Returns:
            Validation report
        """
        report = {
            "valid": True,
            "bow_valid": False,
            "embedding_valid": False,
            "issues": []
        }
        
        # Check BOW
        if result.bow_path and os.path.exists(result.bow_path):
            try:
                bow = sparse.load_npz(result.bow_path)
                report["bow_shape"] = bow.shape
                report["bow_valid"] = True
                
                if bow.shape[0] != result.num_documents:
                    report["issues"].append(f"BOW rows ({bow.shape[0]}) != num_documents ({result.num_documents})")
                    report["valid"] = False
            except Exception as e:
                report["issues"].append(f"Failed to load BOW: {e}")
                report["valid"] = False
        else:
            report["issues"].append("BOW file not found")
            report["valid"] = False
        
        # Check embeddings
        if result.embedding_path and os.path.exists(result.embedding_path):
            try:
                emb = np.load(result.embedding_path)
                report["embedding_shape"] = emb.shape
                report["embedding_valid"] = True
                
                if emb.shape[0] != result.num_documents:
                    report["issues"].append(f"Embedding rows ({emb.shape[0]}) != num_documents ({result.num_documents})")
                    report["valid"] = False
                
                if np.isnan(emb).any():
                    report["issues"].append("Embeddings contain NaN values")
                    report["valid"] = False
            except Exception as e:
                report["issues"].append(f"Failed to load embeddings: {e}")
                report["valid"] = False
        else:
            report["issues"].append("Embedding file not found")
            report["valid"] = False
        
        return report


# Convenience function for CLI usage
def process_dataset(
    csv_path: str,
    text_column: str,
    output_dir: str,
    embedding_model: str = "Qwen-Embedding-0.6B",
    device: str = "auto"
) -> ProcessingResult:
    """
    Convenience function to process a dataset.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        output_dir: Output directory
        embedding_model: Model to use
        device: Device ('cuda', 'cpu', 'auto')
        
    Returns:
        ProcessingResult
    """
    config = ProcessingConfig(
        embedding_model=embedding_model,
        device=device
    )
    
    processor = EmbeddingProcessor(config, dev_mode=True)
    return processor.process(csv_path, text_column, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate BOW and embeddings for ETM")
    parser.add_argument("--csv", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--text-column", default="text", help="Name of text column")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="Qwen-Embedding-0.6B", help="Embedding model")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    result = process_dataset(
        csv_path=args.csv,
        text_column=args.text_column,
        output_dir=args.output,
        embedding_model=args.model,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("Processing Result")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2))

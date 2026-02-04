"""
Vocabulary Embedder for ETM

Generates embeddings for vocabulary words using Qwen model.
These embeddings (rho) are used in the ETM decoder to create interpretable topics.
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import logging

# Import cloud-native configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_qwen_model_path, THETA_MODEL_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VocabEmbedder:
    """
    Generates embeddings for vocabulary words using Qwen model.
    
    These embeddings form the semantic basis (rho) for ETM's decoder,
    allowing topics to be represented and interpreted in the same
    semantic space as documents.
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
        dev_mode: bool = False
    ):
        """
        Initialize vocabulary embedder.
        
        Args:
            model_path: Path to the Qwen model
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            dev_mode: Print debug information
        """
        # Use default path from cloud-native configuration
        self.model_path = model_path or get_qwen_model_path()
        self.batch_size = batch_size
        self.normalize = normalize
        self.dev_mode = dev_mode
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if self.dev_mode:
            logger.info(f"Using device: {self.device}")
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"Batch size: {self.batch_size}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen model with memory optimization"""
        from transformers import AutoModel, AutoTokenizer
        
        if self.dev_mode:
            logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Load model with bfloat16 for memory efficiency
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
        if self.dev_mode:
            logger.info(f"Model loaded successfully")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            logger.info(f"Device: {self.device}")
        
        # Set to eval mode
        self.model.eval()
    
    def embed_vocab(
        self,
        vocab_list: List[str],
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for vocabulary words.
        
        Args:
            vocab_list: List of vocabulary words
            output_path: Path to save embeddings (optional)
            show_progress: Show progress bar
            
        Returns:
            Word embeddings matrix of shape (vocab_size, embedding_dim)
        """
        if self.dev_mode:
            logger.info(f"Embedding {len(vocab_list)} vocabulary words")
            logger.info(f"Sample words: {vocab_list[:5]}")
        
        # Process in batches
        embeddings = np.zeros((len(vocab_list), self.embedding_dim), dtype=np.float32)
        
        iterator = range(0, len(vocab_list), self.batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Embedding vocabulary", total=len(vocab_list)//self.batch_size + 1)
        
        with torch.no_grad():
            for i in iterator:
                batch_words = vocab_list[i:i + self.batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_words,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Mean pooling
                last_hidden = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Normalize if requested
                if self.normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                # Convert to numpy and store
                batch_np = batch_embeddings.cpu().float().numpy()
                embeddings[i:i + len(batch_words)] = batch_np
        
        if self.dev_mode:
            logger.info(f"Vocabulary embeddings shape: {embeddings.shape}")
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1)
                logger.info(f"L2 norms (should be ~1.0): mean={norms.mean():.4f}, std={norms.std():.6f}")
        
        # Save embeddings if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, embeddings)
            if self.dev_mode:
                logger.info(f"Saved vocabulary embeddings to {output_path}")
        
        return embeddings
    
    @staticmethod
    def load_vocab_list(vocab_path: str) -> List[str]:
        """
        Load vocabulary list from file.
        
        Args:
            vocab_path: Path to vocabulary JSON file
            
        Returns:
            List of vocabulary words
        """
        if vocab_path.endswith('_list.json'):
            # Direct list format
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_list = json.load(f)
            return vocab_list
        else:
            # word2idx format
            with open(vocab_path, 'r', encoding='utf-8') as f:
                word2idx = json.load(f)
            
            # Convert to ordered list
            vocab_size = len(word2idx)
            vocab_list = [''] * vocab_size
            for word, idx in word2idx.items():
                vocab_list[int(idx)] = word
            
            return vocab_list


def generate_vocab_embeddings(
    vocab_path: str,
    output_path: Optional[str] = None,
    model_path: str = None,
    device: Optional[str] = None,
    batch_size: int = 64,
    normalize: bool = True,
    dev_mode: bool = False
) -> np.ndarray:
    """
    Generate embeddings for vocabulary words using Qwen model.
    
    Args:
        vocab_path: Path to vocabulary JSON file
        output_path: Path to save embeddings (optional)
        model_path: Path to the Qwen model
        device: Device to use ('cuda', 'cpu', or None for auto)
        batch_size: Batch size for encoding
        normalize: Whether to L2-normalize embeddings
        dev_mode: Print debug information
        
    Returns:
        Word embeddings matrix of shape (vocab_size, embedding_dim)
    """
    # Use config default if model_path not specified
    if model_path is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import get_qwen_model_path
        model_path = get_qwen_model_path()
    
    # Create embedder
    embedder = VocabEmbedder(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
        dev_mode=dev_mode
    )
    
    # Load vocabulary
    vocab_list = embedder.load_vocab_list(vocab_path)
    
    # Generate embeddings
    embeddings = embedder.embed_vocab(
        vocab_list=vocab_list,
        output_path=output_path,
        show_progress=True
    )
    
    return embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate vocabulary embeddings for ETM")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save embeddings")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Qwen model (uses THETA_BASE env var if not set)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding")
    parser.add_argument("--no_normalize", action="store_true", help="Disable L2 normalization")
    parser.add_argument("--dev_mode", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    # Generate embeddings
    generate_vocab_embeddings(
        vocab_path=args.vocab_path,
        output_path=args.output_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        dev_mode=args.dev_mode
    )

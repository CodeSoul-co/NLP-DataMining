"""
Embedding generator using Qwen3-Embedding-0.6B model.
Supports zero-shot, supervised (LoRA), and unsupervised training modes.
"""

import os
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm
from datetime import datetime
import json
from dataclasses import dataclass, asdict


@dataclass
class EmbeddingOutput:
    """Container for embedding results and metadata"""
    embeddings: np.ndarray           # Shape: (num_docs, embedding_dim)
    labels: Optional[np.ndarray]     # Shape: (num_docs,) or None
    dataset_name: str
    model_name: str
    embedding_dim: int
    num_docs: int
    mode: str                        # 'zero_shot', 'supervised', 'unsupervised'
    timestamp: str
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)"""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_docs": self.num_docs,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "config": self.config,
            "has_labels": self.labels is not None
        }


class QwenEmbedder:
    """
    Embedding generator using Qwen3-Embedding-0.6B.
    
    Supports:
        - Zero-shot: Direct embedding without training
        - Supervised: LoRA fine-tuning with labeled data
        - Unsupervised: Self-supervised LoRA training
    """
    
    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/qwen3_embedding_0.6B",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        normalize: bool = True,
        dev_mode: bool = False
    ):
        """
        Initialize the embedder.
        
        Args:
            model_path: Path to the Qwen3-Embedding model
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            dev_mode: Print debug information
        """
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.dev_mode = dev_mode
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if self.dev_mode:
            print(f"[DEV] Using device: {self.device}")
            print(f"[DEV] Model path: {self.model_path}")
            print(f"[DEV] Max length: {self.max_length}")
            print(f"[DEV] Batch size: {self.batch_size}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen3-Embedding model with memory optimization using transformers directly"""
        from transformers import AutoModel, AutoTokenizer
        
        if self.dev_mode:
            print(f"[DEV] Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Load model with bfloat16 for memory efficiency
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        
        if self.dev_mode:
            print(f"[DEV] Model loaded successfully")
            print(f"[DEV] Embedding dimension: {self.embedding_dim}")
            print(f"[DEV] Device: {self.device}")
        
        # Set to eval mode for zero-shot
        self.model.eval()
    
    def _get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        return {
            "model_path": self.model_path,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "device": str(self.device),
            "embedding_dim": self.embedding_dim
        }
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
        chunk_size: int = 500,
        temp_dir: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Uses chunked processing to avoid GPU memory overflow for large datasets.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            chunk_size: Number of samples per chunk (for memory management)
            temp_dir: Directory for temporary chunk files
            
        Returns:
            embeddings: numpy array of shape (num_texts, embedding_dim)
        """
        import gc
        
        if self.dev_mode:
            print(f"[DEV] Embedding {len(texts)} texts")
            print(f"[DEV] Sample text: {texts[0][:100]}...")
            print(f"[DEV] Chunk size: {chunk_size}")
        
        # For small datasets, process directly
        if len(texts) <= chunk_size:
            return self._embed_chunk(texts, show_progress)
        
        # For large datasets, process in chunks and save to disk
        # Use a unique temp directory to avoid conflicts with other runs
        import uuid
        import shutil
        run_id = uuid.uuid4().hex[:8]
        if temp_dir is None:
            temp_dir = f"/root/autodl-tmp/embedding/outputs/.temp_{run_id}"
        
        # Clean up any existing temp dir with same name (shouldn't happen with UUID)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"Temp directory: {temp_dir}")
        
        chunk_files = []
        num_chunks = (len(texts) + chunk_size - 1) // chunk_size
        
        print(f"Processing {len(texts)} texts in {num_chunks} chunks...")
        
        try:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(texts))
                chunk_texts = texts[start_idx:end_idx]
                
                print(f"\nChunk {chunk_idx + 1}/{num_chunks}: samples {start_idx}-{end_idx}")
                
                # Embed this chunk
                chunk_embeddings = self._embed_chunk(chunk_texts, show_progress)
                
                # Save chunk to disk with verification
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.npy")
                np.save(chunk_file, chunk_embeddings)
                
                # Verify file was saved
                if not os.path.exists(chunk_file):
                    raise IOError(f"Failed to save chunk file: {chunk_file}")
                
                chunk_files.append(chunk_file)
                
                # Free memory
                del chunk_embeddings
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Verify all chunk files exist before concatenating
            print("\nVerifying all chunk files...")
            missing_files = [f for f in chunk_files if not os.path.exists(f)]
            if missing_files:
                raise IOError(f"Missing chunk files: {missing_files[:5]}... ({len(missing_files)} total)")
            
            # Load and concatenate all chunks
            print(f"Concatenating {len(chunk_files)} chunks...")
            all_embeddings = []
            for i, chunk_file in enumerate(chunk_files):
                chunk_emb = np.load(chunk_file)
                all_embeddings.append(chunk_emb)
                # Delete temp file after loading
                os.remove(chunk_file)
                if (i + 1) % 50 == 0:
                    print(f"  Loaded {i + 1}/{len(chunk_files)} chunks")
            
            embeddings = np.vstack(all_embeddings)
            print(f"Final embeddings shape: {embeddings.shape}")
            
        finally:
            # Clean up temp directory
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temp dir: {e}")
        
        if self.dev_mode:
            print(f"[DEV] Final embeddings shape: {embeddings.shape}")
            print(f"[DEV] Embeddings dtype: {embeddings.dtype}")
        
        return embeddings
    
    def _embed_chunk(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed a single chunk of texts using transformers directly.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            embeddings: numpy array
        """
        import gc
        
        # Pre-allocate array for efficiency
        num_texts = len(texts)
        embeddings = np.zeros((num_texts, self.embedding_dim), dtype=np.float32)
        
        # Process in batches
        num_batches = (num_texts + self.batch_size - 1) // self.batch_size
        iterator = range(0, num_texts, self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", total=num_batches)
        
        with torch.no_grad():
            for batch_idx, i in enumerate(iterator):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
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
                embeddings[i:i + len(batch_texts)] = batch_np
                
                if self.dev_mode and batch_idx == 0:
                    print(f"[DEV] First batch shape: {batch_np.shape}")
                    print(f"[DEV] First embedding sample: {batch_np[0][:5]}...")
                
                # Clear GPU memory every batch to prevent accumulation
                del input_ids, attention_mask, outputs, last_hidden, mask_expanded
                del sum_embeddings, sum_mask, batch_embeddings, batch_np
                gc.collect()
                torch.cuda.empty_cache()
        
        if self.dev_mode:
            print(f"[DEV] Chunk embeddings shape: {embeddings.shape}")
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1)
                print(f"[DEV] L2 norms (should be ~1.0): mean={norms.mean():.4f}, std={norms.std():.6f}")
        
        return embeddings
    
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Simple encode interface for compatibility.
        
        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            embeddings: numpy array of shape (num_texts, embedding_dim)
        """
        old_normalize = self.normalize
        self.normalize = normalize
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        self.normalize = old_normalize
        return embeddings
    
    def zero_shot_embed(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        dataset_name: str = "unknown",
        show_progress: bool = True
    ) -> EmbeddingOutput:
        """
        Generate zero-shot embeddings (no training).
        
        Args:
            texts: List of text strings
            labels: Optional labels array
            dataset_name: Name of the dataset
            show_progress: Show progress bar
            
        Returns:
            EmbeddingOutput containing embeddings and metadata
        """
        print(f"Generating zero-shot embeddings for {dataset_name}...")
        print(f"  - Texts: {len(texts)}")
        print(f"  - Model: {self.model_path}")
        
        # Generate embeddings
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        
        # Create output object
        output = EmbeddingOutput(
            embeddings=embeddings,
            labels=labels,
            dataset_name=dataset_name,
            model_name="Qwen3-Embedding-0.6B",
            embedding_dim=self.embedding_dim,
            num_docs=len(texts),
            mode="zero_shot",
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            config=self._get_config()
        )
        
        print(f"  - Output shape: {embeddings.shape}")
        
        return output
    
    def validate_embeddings(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Validate embedding quality and return statistics.
        
        Args:
            embeddings: Embedding matrix
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            "shape": embeddings.shape,
            "dtype": str(embeddings.dtype),
            "mean": float(embeddings.mean()),
            "std": float(embeddings.std()),
            "min": float(embeddings.min()),
            "max": float(embeddings.max()),
            "has_nan": bool(np.isnan(embeddings).any()),
            "has_inf": bool(np.isinf(embeddings).any())
        }
        
        # Check L2 norms
        norms = np.linalg.norm(embeddings, axis=1)
        stats["norm_mean"] = float(norms.mean())
        stats["norm_std"] = float(norms.std())
        stats["norm_min"] = float(norms.min())
        stats["norm_max"] = float(norms.max())
        
        # Sample pairwise similarities
        if len(embeddings) > 10:
            sample_idx = np.random.choice(len(embeddings), min(100, len(embeddings)), replace=False)
            sample_embs = embeddings[sample_idx]
            sim_matrix = np.dot(sample_embs, sample_embs.T)
            np.fill_diagonal(sim_matrix, 0)
            stats["avg_similarity"] = float(sim_matrix.mean())
        
        return stats


class EmbeddingManager:
    """
    Manager for saving and loading embedding results.
    Handles result versioning to prevent overwriting.
    """
    
    def __init__(
        self,
        output_dir: str = "/root/autodl-tmp/embedding/outputs",
        result_dir: str = "/root/autodl-tmp/result",
        model_size: str = "0.6B",
        exp_dir: str = "",
        dev_mode: bool = False
    ):
        """
        Initialize embedding manager.
        
        Args:
            output_dir: Directory for intermediate outputs
            result_dir: Directory for final results (with versioning)
            model_size: Qwen model size (0.6B, 4B, 8B) for result directory structure
            exp_dir: If set, save embeddings to {exp_dir}/embeddings/ (exp management)
            dev_mode: Print debug information
        """
        self.output_dir = output_dir
        self.result_dir = result_dir
        self.model_size = model_size
        self.exp_dir = exp_dir
        self.dev_mode = dev_mode
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
    
    def _get_versioned_path(self, base_name: str, ext: str) -> str:
        """
        Get a versioned file path that doesn't overwrite existing files.
        
        Args:
            base_name: Base file name (without extension)
            ext: File extension
            
        Returns:
            Full path with version suffix if needed
        """
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_name = f"{base_name}_{timestamp}"
        return os.path.join(self.result_dir, f"{versioned_name}.{ext}")
    
    def save_embeddings(
        self,
        output: EmbeddingOutput,
        mode_subdir: str = "zero_shot"
    ) -> Dict[str, str]:
        """
        Save embedding output to files.
        
        Args:
            output: EmbeddingOutput object
            mode_subdir: Subdirectory for the mode (zero_shot, supervised, unsupervised)
            
        Returns:
            Dictionary with saved file paths
        """
        # Create mode-specific output directory
        mode_dir = os.path.join(self.output_dir, mode_subdir)
        os.makedirs(mode_dir, exist_ok=True)
        
        # Base name for files
        base_name = f"{output.dataset_name}_{output.mode}"
        
        # Save embeddings to output dir (intermediate)
        emb_path = os.path.join(mode_dir, f"{base_name}_embeddings.npy")
        np.save(emb_path, output.embeddings)
        
        # Save labels if available
        label_path = None
        if output.labels is not None:
            label_path = os.path.join(mode_dir, f"{base_name}_labels.npy")
            np.save(label_path, output.labels)
        
        # Save metadata
        meta_path = os.path.join(mode_dir, f"{base_name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(output.to_dict(), f, indent=2)
        
        # Save to result directory with THETA-compatible structure:
        # If exp_dir is set, save to {exp_dir}/embeddings/
        # Otherwise fallback: /result/{model_size}/{dataset}/{mode}/embeddings/
        if self.exp_dir:
            dataset_result_dir = os.path.join(self.exp_dir, "embeddings")
        else:
            dataset_result_dir = os.path.join(
                self.result_dir, 
                self.model_size,           # e.g., 0.6B, 4B, 8B
                output.dataset_name, 
                output.mode,               # e.g., zero_shot, supervised, unsupervised
                "embeddings"
            )
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        # THETA expects: {dataset}_{mode}_embeddings.npy (no timestamp)
        result_emb_path = os.path.join(
            dataset_result_dir,
            f"{output.dataset_name}_{output.mode}_embeddings.npy"
        )
        np.save(result_emb_path, output.embeddings)
        
        # Also save as embeddings.npy for compatibility
        simple_emb_path = os.path.join(dataset_result_dir, "embeddings.npy")
        np.save(simple_emb_path, output.embeddings)
        
        result_meta_path = os.path.join(
            dataset_result_dir,
            f"{output.dataset_name}_{output.mode}_metadata.json"
        )
        with open(result_meta_path, 'w') as f:
            json.dump(output.to_dict(), f, indent=2)
        
        # Save simple metadata.json
        simple_meta = {
            'num_documents': output.embeddings.shape[0],
            'embedding_dim': output.embeddings.shape[1],
            'model_size': self.model_size,
            'mode': output.mode,
        }
        simple_meta_path = os.path.join(dataset_result_dir, "metadata.json")
        with open(simple_meta_path, 'w') as f:
            json.dump(simple_meta, f, indent=2)
        
        # Save labels to result dir too
        result_label_path = None
        if output.labels is not None:
            result_label_path = os.path.join(
                dataset_result_dir,
                f"{output.dataset_name}_{output.mode}_labels.npy"
            )
            np.save(result_label_path, output.labels)
        
        saved_paths = {
            "embeddings": emb_path,
            "labels": label_path,
            "metadata": meta_path,
            "result_embeddings": result_emb_path,
            "result_labels": result_label_path,
            "result_metadata": result_meta_path
        }
        
        if self.dev_mode:
            print(f"[DEV] Saved files:")
            for key, path in saved_paths.items():
                if path:
                    print(f"[DEV]   {key}: {path}")
        
        print(f"Saved to result (THETA-compatible): {dataset_result_dir}")
        
        return saved_paths
    
    def load_embeddings(
        self,
        dataset_name: str,
        mode: str = "zero_shot"
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Load embeddings from files.
        
        Args:
            dataset_name: Name of the dataset
            mode: Embedding mode
            
        Returns:
            embeddings, labels, metadata
        """
        mode_dir = os.path.join(self.output_dir, mode)
        base_name = f"{dataset_name}_{mode}"
        
        # Load embeddings
        emb_path = os.path.join(mode_dir, f"{base_name}_embeddings.npy")
        embeddings = np.load(emb_path)
        
        # Load labels if exist
        label_path = os.path.join(mode_dir, f"{base_name}_labels.npy")
        labels = np.load(label_path) if os.path.exists(label_path) else None
        
        # Load metadata
        meta_path = os.path.join(mode_dir, f"{base_name}_metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        return embeddings, labels, metadata


if __name__ == "__main__":
    # Test embedder
    print("=" * 60)
    print("Testing QwenEmbedder")
    print("=" * 60)
    
    embedder = QwenEmbedder(dev_mode=True)
    
    # Test with sample texts
    test_texts = [
        "This is a test sentence for embedding.",
        "Another sample text to verify the model works.",
        "Machine learning is transforming natural language processing."
    ]
    
    embeddings = embedder.embed_texts(test_texts, show_progress=False)
    print(f"\nTest embeddings shape: {embeddings.shape}")
    
    # Validate
    stats = embedder.validate_embeddings(embeddings)
    print(f"\nValidation stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")



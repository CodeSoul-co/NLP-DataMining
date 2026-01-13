#!/usr/bin/env python3
"""
Engine B: Embedding Generation
Generates document and vocabulary embeddings for ETM

Supports two modes:
1. Qwen mode: Uses Qwen3-Embedding model for high-quality embeddings (requires model path)
2. Simple mode: Uses random embeddings for testing/demo purposes
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class SimpleEmbeddingModel(nn.Module):
    """Simple embedding model for demonstration/fallback"""
    def __init__(self, vocab_size, embed_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        return self.fc(embedded)

class QwenEmbedder:
    """Qwen embedding wrapper matching THETA-main interface"""
    
    def __init__(self, model_path: str, device: torch.device, max_length: int = 512, batch_size: int = 32):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._load_model()
    
    def _load_model(self):
        """Load Qwen model"""
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        self.embedding_dim = self.model.config.hidden_size
        self.model.eval()
    
    def embed_texts(self, texts: list, show_progress: bool = True) -> np.ndarray:
        """Embed a list of texts"""
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Mean pooling
                last_hidden = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Normalize
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings[i:i + len(batch_texts)] = batch_embeddings.cpu().float().numpy()
        
        return embeddings

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings')
    parser.add_argument('--vocab', required=True, help='Vocabulary JSON path')
    parser.add_argument('--bow', required=True, help='BOW NPZ path')
    parser.add_argument('--doc_emb_output', required=True, help='Document embeddings output path')
    parser.add_argument('--vocab_emb_output', required=True, help='Vocabulary embeddings output path')
    parser.add_argument('--job_id', required=True, help='Job ID')
    parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--model_path', type=str, default=None, help='Path to Qwen embedding model (optional)')
    parser.add_argument('--data_csv', type=str, default=None, help='Path to original data CSV for document embeddings')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info(f"Processing job {args.job_id}")
    
    try:
        # Load vocabulary
        with open(args.vocab, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab_list = vocab_data.get('vocab', list(vocab_data.keys()) if isinstance(vocab_data, dict) else vocab_data)
        vocab_size = len(vocab_list)
        
        # Load BOW matrix
        bow_data = np.load(args.bow)
        bow_matrix = bow_data['bow']  # Shape: (N, V)
        
        logger.info(f"Vocabulary size: {vocab_size}")
        logger.info(f"BOW matrix shape: {bow_matrix.shape}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Check if Qwen model is available
        use_qwen = args.model_path and Path(args.model_path).exists()
        
        if use_qwen:
            logger.info(f"Using Qwen embedding model: {args.model_path}")
            embedder = QwenEmbedder(args.model_path, device)
            embed_dim = embedder.embedding_dim
            
            # Generate vocabulary embeddings
            logger.info("Generating vocabulary embeddings...")
            vocab_embeddings = embedder.embed_texts(vocab_list, show_progress=True)
            
            # Generate document embeddings
            if args.data_csv and Path(args.data_csv).exists():
                # Load original texts for better embeddings
                import pandas as pd
                df = pd.read_csv(args.data_csv)
                texts = df['text'].fillna('').astype(str).tolist()
                logger.info(f"Generating document embeddings from {len(texts)} texts...")
                doc_embeddings = embedder.embed_texts(texts, show_progress=True)
            else:
                # Fallback: weighted average of word embeddings
                logger.info("Generating document embeddings from BOW (weighted average)...")
                doc_embeddings = np.zeros((bow_matrix.shape[0], embed_dim), dtype=np.float32)
                for doc_idx in tqdm(range(bow_matrix.shape[0]), desc="Doc embeddings"):
                    doc_bow = bow_matrix[doc_idx]
                    if doc_bow.sum() > 0:
                        weighted_emb = np.zeros(embed_dim, dtype=np.float32)
                        total_weight = 0
                        for word_idx, weight in enumerate(doc_bow):
                            if weight > 0 and word_idx < vocab_size:
                                weighted_emb += weight * vocab_embeddings[word_idx]
                                total_weight += weight
                        if total_weight > 0:
                            doc_embeddings[doc_idx] = weighted_emb / total_weight
                # Normalize
                norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                doc_embeddings = doc_embeddings / norms
        else:
            # Simple mode: random embeddings for testing
            logger.info("Using simple embedding model (random initialization)")
            embed_dim = args.embed_dim
            model = SimpleEmbeddingModel(vocab_size, embed_dim).to(device)
            
            # Generate vocabulary embeddings
            model.eval()
            with torch.no_grad():
                vocab_indices = torch.arange(vocab_size).to(device)
                vocab_embeddings = model(vocab_indices).cpu().numpy()
            
            # Normalize vocabulary embeddings
            norms = np.linalg.norm(vocab_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vocab_embeddings = vocab_embeddings / norms
            
            # Generate document embeddings (weighted average of word embeddings)
            doc_embeddings = np.zeros((bow_matrix.shape[0], embed_dim), dtype=np.float32)
            for doc_idx in range(bow_matrix.shape[0]):
                doc_bow = bow_matrix[doc_idx]
                if doc_bow.sum() > 0:
                    weighted_emb = np.zeros(embed_dim, dtype=np.float32)
                    total_weight = 0
                    for word_idx, weight in enumerate(doc_bow):
                        if weight > 0 and word_idx < vocab_size:
                            weighted_emb += weight * vocab_embeddings[word_idx]
                            total_weight += weight
                    if total_weight > 0:
                        doc_embeddings[doc_idx] = weighted_emb / total_weight
            
            # Normalize document embeddings
            norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            doc_embeddings = doc_embeddings / norms
        
        # Save embeddings
        np.save(args.doc_emb_output, doc_embeddings.astype(np.float32))
        np.save(args.vocab_emb_output, vocab_embeddings.astype(np.float32))
        
        logger.info(f"Generated document embeddings: {doc_embeddings.shape}")
        logger.info(f"Generated vocabulary embeddings: {vocab_embeddings.shape}")
        logger.info(f"Saved document embeddings to {args.doc_emb_output}")
        logger.info(f"Saved vocabulary embeddings to {args.vocab_emb_output}")
        
    except Exception as e:
        logger.error(f"Error processing job {args.job_id}: {str(e)}")
        raise

if __name__ == "__main__":
    main()

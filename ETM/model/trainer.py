"""
Unified Trainer

Supports training for all topic models, including:
- ETM: Requires doc_embeddings + word_embeddings + BOW
- CTM: Requires doc_embeddings + BOW
- LDA: Only requires BOW
- DTM: Requires doc_embeddings + word_embeddings + BOW + time_indices
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List, Any, Union
from tqdm import tqdm
from pathlib import Path

from .registry import get_model_class, get_model_info


class TopicModelTrainer:
    """
    Unified topic model trainer
    
    Supports all registered topic models, automatically handles different model input requirements.
    """
    
    def __init__(
        self,
        model_type: str = 'etm',
        num_topics: int = 20,
        vocab_size: int = 5000,
        device: str = 'auto',
        **model_kwargs
    ):
        """
        Initialize trainer
        
        Args:
            model_type: Model type ('etm', 'ctm', 'lda', 'neural_lda', 'dtm')
            num_topics: Number of topics
            vocab_size: Vocabulary size
            device: Device ('auto', 'cuda', 'cpu')
            **model_kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Get model info
        self.model_info = get_model_info(model_type)
        if self.model_info is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_kwargs = model_kwargs
        self.model = None
        self.vocab = None
        self.training_history = []
    
    def _create_model(
        self,
        word_embeddings: Optional[np.ndarray] = None,
        doc_embedding_dim: int = 1024
    ) -> nn.Module:
        """Create model instance"""
        ModelClass = get_model_class(self.model_type)
        
        # Merge default parameters and user parameters
        params = {
            'vocab_size': self.vocab_size,
            'num_topics': self.num_topics,
        }
        
        # Add model-specific parameters
        if self.model_info.supports_embeddings:
            params['doc_embedding_dim'] = doc_embedding_dim
        
        if self.model_info.supports_pretrained_words and word_embeddings is not None:
            params['word_embeddings'] = torch.tensor(word_embeddings, dtype=torch.float32)
            params['word_embedding_dim'] = word_embeddings.shape[1]
        
        # Add user-defined parameters
        params.update(self.model_kwargs)
        
        return ModelClass(**params)
    
    def train_sklearn_lda(
        self,
        bow_matrix: np.ndarray,
        vocab: List[str],
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train sklearn LDA
        
        Args:
            bow_matrix: BOW matrix (num_docs, vocab_size)
            vocab: Vocabulary list
            save_dir: Save directory
            
        Returns:
            Training results
        """
        from .lda import SklearnLDA
        
        print(f"Training SklearnLDA with {self.num_topics} topics...")
        
        # Create model
        model = SklearnLDA(
            vocab_size=self.vocab_size,
            num_topics=self.num_topics,
            **self.model_kwargs
        )
        
        # Train
        start_time = time.time()
        model.fit(bow_matrix)
        train_time = time.time() - start_time
        
        # Get results
        theta = model.get_theta()
        beta = model.get_beta()
        topic_words = model.get_topic_words(vocab, top_k=10)
        perplexity = model.get_perplexity(bow_matrix)
        
        self.model = model
        self.vocab = vocab
        
        results = {
            'theta': theta,
            'beta': beta,
            'topic_words': topic_words,
            'perplexity': perplexity,
            'train_time': train_time,
            'num_topics': self.num_topics,
            'model_type': 'lda'
        }
        
        # Save results
        if save_dir:
            self._save_results(results, save_dir)
        
        print(f"Training completed in {train_time:.2f}s, Perplexity: {perplexity:.2f}")
        return results
    
    def train_neural_model(
        self,
        bow_matrix: np.ndarray,
        doc_embeddings: Optional[np.ndarray],
        word_embeddings: Optional[np.ndarray],
        vocab: List[str],
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.002,
        save_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
        time_indices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train neural network topic model (ETM, CTM, NeuralLDA, DTM)
        
        Args:
            bow_matrix: BOW matrix (num_docs, vocab_size)
            doc_embeddings: Document embeddings (num_docs, embedding_dim)
            word_embeddings: Word embeddings (vocab_size, embedding_dim)
            vocab: Vocabulary list
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_dir: Save directory
            early_stopping_patience: Early stopping patience
            time_indices: Time indices (DTM only)
            
        Returns:
            Training results
        """
        print(f"Training {self.model_type.upper()} with {self.num_topics} topics...")
        
        # Determine embedding dimension
        doc_embedding_dim = doc_embeddings.shape[1] if doc_embeddings is not None else 1024
        
        # Create model
        self.model = self._create_model(word_embeddings, doc_embedding_dim)
        self.model = self.model.to(self.device)
        self.vocab = vocab
        
        # Prepare data
        bow_tensor = torch.tensor(bow_matrix, dtype=torch.float32)
        
        if doc_embeddings is not None:
            doc_emb_tensor = torch.tensor(doc_embeddings, dtype=torch.float32)
        else:
            # For models that don't need embeddings, create dummy tensor
            doc_emb_tensor = torch.zeros(bow_matrix.shape[0], doc_embedding_dim)
        
        if time_indices is not None:
            time_tensor = torch.tensor(time_indices, dtype=torch.long)
            dataset = TensorDataset(doc_emb_tensor, bow_tensor, time_tensor)
        else:
            dataset = TensorDataset(doc_emb_tensor, bow_tensor)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                if time_indices is not None:
                    doc_emb, bow, time_idx = batch
                    doc_emb = doc_emb.to(self.device)
                    bow = bow.to(self.device)
                    time_idx = time_idx.to(self.device)
                    output = self.model(doc_emb, bow, time_indices=time_idx)
                else:
                    doc_emb, bow = batch
                    doc_emb = doc_emb.to(self.device)
                    bow = bow.to(self.device)
                    output = self.model(doc_emb, bow)
                
                loss = output.get('loss') or output.get('total_loss')
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon_loss += output.get('recon_loss', torch.tensor(0)).item()
                epoch_kl_loss += output.get('kl_loss', torch.tensor(0)).item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon_loss / num_batches
            avg_kl = epoch_kl_loss / num_batches
            
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'recon_loss': avg_recon,
                'kl_loss': avg_kl
            })
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        train_time = time.time() - start_time
        
        # Get final results
        self.model.eval()
        with torch.no_grad():
            # Get theta for all documents
            all_thetas = []
            for i in range(0, len(bow_tensor), batch_size):
                batch_doc = doc_emb_tensor[i:i+batch_size].to(self.device)
                batch_bow = bow_tensor[i:i+batch_size].to(self.device)
                output = self.model(batch_doc, batch_bow)
                all_thetas.append(output['theta'].cpu().numpy())
            theta = np.concatenate(all_thetas, axis=0)
            
            # Get beta
            if hasattr(self.model, 'get_beta'):
                beta = self.model.get_beta()
                if isinstance(beta, torch.Tensor):
                    beta = beta.cpu().numpy()
            else:
                beta = None
            
            # Get topic words
            if hasattr(self.model, 'get_topic_words'):
                topic_words = self.model.get_topic_words(vocab, top_k=10)
            else:
                topic_words = self._extract_topic_words(beta, vocab, top_k=10)
        
        results = {
            'theta': theta,
            'beta': beta,
            'topic_words': topic_words,
            'train_time': train_time,
            'final_loss': best_loss,
            'num_topics': self.num_topics,
            'model_type': self.model_type,
            'training_history': self.training_history
        }
        
        # Save results
        if save_dir:
            self._save_results(results, save_dir)
        
        print(f"Training completed in {train_time:.2f}s, Final Loss: {best_loss:.4f}")
        return results
    
    def _extract_topic_words(
        self,
        beta: np.ndarray,
        vocab: List[str],
        top_k: int = 10
    ) -> Dict[str, List[str]]:
        """Extract topic words from beta matrix"""
        topic_words = {}
        for k in range(beta.shape[0]):
            top_indices = np.argsort(-beta[k])[:top_k]
            topic_words[f"topic_{k}"] = [vocab[i] for i in top_indices]
        return topic_words
    
    def _save_results(self, results: Dict[str, Any], save_dir: str):
        """Save training results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save theta
        if results.get('theta') is not None:
            np.save(os.path.join(save_dir, f'theta_{self.model_type}.npy'), results['theta'])
        
        # Save beta
        if results.get('beta') is not None:
            np.save(os.path.join(save_dir, f'beta_{self.model_type}.npy'), results['beta'])
        
        # Save topic words
        if results.get('topic_words') is not None:
            with open(os.path.join(save_dir, f'topic_words_{self.model_type}.json'), 'w', encoding='utf-8') as f:
                json.dump(results['topic_words'], f, ensure_ascii=False, indent=2)
        
        # Save training info
        info = {
            'model_type': self.model_type,
            'num_topics': self.num_topics,
            'vocab_size': self.vocab_size,
            'train_time': results.get('train_time'),
            'final_loss': results.get('final_loss'),
            'perplexity': results.get('perplexity'),
        }
        with open(os.path.join(save_dir, f'training_info_{self.model_type}.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        # Save model
        if self.model is not None:
            if hasattr(self.model, 'save_model'):
                self.model.save_model(os.path.join(save_dir, f'model_{self.model_type}.pt'))
            elif hasattr(self.model, 'state_dict'):
                torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_{self.model_type}.pt'))
        
        print(f"Results saved to {save_dir}")


def train_baseline_models(
    dataset: str,
    mode: str = 'zero_shot',
    num_topics: int = 20,
    vocab_size: int = 5000,
    models: List[str] = None,
    base_result_dir: str = '/root/autodl-tmp/result/0.6B',
    epochs: int = 50,
    batch_size: int = 64
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to train all baseline models
    
    Args:
        dataset: Dataset name
        mode: Embedding mode ('zero_shot', 'supervised', 'unsupervised')
        num_topics: Number of topics
        vocab_size: Vocabulary size
        models: List of models to train, default ['lda', 'ctm', 'etm']
        base_result_dir: Base result directory
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Training results for all models
    """
    if models is None:
        models = ['lda', 'ctm', 'etm']
    
    # Load data
    bow_dir = os.path.join(base_result_dir, dataset, 'bow')
    emb_dir = os.path.join(base_result_dir, dataset, mode, 'embeddings')
    save_dir = os.path.join(base_result_dir, dataset, mode, 'model')
    
    # Load BOW
    import scipy.sparse as sp
    bow_path = os.path.join(bow_dir, 'bow_matrix.npz')
    if os.path.exists(bow_path):
        bow_matrix = sp.load_npz(bow_path).toarray()
    else:
        raise FileNotFoundError(f"BOW matrix not found: {bow_path}")
    
    # Load vocabulary
    vocab_path = os.path.join(bow_dir, 'vocab.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Load embeddings
    doc_emb_path = os.path.join(emb_dir, 'doc_embeddings.npy')
    doc_embeddings = np.load(doc_emb_path) if os.path.exists(doc_emb_path) else None
    
    vocab_emb_path = os.path.join(emb_dir, 'vocab_embeddings.npy')
    word_embeddings = np.load(vocab_emb_path) if os.path.exists(vocab_emb_path) else None
    
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        trainer = TopicModelTrainer(
            model_type=model_type,
            num_topics=num_topics,
            vocab_size=len(vocab)
        )
        
        if model_type == 'lda':
            result = trainer.train_sklearn_lda(
                bow_matrix=bow_matrix,
                vocab=vocab,
                save_dir=save_dir
            )
        else:
            result = trainer.train_neural_model(
                bow_matrix=bow_matrix,
                doc_embeddings=doc_embeddings,
                word_embeddings=word_embeddings,
                vocab=vocab,
                epochs=epochs,
                batch_size=batch_size,
                save_dir=save_dir
            )
        
        results[model_type] = result
    
    return results

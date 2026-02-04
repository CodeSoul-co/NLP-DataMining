"""
Baseline Trainer - Dedicated trainer for baseline models

Trains LDA, CTM and other baseline models from raw CSV files.
Does not use Qwen embedding, instead uses sklearn BOW and SBERT embedding.

Usage:
    python -m model.baseline_trainer --dataset hatespeech --models lda,ctm --num_topics 20
"""

import os
import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .baseline_data import BaselineDataProcessor, prepare_baseline_data
from .lda import SklearnLDA
from .ctm import CTM, ZeroShotTM, CombinedTM
from .etm_original import OriginalETM, train_word2vec_embeddings

# Import config for environment variable support
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RESULT_DIR, get_sbert_model_path


class BaselineTrainer:
    """
    Baseline model trainer
    
    Supports training from raw CSV files:
    - LDA: Uses sklearn, only requires BOW
    - CTM: Uses SBERT embedding + BOW
    """
    
    def __init__(
        self,
        dataset: str,
        num_topics: int = 20,
        vocab_size: int = 5000,
        data_dir: str = None,
        result_dir: str = None,
        device: str = 'auto',
        job_id: str = None
    ):
        """
        Initialize trainer
        
        Args:
            dataset: Dataset name
            num_topics: Number of topics
            vocab_size: Vocabulary size
            data_dir: Data directory
            result_dir: Result directory
            device: Device
        """
        self.dataset = dataset
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.job_id = job_id
        
        # Use environment variable defaults if not specified
        self.data_dir = data_dir if data_dir else str(DATA_DIR)
        self.result_dir = result_dir if result_dir else os.path.join(str(RESULT_DIR), 'baseline')
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Data
        self.texts = None
        self.labels = None
        self.bow_matrix = None
        self.vocab = None
        self.sbert_embeddings = None
        
        # Result directory
        self.output_dir = os.path.join(result_dir, dataset)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_data(
        self,
        generate_sbert: bool = True,
        sbert_model: str = None
    ):
        """
        Prepare data
        
        Args:
            generate_sbert: Whether to generate SBERT embedding
            sbert_model: SBERT model name
        """
        # Use environment variable default for SBERT model
        if sbert_model is None:
            sbert_model = get_sbert_model_path()
        # Check if processed data already exists
        bow_path = os.path.join(self.output_dir, 'bow_matrix.npz')
        vocab_path = os.path.join(self.output_dir, 'vocab.json')
        
        if os.path.exists(bow_path) and os.path.exists(vocab_path):
            print("Loading existing processed data...")
            self.bow_matrix = sp.load_npz(bow_path)
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            
            # Load SBERT embedding (if exists)
            sbert_path = os.path.join(self.output_dir, 'sbert_embeddings.npy')
            if os.path.exists(sbert_path):
                self.sbert_embeddings = np.load(sbert_path)
            elif generate_sbert:
                # Need to generate SBERT embeddings but not exists, need to load raw text
                print("SBERT embeddings not found, generating...")
                from .baseline_data import BaselineDataProcessor
                processor = BaselineDataProcessor(max_features=self.vocab_size)
                # Find CSV file
                csv_path = os.path.join(self.data_dir, self.dataset, f"{self.dataset}_text_only.csv")
                if not os.path.exists(csv_path):
                    # Try other possible filenames
                    data_dir = os.path.join(self.data_dir, self.dataset)
                    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                    if csv_files:
                        csv_path = os.path.join(data_dir, csv_files[0])
                    else:
                        raise FileNotFoundError(f"No CSV file found in {data_dir}")
                processor.load_csv(csv_path)
                self.texts = processor.texts
                self.sbert_embeddings = processor.get_sbert_embeddings(
                    texts=self.texts,
                    model_name=sbert_model
                )
                # Save SBERT embeddings
                np.save(sbert_path, self.sbert_embeddings)
                print(f"SBERT embeddings saved to {sbert_path}")
            
            print(f"BOW matrix: {self.bow_matrix.shape}")
            print(f"Vocab size: {len(self.vocab)}")
            if self.sbert_embeddings is not None:
                print(f"SBERT embeddings: {self.sbert_embeddings.shape}")
        else:
            print("Processing data from CSV...")
            result = prepare_baseline_data(
                dataset=self.dataset,
                vocab_size=self.vocab_size,
                data_dir=self.data_dir,
                save_dir=self.result_dir,
                generate_sbert=generate_sbert,
                sbert_model=sbert_model
            )
            
            self.texts = result['texts']
            self.labels = result['labels']
            self.bow_matrix = result['bow_matrix']
            self.vocab = result['vocab']
            self.sbert_embeddings = result.get('sbert_embeddings')
    
    def train_lda(
        self,
        max_iter: int = 100,
        learning_method: str = 'batch'
    ) -> Dict[str, Any]:
        """
        Train LDA model
        
        Args:
            max_iter: Maximum iterations
            learning_method: Learning method
            
        Returns:
            Training results
        """
        print("\n" + "="*60)
        print("Training LDA (sklearn)")
        print("="*60)
        
        if self.bow_matrix is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        
        # Create model
        model = SklearnLDA(
            vocab_size=self.bow_matrix.shape[1],
            num_topics=self.num_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            dev_mode=True
        )
        
        # Train
        start_time = time.time()
        model.fit(self.bow_matrix)
        train_time = time.time() - start_time
        
        # Get results
        theta = model.get_theta()
        beta = model.get_beta()
        topic_words = model.get_topic_words(self.vocab, top_k=10)
        perplexity = model.get_perplexity(self.bow_matrix)
        
        # Save results
        model_dir = os.path.join(self.output_dir, 'lda')
        os.makedirs(model_dir, exist_ok=True)
        
        np.save(os.path.join(model_dir, f'theta_k{self.num_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{self.num_topics}.npy'), beta)
        
        with open(os.path.join(model_dir, f'topic_words_k{self.num_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_words, f, ensure_ascii=False, indent=2)
        
        model.save_model(os.path.join(model_dir, f'model_k{self.num_topics}.pkl'))
        
        # Save training info
        info = {
            'model': 'lda',
            'num_topics': self.num_topics,
            'vocab_size': len(self.vocab),
            'num_docs': self.bow_matrix.shape[0],
            'train_time': train_time,
            'perplexity': perplexity,
            'max_iter': max_iter
        }
        with open(os.path.join(model_dir, f'info_k{self.num_topics}.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nLDA Training Complete:")
        print(f"  - Train time: {train_time:.2f}s")
        print(f"  - Perplexity: {perplexity:.2f}")
        print(f"  - Results saved to: {model_dir}")
        
        # Print topic words example
        print("\nTop 10 words for first 3 topics:")
        for i in range(min(3, self.num_topics)):
            words = topic_words[f'topic_{i}'][:10]
            print(f"  Topic {i}: {', '.join(words)}")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta,
            'topic_words': topic_words,
            'perplexity': perplexity,
            'train_time': train_time
        }
    
    def train_ctm(
        self,
        inference_type: str = 'zeroshot',
        model_type: str = 'prodLDA',
        hidden_sizes: tuple = (100, 100),
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.002,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train CTM model
        
        Args:
            inference_type: Inference type ('zeroshot' or 'combined')
            model_type: Model type ('prodLDA' or 'LDA')
            hidden_sizes: Hidden layer sizes
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training results
        """
        print("\n" + "="*60)
        print(f"Training CTM ({inference_type})")
        print("="*60)
        
        if self.bow_matrix is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        
        if self.sbert_embeddings is None:
            raise RuntimeError(
                "SBERT embeddings not available. "
                "Call prepare_data(generate_sbert=True) first."
            )
        
        # Create model
        model = CTM(
            vocab_size=self.bow_matrix.shape[1],
            num_topics=self.num_topics,
            doc_embedding_dim=self.sbert_embeddings.shape[1],
            hidden_sizes=hidden_sizes,
            model_type=model_type,
            inference_type=inference_type,
            dev_mode=True
        )
        model = model.to(self.device)
        
        # Prepare data
        bow_dense = self.bow_matrix.toarray() if sp.issparse(self.bow_matrix) else self.bow_matrix
        bow_tensor = torch.tensor(bow_dense, dtype=torch.float32)
        emb_tensor = torch.tensor(self.sbert_embeddings, dtype=torch.float32)
        
        dataset = TensorDataset(emb_tensor, bow_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for emb, bow in pbar:
                emb = emb.to(self.device)
                bow = bow.to(self.device)
                
                output = model(emb, bow)
                loss = output['loss']
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = epoch_loss / num_batches
            training_history.append({'epoch': epoch + 1, 'loss': avg_loss})
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        train_time = time.time() - start_time
        
        # Get results
        model.eval()
        with torch.no_grad():
            # Get theta
            all_thetas = []
            for i in range(0, len(bow_tensor), batch_size):
                emb = emb_tensor[i:i+batch_size].to(self.device)
                bow = bow_tensor[i:i+batch_size].to(self.device)
                output = model(emb, bow)
                all_thetas.append(output['theta'].cpu().numpy())
            theta = np.concatenate(all_thetas, axis=0)
            
            # Get beta and topic words
            beta = model.get_beta()
            topic_words = model.get_topic_words(self.vocab, top_k=10)
        
        # Save results
        model_dir = os.path.join(self.output_dir, f'ctm_{inference_type}')
        os.makedirs(model_dir, exist_ok=True)
        
        np.save(os.path.join(model_dir, f'theta_k{self.num_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{self.num_topics}.npy'), beta)
        
        with open(os.path.join(model_dir, f'topic_words_k{self.num_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_words, f, ensure_ascii=False, indent=2)
        
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_k{self.num_topics}.pt'))
        
        # Save training info
        info = {
            'model': 'ctm',
            'inference_type': inference_type,
            'model_type': model_type,
            'num_topics': self.num_topics,
            'vocab_size': len(self.vocab),
            'num_docs': self.bow_matrix.shape[0],
            'embedding_dim': self.sbert_embeddings.shape[1],
            'train_time': train_time,
            'final_loss': best_loss,
            'epochs_trained': len(training_history)
        }
        with open(os.path.join(model_dir, f'info_k{self.num_topics}.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        # Save training history (for loss curve visualization)
        training_history_data = {
            'train_loss': [h['loss'] for h in training_history],
            'epochs_trained': len(training_history),
            'best_loss': best_loss
        }
        with open(os.path.join(model_dir, f'training_history_k{self.num_topics}.json'), 'w') as f:
            json.dump(training_history_data, f, indent=2)
        
        print(f"\nCTM ({inference_type}) training completed:")
        print(f"  - Train time: {train_time:.2f}s")
        print(f"  - Final loss: {best_loss:.4f}")
        print(f"  - Results saved to: {model_dir}")
        
        # Print topic words example
        print("\nTop 10 words for first 3 topics:")
        for i in range(min(3, self.num_topics)):
            words = topic_words[f'topic_{i}'][:10]
            print(f"  Topic {i}: {', '.join(words)}")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta,
            'topic_words': topic_words,
            'train_time': train_time,
            'final_loss': best_loss,
            'training_history': training_history
        }
    
    def train_etm(
        self,
        embedding_dim: int = 300,
        hidden_dim: int = 800,
        dropout: float = 0.5,
        train_embeddings: bool = True,
        use_pretrained_embeddings: bool = True,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.002,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train original ETM model (Baseline version, without Qwen)
        
        Args:
            embedding_dim: Word embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            train_embeddings: Whether to train word embeddings
            use_pretrained_embeddings: Whether to use pretrained embeddings (Word2Vec)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training results
        """
        print("\n" + "="*60)
        print("Training Original ETM (Baseline)")
        print("="*60)
        
        if self.bow_matrix is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")
        
        # Prepare word embeddings
        word_embeddings = None
        if use_pretrained_embeddings and self.texts is not None:
            # Train Word2Vec embeddings
            emb_path = os.path.join(self.output_dir, f'word2vec_embeddings_{embedding_dim}.npy')
            if os.path.exists(emb_path):
                print(f"Loading existing Word2Vec embeddings from {emb_path}")
                word_embeddings = np.load(emb_path)
            else:
                word_embeddings = train_word2vec_embeddings(
                    texts=self.texts,
                    vocab=self.vocab,
                    embedding_dim=embedding_dim
                )
                np.save(emb_path, word_embeddings)
                print(f"Word2Vec embeddings saved to {emb_path}")
        
        # Create model
        model = OriginalETM(
            vocab_size=self.bow_matrix.shape[1],
            num_topics=self.num_topics,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            word_embeddings=word_embeddings,
            train_embeddings=train_embeddings,
            dev_mode=True
        )
        model = model.to(self.device)
        
        # Prepare data
        bow_dense = self.bow_matrix.toarray() if sp.issparse(self.bow_matrix) else self.bow_matrix
        bow_tensor = torch.tensor(bow_dense, dtype=torch.float32)
        
        dataset = TensorDataset(bow_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for (bow,) in pbar:
                bow = bow.to(self.device)
                
                # Original ETM doesn't need doc_embeddings, pass dummy
                dummy_emb = torch.zeros(bow.size(0), 1).to(self.device)
                output = model(dummy_emb, bow)
                loss = output['loss']
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += output['recon_loss'].item()
                epoch_kl += output['kl_loss'].item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon / num_batches
            avg_kl = epoch_kl / num_batches
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'recon_loss': avg_recon,
                'kl_loss': avg_kl
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        train_time = time.time() - start_time
        
        # Get results
        model.eval()
        with torch.no_grad():
            # Get theta
            all_thetas = []
            for i in range(0, len(bow_tensor), batch_size):
                bow = bow_tensor[i:i+batch_size].to(self.device)
                theta, _, _ = model.encode(bow)
                all_thetas.append(theta.cpu().numpy())
            theta = np.concatenate(all_thetas, axis=0)
            
            # Get beta and topic words
            beta = model.get_beta().cpu().numpy()
            topic_words = model.get_topic_words(self.vocab, top_k=10)
        
        # Save results
        model_dir = os.path.join(self.output_dir, 'etm')
        os.makedirs(model_dir, exist_ok=True)
        
        np.save(os.path.join(model_dir, f'theta_k{self.num_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{self.num_topics}.npy'), beta)
        
        with open(os.path.join(model_dir, f'topic_words_k{self.num_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_words, f, ensure_ascii=False, indent=2)
        
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_k{self.num_topics}.pt'))
        
        # Save training info
        info = {
            'model': 'etm_original',
            'num_topics': self.num_topics,
            'vocab_size': len(self.vocab),
            'num_docs': self.bow_matrix.shape[0],
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'train_time': train_time,
            'final_loss': best_loss,
            'epochs_trained': len(training_history),
            'use_pretrained_embeddings': use_pretrained_embeddings
        }
        with open(os.path.join(model_dir, f'info_k{self.num_topics}.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        # Save training history (for loss curve visualization)
        training_history_data = {
            'train_loss': [h['loss'] for h in training_history],
            'recon_loss': [h['recon_loss'] for h in training_history],
            'kl_loss': [h['kl_loss'] for h in training_history],
            'epochs_trained': len(training_history),
            'best_loss': best_loss
        }
        with open(os.path.join(model_dir, f'training_history_k{self.num_topics}.json'), 'w') as f:
            json.dump(training_history_data, f, indent=2)
        
        print(f"\nOriginal ETM Training Complete:")
        print(f"  - Train time: {train_time:.2f}s")
        print(f"  - Final loss: {best_loss:.4f}")
        print(f"  - Results saved to: {model_dir}")
        
        # 打印主题词示例
        print("\nTop 10 words for first 3 topics:")
        for i in range(min(3, self.num_topics)):
            words = topic_words[f'topic_{i}'][:10]
            print(f"  Topic {i}: {', '.join(words)}")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta,
            'topic_words': topic_words,
            'final_loss': best_loss,
            'train_time': train_time,
            'training_history': training_history
        }
    
    def train_dtm(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.002,
        hidden_dim: int = 256,
        embedding_dim: int = 300
    ) -> Dict[str, Any]:
        """
        Train DTM (Dynamic Topic Model)
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            hidden_dim: Hidden layer dimension
            embedding_dim: Word embedding dimension
            
        Returns:
            Training results
        """
        from .dtm import DTM
        
        print(f"\n{'='*60}")
        print(f"Training DTM: {self.dataset}")
        print(f"  - Topics: {self.num_topics}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"{'='*60}")
        
        # Load time slice information
        time_slices_path = os.path.join(self.output_dir, 'time_slices.json')
        time_indices_path = os.path.join(self.output_dir, 'time_indices.npy')
        
        if not os.path.exists(time_slices_path) or not os.path.exists(time_indices_path):
            raise ValueError(
                f"DTM requires time slice information, please run first:\n"
                f"  python prepare_data.py --dataset {self.dataset} --model dtm"
            )
        
        with open(time_slices_path, 'r') as f:
            time_info = json.load(f)
        time_indices = np.load(time_indices_path)
        
        num_time_slices = time_info['num_time_slices']
        print(f"  - Time slices: {num_time_slices}")
        print(f"  - Time range: {time_info['unique_times'][0]} - {time_info['unique_times'][-1]}")
        
        # Prepare data
        bow_dense = self.bow_matrix.toarray().astype(np.float32)
        bow_dense = bow_dense / (bow_dense.sum(axis=1, keepdims=True) + 1e-10)
        
        # Use SBERT embedding as document representation (if available)
        if self.sbert_embeddings is not None:
            doc_embeddings = self.sbert_embeddings.astype(np.float32)
            doc_embedding_dim = doc_embeddings.shape[1]
        else:
            # Use dimensionality-reduced BOW as document representation
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=min(256, self.bow_matrix.shape[1] - 1))
            doc_embeddings = svd.fit_transform(self.bow_matrix).astype(np.float32)
            doc_embedding_dim = doc_embeddings.shape[1]
        
        # Create dataset
        dataset = TensorDataset(
            torch.tensor(doc_embeddings),
            torch.tensor(bow_dense),
            torch.tensor(time_indices, dtype=torch.long)
        )
        
        # Split train and validation sets
        n_total = len(dataset)
        n_train = int(n_total * 0.9)
        n_val = n_total - n_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = DTM(
            vocab_size=len(self.vocab),
            num_topics=self.num_topics,
            time_slices=num_time_slices,
            doc_embedding_dim=doc_embedding_dim,
            word_embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Train
        training_history = []
        best_loss = float('inf')
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_recon = 0.0
            train_kl = 0.0
            
            for batch in train_loader:
                doc_emb, bow, time_idx = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                output = model(doc_emb, bow, time_idx)
                loss = output['total_loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * doc_emb.size(0)
                train_recon += output['recon_loss'].item() * doc_emb.size(0)
                train_kl += output['kl_loss'].item() * doc_emb.size(0)
            
            train_loss /= n_train
            train_recon /= n_train
            train_kl /= n_train
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_recon = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    doc_emb, bow, time_idx = [b.to(self.device) for b in batch]
                    output = model(doc_emb, bow, time_idx)
                    val_loss += output['total_loss'].item() * doc_emb.size(0)
                    val_recon += output['recon_loss'].item() * doc_emb.size(0)
            
            val_loss /= n_val
            val_recon /= n_val
            
            # Compute perplexity: exp(recon_loss)
            train_ppl = np.exp(train_recon)
            val_ppl = np.exp(val_recon)
            
            scheduler.step(val_loss)
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'recon_loss': train_recon,
                'kl_loss': train_kl,
                'train_ppl': train_ppl,
                'val_ppl': val_ppl
            })
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        train_time = time.time() - start_time
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Extract results
        model.eval()
        with torch.no_grad():
            # Get beta for all time slices
            all_betas = model.decoder.get_beta()  # (time_slices, num_topics, vocab_size)
            
            # Get theta
            all_theta = []
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                doc_emb, bow, time_idx = [b.to(self.device) for b in batch]
                theta, _, _ = model.encoder(doc_emb, time_idx)
                all_theta.append(theta.cpu().numpy())
            theta = np.vstack(all_theta)
        
        # Convert to numpy
        all_betas = all_betas.cpu().numpy()
        
        # Use last time slice beta as main beta (can also use average)
        beta = all_betas[-1]  # (num_topics, vocab_size)
        
        # Extract topic words
        topic_words = {}
        for k in range(self.num_topics):
            top_indices = beta[k].argsort()[-20:][::-1]
            topic_words[f'topic_{k}'] = [self.vocab[i] for i in top_indices]
        
        # Save results using ResultManager
        from utils.result_manager import ResultManager
        
        manager = ResultManager(
            result_dir=self.result_dir,
            dataset=self.dataset,
            model='dtm',
            num_topics=self.num_topics
        )
        
        # Compute final perplexity
        final_ppl = training_history[-1]['val_ppl'] if training_history else 0
        
        # Prepare training history data
        training_history_data = {
            'train_loss': [h['train_loss'] for h in training_history],
            'val_loss': [h['val_loss'] for h in training_history],
            'recon_loss': [h['recon_loss'] for h in training_history],
            'kl_loss': [h['kl_loss'] for h in training_history],
            'train_ppl': [h['train_ppl'] for h in training_history],
            'val_ppl': [h['val_ppl'] for h in training_history],
            'epochs_trained': len(training_history),
            'best_loss': best_loss,
            'final_ppl': final_ppl,
            'time_slices': num_time_slices,
            'time_range': [time_info['unique_times'][0], time_info['unique_times'][-1]]
        }
        
        # Prepare topic evolution data
        topic_evolution = {}
        for t_idx, t_year in enumerate(time_info['unique_times']):
            topic_evolution[str(t_year)] = {}
            for k in range(self.num_topics):
                top_indices = all_betas[t_idx, k].argsort()[-10:][::-1]
                topic_evolution[str(t_year)][f'topic_{k}'] = [self.vocab[i] for i in top_indices]
        
        # Save all results using ResultManager
        manager.save_all(
            theta=theta,
            beta=beta,
            vocab=self.vocab,
            topic_words=topic_words,
            num_topics=self.num_topics,
            beta_over_time=all_betas,
            topic_evolution=topic_evolution,
            training_history=training_history_data
        )
        
        print(f"\nDTM Training Complete:")
        print(f"  - Train time: {train_time:.2f}s")
        print(f"  - Best loss: {best_loss:.4f}")
        print(f"  - Final PPL: {final_ppl:.2f}")
        print(f"  - Time slices: {num_time_slices}")
        print(f"  - Results saved to: {manager.base_dir}")
        
        # Print topic words example
        print("\nTop 10 words for first 3 topics (latest time slice):")
        for i in range(min(3, self.num_topics)):
            words = topic_words[f'topic_{i}'][:10]
            print(f"  Topic {i}: {', '.join(words)}")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta,
            'beta_over_time': all_betas,
            'topic_words': topic_words,
            'topic_evolution': topic_evolution,
            'final_loss': best_loss,
            'perplexity': final_ppl,
            'train_time': train_time,
            'training_history': training_history,
            'time_info': time_info
        }
    
    def train_all(
        self,
        models: List[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all specified models
        
        Args:
            models: List of models to train, default ['lda', 'etm', 'ctm']
            **kwargs: Parameters passed to each model
            
        Returns:
            Results for all models
        """
        if models is None:
            models = ['lda', 'etm', 'ctm']
        
        results = {}
        
        for model_name in models:
            if model_name == 'lda':
                results['lda'] = self.train_lda(
                    max_iter=kwargs.get('lda_max_iter', 100)
                )
            elif model_name == 'etm':
                results['etm'] = self.train_etm(
                    epochs=kwargs.get('etm_epochs', 100),
                    batch_size=kwargs.get('batch_size', 64),
                    use_pretrained_embeddings=kwargs.get('use_word2vec', True)
                )
            elif model_name == 'ctm':
                results['ctm'] = self.train_ctm(
                    inference_type=kwargs.get('ctm_inference_type', 'zeroshot'),
                    epochs=kwargs.get('ctm_epochs', 100),
                    batch_size=kwargs.get('batch_size', 64)
                )
            elif model_name == 'dtm':
                results['dtm'] = self.train_dtm(
                    epochs=kwargs.get('dtm_epochs', 100),
                    batch_size=kwargs.get('batch_size', 64)
                )
            else:
                print(f"Unknown model: {model_name}, skipping...")
        
        return results


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='Train Baseline Topic Models')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--models', type=str, default='lda,etm,ctm', help='Models to train (comma-separated)')
    parser.add_argument('--num_topics', type=int, default=20, help='Number of topics')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs for neural models')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--sbert_model', type=str, default=None, help='SBERT model path (uses THETA_SBERT_PATH env var if not set)')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory (uses THETA_DATA_DIR env var if not set)')
    parser.add_argument('--result_dir', type=str, default=None, help='Result directory (uses THETA_RESULT_DIR env var if not set)')
    parser.add_argument('--job_id', type=str, default=None, help='Job ID for multi-user isolation')
    
    args = parser.parse_args()
    
    # Parse model list
    models = [m.strip() for m in args.models.split(',')]
    
    # Create trainer
    trainer = BaselineTrainer(
        dataset=args.dataset,
        num_topics=args.num_topics,
        vocab_size=args.vocab_size,
        data_dir=args.data_dir,
        result_dir=args.result_dir
    )
    
    # Prepare data
    generate_sbert = 'ctm' in models
    trainer.prepare_data(
        generate_sbert=generate_sbert,
        sbert_model=args.sbert_model
    )
    
    # Train models
    results = trainer.train_all(
        models=models,
        ctm_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  - Train time: {result['train_time']:.2f}s")
        if 'perplexity' in result:
            print(f"  - Perplexity: {result['perplexity']:.2f}")
        if 'final_loss' in result:
            print(f"  - Final loss: {result['final_loss']:.4f}")


if __name__ == '__main__':
    main()

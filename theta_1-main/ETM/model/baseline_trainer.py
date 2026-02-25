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
from torch.utils.data import TensorDataset, DataLoader
from data.dataloader import create_dataloader
from tqdm import tqdm

from .baseline_data import BaselineDataProcessor, prepare_baseline_data
from .baselines.lda import SklearnLDA
from .baselines.ctm import CTM, ZeroShotTM, CombinedTM
from .baselines.etm_original import OriginalETM, train_word2vec_embeddings
from .baselines.hdp import HDP
from .baselines.stm import STM, CovariatesRequiredError
from .baselines.btm import BTM
from .baselines.ntm import NVDM, GSM, ProdLDA


class BaselineTrainer:
    """
    Baseline model trainer
    
    Supports training from raw CSV files:
    - LDA: Uses sklearn, only requires BOW
    - CTM: Uses SBERT embedding + BOW
    - ETM: Uses Word2Vec embedding + BOW
    - HDP: Hierarchical Dirichlet Process (auto topic number)
    - STM: Structural Topic Model (with covariates)
    - BTM: Biterm Topic Model (for short texts)
    - NVDM/GSM/ProdLDA: Neural topic model variants
    """
    
    def __init__(
        self,
        dataset: str,
        num_topics: int = 20,
        vocab_size: int = 5000,
        data_dir: str = '/root/autodl-tmp/data',
        result_dir: str = '/root/autodl-tmp/result/baseline',
        data_exp_dir: str = None,
        output_dir: str = None,
        device: str = 'auto'
    ):
        """
        Initialize trainer
        
        Args:
            dataset: Dataset name
            num_topics: Number of topics
            vocab_size: Vocabulary size (only used if data_exp_dir is None)
            data_dir: Data directory
            result_dir: Result directory (only used if output_dir is None)
            data_exp_dir: Data experiment directory (new structure)
            output_dir: Model output directory (new structure)
            device: Device
        """
        self.dataset = dataset
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.data_exp_dir = data_exp_dir
        
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
        
        # Output directory - use provided output_dir or fallback to old structure
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(result_dir, dataset, f'vocab_{vocab_size}')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_preprocessed_data(self):
        """
        Load preprocessed data from data_exp_dir (new directory structure).
        This method loads BOW matrix, vocab, and optionally SBERT embeddings
        from a previously created data experiment directory.
        """
        if not self.data_exp_dir:
            raise ValueError("data_exp_dir not set. Use prepare_data() instead or provide data_exp_dir in __init__.")
        
        bow_path = os.path.join(self.data_exp_dir, 'bow_matrix.npy')
        vocab_path = os.path.join(self.data_exp_dir, 'vocab.json')
        
        if not os.path.exists(bow_path):
            raise FileNotFoundError(f"BOW matrix not found: {bow_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        
        print(f"Loading preprocessed data from: {self.data_exp_dir}")
        
        # Load BOW matrix
        self.bow_matrix = np.load(bow_path)
        
        # Load vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # Update vocab_size from loaded data
        self.vocab_size = len(self.vocab)
        
        # Load SBERT embeddings if exists
        sbert_path = os.path.join(self.data_exp_dir, 'sbert_embeddings.npy')
        if os.path.exists(sbert_path):
            self.sbert_embeddings = np.load(sbert_path)
            print(f"  SBERT embeddings: {self.sbert_embeddings.shape}")
        
        # Load Word2Vec embeddings if exists (for ETM)
        word2vec_path = os.path.join(self.data_exp_dir, 'word2vec_embeddings.npy')
        if os.path.exists(word2vec_path):
            self.word2vec_embeddings = np.load(word2vec_path)
            print(f"  Word2Vec embeddings: {self.word2vec_embeddings.shape}")
        else:
            self.word2vec_embeddings = None
        
        # Load config to get additional info
        config_path = os.path.join(self.data_exp_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                data_config = json.load(f)
            print(f"  Data config: vocab_size={data_config.get('vocab_size')}, created={data_config.get('created_at')}")
        
        print(f"  BOW matrix: {self.bow_matrix.shape}")
        print(f"  Vocab size: {len(self.vocab)}")
    
    def prepare_data(
        self,
        generate_sbert: bool = True,
        sbert_model: str = '/root/autodl-tmp/ETM/model/baselines/sbert/sentence-transformers/all-MiniLM-L6-v2'
    ):
        """
        Prepare data
        
        Args:
            generate_sbert: Whether to generate SBERT embedding
            sbert_model: SBERT model name
        """
        # Check if processed data already exists
        bow_path = os.path.join(self.output_dir, 'bow_matrix.npy')
        vocab_path = os.path.join(self.output_dir, 'vocab.json')
        
        if os.path.exists(bow_path) and os.path.exists(vocab_path):
            print("Loading existing processed data...")
            self.bow_matrix = np.load(bow_path)
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
        dataloader = create_dataloader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
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
        if use_pretrained_embeddings:
            # Try to load from data_exp_dir first
            if hasattr(self, 'word2vec_embeddings') and self.word2vec_embeddings is not None:
                word_embeddings = self.word2vec_embeddings
                print(f"Using preloaded Word2Vec embeddings: {word_embeddings.shape}")
            elif self.data_exp_dir:
                # Try to load from data experiment directory
                emb_path = os.path.join(self.data_exp_dir, 'word2vec_embeddings.npy')
                if os.path.exists(emb_path):
                    word_embeddings = np.load(emb_path)
                    print(f"Loaded Word2Vec embeddings from data_exp_dir: {word_embeddings.shape}")
            
            # If still no embeddings, try output_dir or train new ones
            if word_embeddings is None:
                emb_path = os.path.join(self.output_dir, f'word2vec_embeddings_{embedding_dim}.npy')
                if os.path.exists(emb_path):
                    print(f"Loading existing Word2Vec embeddings from {emb_path}")
                    word_embeddings = np.load(emb_path)
                elif self.texts is not None:
                    print("Training Word2Vec embeddings...")
                    word_embeddings = train_word2vec_embeddings(
                        texts=self.texts,
                        vocab=self.vocab,
                        embedding_dim=embedding_dim
                    )
                    np.save(emb_path, word_embeddings)
                    print(f"Word2Vec embeddings saved to {emb_path}")
                else:
                    print("⚠ No Word2Vec embeddings available, using random initialization")
        
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
        
        # Train with KL annealing to prevent posterior collapse
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        # KL annealing: gradually increase KL weight from 0 to 1
        kl_warmup_epochs = min(20, epochs // 3)
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            num_batches = 0
            
            # KL annealing weight
            kl_weight = min(1.0, epoch / kl_warmup_epochs) if kl_warmup_epochs > 0 else 1.0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for (bow,) in pbar:
                bow = bow.to(self.device)
                
                # Original ETM doesn't need doc_embeddings, pass dummy
                dummy_emb = torch.zeros(bow.size(0), 1).to(self.device)
                output = model(dummy_emb, bow)
                
                # Apply KL annealing
                loss = output['recon_loss'] + kl_weight * output['kl_loss']
                
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
        from .baselines.dtm import DTM
        
        print(f"\n{'='*60}")
        print(f"Training DTM: {self.dataset}")
        print(f"  - Topics: {self.num_topics}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"{'='*60}")
        
        # Load time slice information - try data_exp_dir first, then output_dir
        time_slices_path = None
        time_indices_path = None
        
        # 1. Try data_exp_dir first (new structure)
        if self.data_exp_dir:
            p1 = os.path.join(self.data_exp_dir, 'time_slices.json')
            p2 = os.path.join(self.data_exp_dir, 'time_indices.npy')
            if os.path.exists(p1) and os.path.exists(p2):
                time_slices_path = p1
                time_indices_path = p2
                print(f"  Loading time slices from data_exp_dir: {self.data_exp_dir}")
        
        # 2. Fallback to output_dir (old structure)
        if time_slices_path is None:
            p1 = os.path.join(self.output_dir, 'time_slices.json')
            p2 = os.path.join(self.output_dir, 'time_indices.npy')
            if os.path.exists(p1) and os.path.exists(p2):
                time_slices_path = p1
                time_indices_path = p2
        
        if time_slices_path is None or not os.path.exists(time_slices_path):
            raise ValueError(
                f"DTM requires time slice information, please run first:\n"
                f"  python prepare_data.py --dataset {self.dataset} --model dtm --exp_name dtm_vocab{self.vocab_size}"
            )
        
        with open(time_slices_path, 'r') as f:
            time_info = json.load(f)
        time_indices = np.load(time_indices_path)
        
        num_time_slices = time_info['num_time_slices']
        print(f"  - Time slices: {num_time_slices}")
        print(f"  - Time range: {time_info['unique_times'][0]} - {time_info['unique_times'][-1]}")
        
        # Prepare data - handle both sparse and dense matrices
        bow_dense = self.bow_matrix.toarray().astype(np.float32) if sp.issparse(self.bow_matrix) else self.bow_matrix.astype(np.float32)
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
        
        train_loader = create_dataloader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        val_loader = create_dataloader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
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
            from torch.utils.data import DataLoader as TorchDataLoader
            for batch in TorchDataLoader(dataset, batch_size=batch_size, shuffle=False):
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
        
        # Save results directly to output_dir (consistent with other models)
        model_dir = os.path.join(self.output_dir, 'dtm')
        os.makedirs(model_dir, exist_ok=True)
        
        # Compute final perplexity
        final_ppl = training_history[-1]['val_ppl'] if training_history else 0
        
        # Save theta, beta, topic_words
        np.save(os.path.join(model_dir, f'theta_k{self.num_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{self.num_topics}.npy'), beta)
        np.save(os.path.join(model_dir, f'beta_over_time_k{self.num_topics}.npy'), all_betas)
        
        with open(os.path.join(model_dir, f'topic_words_k{self.num_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_words, f, ensure_ascii=False, indent=2)
        
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_k{self.num_topics}.pt'))
        
        # Save training history
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
        with open(os.path.join(model_dir, f'training_history_k{self.num_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(training_history_data, f, ensure_ascii=False, indent=2)
        
        # Save topic evolution data
        topic_evolution = {}
        for t_idx, t_year in enumerate(time_info['unique_times']):
            topic_evolution[str(t_year)] = {}
            for k in range(self.num_topics):
                top_indices = all_betas[t_idx, k].argsort()[-10:][::-1]
                topic_evolution[str(t_year)][f'topic_{k}'] = [self.vocab[i] for i in top_indices]
        
        with open(os.path.join(model_dir, f'topic_evolution_k{self.num_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_evolution, f, ensure_ascii=False, indent=2)
        
        # Save info
        info = {
            'model': 'dtm',
            'num_topics': self.num_topics,
            'vocab_size': len(self.vocab),
            'train_time': train_time,
            'best_loss': best_loss,
            'final_ppl': final_ppl,
            'time_slices': num_time_slices,
            'time_range': [time_info['unique_times'][0], time_info['unique_times'][-1]],
        }
        with open(os.path.join(model_dir, f'info_k{self.num_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print(f"\nDTM Training Complete:")
        print(f"  - Train time: {train_time:.2f}s")
        print(f"  - Best loss: {best_loss:.4f}")
        print(f"  - Final PPL: {final_ppl:.2f}")
        print(f"  - Time slices: {num_time_slices}")
        print(f"  - Results saved to: {model_dir}")
        
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
    
    def train_hdp(
        self,
        max_topics: int = 150,
        alpha: float = 1.0,
        gamma: float = 1.0
    ) -> Dict[str, Any]:
        """Train HDP model"""
        print(f"\n{'='*60}")
        print(f"Training HDP (max_topics={max_topics})")
        print("="*60)
        
        start_time = time.time()
        
        model = HDP(
            vocab_size=self.vocab_size,
            max_topics=max_topics,
            alpha=alpha,
            gamma=gamma
        )
        
        # Handle both sparse and dense matrices
        bow_dense = self.bow_matrix.toarray() if sp.issparse(self.bow_matrix) else self.bow_matrix
        model.fit(bow_dense, vocab=self.vocab)
        
        train_time = time.time() - start_time
        
        # Get results
        theta = model.get_theta()
        beta = model.get_beta()
        actual_topics = model.actual_num_topics
        
        # Save results
        model_dir = os.path.join(self.output_dir, 'hdp', 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        np.save(os.path.join(model_dir, f'theta_k{actual_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{actual_topics}.npy'), beta)
        
        print(f"HDP training completed in {train_time:.2f}s")
        print(f"Actual topics inferred: {actual_topics}")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta,
            'actual_num_topics': actual_topics,
            'train_time': train_time
        }
    
    def train_stm(
        self,
        max_iter: int = 100,
        covariates: Optional[np.ndarray] = None,
        covariate_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train STM model. Requires covariates (document-level metadata).
        
        Raises:
            CovariatesRequiredError: If no covariates are provided.
        """
        print(f"\n{'='*60}")
        print(f"Training STM (num_topics={self.num_topics})")
        print("="*60)
        
        # STM requires covariates — check before attempting training
        can_run, reason = STM.check_requirements(
            dataset=self.dataset,
            covariates=covariates,
        )
        if not can_run:
            raise CovariatesRequiredError(dataset=self.dataset)
        
        start_time = time.time()
        
        model = STM(
            vocab_size=self.vocab_size,
            num_topics=self.num_topics,
            max_iter=max_iter
        )
        
        # Handle both sparse and dense matrices
        bow_dense = self.bow_matrix.toarray() if sp.issparse(self.bow_matrix) else self.bow_matrix
        model.fit(
            bow_dense,
            covariates=covariates,
            covariate_names=covariate_names,
            vocab=self.vocab,
            dataset=self.dataset
        )
        
        train_time = time.time() - start_time
        
        # Get results
        theta = model.get_theta()
        beta = model.get_beta()
        
        # Save results
        model_dir = os.path.join(self.output_dir, 'stm', 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        np.save(os.path.join(model_dir, f'theta_k{self.num_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{self.num_topics}.npy'), beta)
        
        print(f"STM training completed in {train_time:.2f}s")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta,
            'train_time': train_time,
            'covariate_effects': model.get_covariate_effects()
        }
    
    def train_btm(
        self,
        n_iter: int = 100,
        alpha: float = 1.0,
        beta: float = 0.01
    ) -> Dict[str, Any]:
        """Train BTM model"""
        print(f"\n{'='*60}")
        print(f"Training BTM (num_topics={self.num_topics})")
        print("="*60)
        
        start_time = time.time()
        
        model = BTM(
            vocab_size=self.vocab_size,
            num_topics=self.num_topics,
            alpha=alpha,
            beta=beta,
            n_iter=n_iter
        )
        
        # Handle both sparse and dense matrices
        bow_dense = self.bow_matrix.toarray() if sp.issparse(self.bow_matrix) else self.bow_matrix
        model.fit(bow_dense, vocab=self.vocab)
        
        train_time = time.time() - start_time
        
        # Get results
        theta = model.get_theta()
        beta_matrix = model.get_beta()
        
        # Save results
        model_dir = os.path.join(self.output_dir, 'btm', 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        np.save(os.path.join(model_dir, f'theta_k{self.num_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{self.num_topics}.npy'), beta_matrix)
        
        print(f"BTM training completed in {train_time:.2f}s")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta_matrix,
            'train_time': train_time
        }
    
    def _train_neural_topic_model(
        self,
        model_class,
        model_name: str,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.002,
        hidden_dim: int = 256
    ) -> Dict[str, Any]:
        """Generic training for neural topic models (NVDM, GSM, ProdLDA)"""
        print(f"\n{'='*60}")
        print(f"Training {model_name} (num_topics={self.num_topics})")
        print("="*60)
        
        start_time = time.time()
        
        # Create model
        model = model_class(
            vocab_size=self.vocab_size,
            num_topics=self.num_topics,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Prepare data - handle both sparse and dense matrices
        # Keep data on CPU for DataLoader, move to GPU in training loop
        bow_dense = self.bow_matrix.toarray() if sp.issparse(self.bow_matrix) else self.bow_matrix
        bow_tensor = torch.FloatTensor(bow_dense)
        dataset = TensorDataset(bow_tensor)
        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training
        training_history = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_recon = 0
            total_kl = 0
            
            for batch in dataloader:
                bow_batch = batch[0].to(self.device)
                
                optimizer.zero_grad()
                output = model(bow_batch)
                
                loss = output['recon_loss'].mean() + output['kl_loss'].mean()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * bow_batch.size(0)
                total_recon += output['recon_loss'].sum().item()
                total_kl += output['kl_loss'].sum().item()
            
            avg_loss = total_loss / len(bow_tensor)
            avg_recon = total_recon / len(bow_tensor)
            avg_kl = total_kl / len(bow_tensor)
            
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'recon_loss': avg_recon,
                'kl_loss': avg_kl
            })
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")
        
        train_time = time.time() - start_time
        
        # Get results - move data to device for inference
        model.eval()
        with torch.no_grad():
            theta = model.get_theta(bow_tensor.to(self.device)).cpu().numpy()
            beta = model.get_beta().cpu().numpy()
        
        # Save results
        model_dir = os.path.join(self.output_dir, model_name.lower(), 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        np.save(os.path.join(model_dir, f'theta_k{self.num_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{self.num_topics}.npy'), beta)
        
        print(f"{model_name} training completed in {train_time:.2f}s")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta,
            'final_loss': best_loss,
            'train_time': train_time,
            'training_history': training_history
        }
    
    def train_nvdm(self, epochs: int = 100, batch_size: int = 64, **kwargs) -> Dict[str, Any]:
        """Train NVDM model"""
        return self._train_neural_topic_model(NVDM, 'NVDM', epochs, batch_size, **kwargs)
    
    def train_gsm(self, epochs: int = 100, batch_size: int = 64, **kwargs) -> Dict[str, Any]:
        """Train GSM model"""
        return self._train_neural_topic_model(GSM, 'GSM', epochs, batch_size, **kwargs)
    
    def train_prodlda(self, epochs: int = 100, batch_size: int = 64, **kwargs) -> Dict[str, Any]:
        """Train ProdLDA model"""
        return self._train_neural_topic_model(ProdLDA, 'ProdLDA', epochs, batch_size, **kwargs)
    
    def train_bertopic(
        self,
        n_neighbors: int = 15,
        n_components: int = 5,
        min_cluster_size: int = 10,
        top_n_words: int = 10,
        language: str = "english"
    ) -> Dict[str, Any]:
        """
        Train BERTopic model
        
        Args:
            n_neighbors: UMAP n_neighbors
            n_components: UMAP output dimensions
            min_cluster_size: HDBSCAN minimum cluster size
            top_n_words: Number of top words per topic
            language: Language for stopwords
            
        Returns:
            Training results
        """
        from .baselines.bertopic_model import BERTopicModel
        
        print(f"\n{'='*60}")
        print(f"Training BERTopic")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Use local SBERT model
        local_sbert_path = '/root/autodl-tmp/ETM/model/baselines/sbert/sentence-transformers/all-MiniLM-L6-v2'
        if not os.path.exists(local_sbert_path):
            local_sbert_path = 'all-MiniLM-L6-v2'
        
        model = BERTopicModel(
            vocab_size=self.vocab_size,
            num_topics=self.num_topics if self.num_topics > 0 else None,
            embedding_model=local_sbert_path,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_cluster_size=min_cluster_size,
            top_n_words=top_n_words,
            language=language,
            calculate_probabilities=True,
            verbose=True
        )
        
        # BERTopic needs texts - reconstruct from BOW if no raw texts
        if self.texts is not None:
            texts = self.texts
        else:
            # Reconstruct texts from BOW
            print("  Reconstructing texts from BOW matrix...")
            bow_dense = self.bow_matrix.toarray() if sp.issparse(self.bow_matrix) else self.bow_matrix
            texts = []
            for doc in bow_dense:
                words = []
                for word_id, count in enumerate(doc):
                    if count > 0:
                        words.extend([self.vocab[word_id]] * int(min(count, 5)))
                texts.append(' '.join(words))
        
        # Use preloaded SBERT embeddings if available
        embeddings = None
        if hasattr(self, 'sbert_embeddings') and self.sbert_embeddings is not None:
            embeddings = self.sbert_embeddings
            print(f"  Using preloaded SBERT embeddings: {embeddings.shape}")
        
        # Fit model
        print(f"  Fitting BERTopic on {len(texts)} documents...")
        model.fit(texts, embeddings=embeddings)
        
        train_time = time.time() - start_time
        
        actual_topics = model.num_topics
        print(f"  BERTopic found {actual_topics} topics")
        print(f"  Outlier documents: {model.outlier_count}")
        
        # Get theta (document-topic distribution)
        theta = model.get_theta()
        if theta is not None and len(theta.shape) == 1:
            # If 1D (topic assignments), convert to probability matrix
            num_docs = len(texts)
            theta_matrix = np.zeros((num_docs, actual_topics))
            topics = model.get_topics()
            for i, t in enumerate(topics):
                if 0 <= t < actual_topics:
                    theta_matrix[i, t] = 1.0
            theta = theta_matrix
        
        # Get beta (topic-word distribution)
        beta = model.get_beta()
        
        # Get topic words
        topic_words = {}
        for k in range(actual_topics):
            words = model.get_topic_words(k, top_n=20)
            topic_words[f'topic_{k}'] = [w for w, _ in words]
        
        # Save results
        model_dir = os.path.join(self.output_dir, 'bertopic')
        os.makedirs(model_dir, exist_ok=True)
        
        np.save(os.path.join(model_dir, f'theta_k{actual_topics}.npy'), theta)
        np.save(os.path.join(model_dir, f'beta_k{actual_topics}.npy'), beta)
        
        with open(os.path.join(model_dir, f'topic_words_k{actual_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(topic_words, f, ensure_ascii=False, indent=2)
        
        # Save info
        info = {
            'model': 'bertopic',
            'actual_num_topics': actual_topics,
            'vocab_size': len(self.vocab),
            'train_time': train_time,
            'outlier_count': model.outlier_count,
            'n_neighbors': n_neighbors,
            'n_components': n_components,
            'min_cluster_size': min_cluster_size,
        }
        with open(os.path.join(model_dir, f'info_k{actual_topics}.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print(f"\nBERTopic Training Complete:")
        print(f"  - Train time: {train_time:.2f}s")
        print(f"  - Topics found: {actual_topics}")
        print(f"  - Results saved to: {model_dir}")
        
        # Print topic words example
        print(f"\nTop 10 words for first 3 topics:")
        for i in range(min(3, actual_topics)):
            words = topic_words[f'topic_{i}'][:10]
            print(f"  Topic {i}: {', '.join(words)}")
        
        return {
            'model': model,
            'theta': theta,
            'beta': beta,
            'actual_num_topics': actual_topics,
            'train_time': train_time,
            'topic_words': topic_words
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
            elif model_name == 'hdp':
                results['hdp'] = self.train_hdp(
                    max_topics=kwargs.get('hdp_max_topics', 150)
                )
            elif model_name == 'stm':
                results['stm'] = self.train_stm(
                    max_iter=kwargs.get('stm_max_iter', 100),
                    covariates=kwargs.get('covariates', None)
                )
            elif model_name == 'btm':
                results['btm'] = self.train_btm(
                    n_iter=kwargs.get('btm_n_iter', 100)
                )
            elif model_name == 'nvdm':
                results['nvdm'] = self.train_nvdm(
                    epochs=kwargs.get('nvdm_epochs', 100),
                    batch_size=kwargs.get('batch_size', 64)
                )
            elif model_name == 'gsm':
                results['gsm'] = self.train_gsm(
                    epochs=kwargs.get('gsm_epochs', 100),
                    batch_size=kwargs.get('batch_size', 64)
                )
            elif model_name == 'prodlda':
                results['prodlda'] = self.train_prodlda(
                    epochs=kwargs.get('prodlda_epochs', 100),
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
    parser.add_argument('--sbert_model', type=str, default='/root/autodl-tmp/ETM/model/baselines/sbert/sentence-transformers/all-MiniLM-L6-v2', help='SBERT model name')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/data', help='Data directory')
    parser.add_argument('--result_dir', type=str, default='/root/autodl-tmp/result/baseline', help='Result directory')
    
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

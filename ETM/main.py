#!/usr/bin/env python
"""
ETM Topic Model Pipeline - Main Entry Point

Usage:
    python main.py train --dataset socialTwitter --mode zero_shot --num_topics 20
    python main.py evaluate --dataset socialTwitter --mode zero_shot
    python main.py visualize --dataset socialTwitter --mode zero_shot
    python main.py pipeline --dataset socialTwitter --mode zero_shot --num_topics 20
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from scipy import sparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PipelineConfig, create_parser, config_from_args,
    DATA_DIR, EMBEDDING_DIR, ETM_DIR
)


def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(config.log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config.log_dir,
        f"{config.data.dataset}_{config.embedding.mode}_{timestamp}.log"
    )
    
    # Create logger
    logger = logging.getLogger("ETM")
    logger.setLevel(logging.DEBUG if config.dev_mode else logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Log file: {log_file}")
    return logger


def setup_device(config: PipelineConfig, local_rank: int = -1) -> torch.device:
    """Setup compute device with optional DDP support
    
    Args:
        config: Pipeline configuration
        local_rank: Local rank for DDP (-1 for single GPU)
    
    Returns:
        torch.device: The device to use
    """
    if config.device == "cuda" and torch.cuda.is_available():
        if local_rank >= 0:
            # DDP mode: use the assigned GPU
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
        else:
            # Single GPU mode
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def setup_ddp(local_rank: int, world_size: int):
    """Initialize Distributed Data Parallel
    
    Args:
        local_rank: Local rank of the current process
        world_size: Total number of processes
    """
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )


def cleanup_ddp():
    """Clean up DDP resources"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(local_rank: int = -1) -> bool:
    """Check if current process is the main process"""
    return local_rank <= 0


def load_texts(config: PipelineConfig, logger: logging.Logger) -> Tuple[List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load texts from dataset, optionally with timestamps"""
    import pandas as pd
    
    csv_path = config.data.raw_data_path
    logger.info(f"Loading texts from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Find text column - support various naming conventions across datasets
    text_col = None
    for col in ['cleaned_content', 'clean_text', 'cleaned_text', 'text', 'content', 'Text', 'Consumer complaint narrative']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")
    
    texts = df[text_col].fillna('').astype(str).tolist()
    
    # Find label column
    labels = None
    for col in ['label', 'Label', 'labels', 'category']:
        if col in df.columns:
            labels = df[col].values
            break
    
    # Load timestamps if enabled and column exists
    timestamps = None
    if config.visualization.enable_temporal and config.data.timestamp_column:
        ts_col = config.data.timestamp_column
        if ts_col in df.columns:
            try:
                timestamps = pd.to_datetime(df[ts_col], errors='coerce').values
                valid_count = pd.notna(timestamps).sum()
                logger.info(f"Loaded {valid_count}/{len(timestamps)} valid timestamps from column '{ts_col}'")
            except Exception as e:
                logger.warning(f"Failed to parse timestamps from column '{ts_col}': {e}")
        else:
            logger.warning(f"Timestamp column '{ts_col}' not found in data. Available: {df.columns.tolist()}")
    
    logger.info(f"Loaded {len(texts)} documents, text_col={text_col}, has_labels={labels is not None}")
    return texts, labels, timestamps


def generate_bow(
    texts: List[str],
    config: PipelineConfig,
    logger: logging.Logger
) -> Tuple[sparse.csr_matrix, List[str]]:
    """Generate BOW matrix"""
    from bow.vocab_builder import VocabBuilder, VocabConfig
    from bow.bow_generator import BOWGenerator
    
    logger.info(f"Generating BOW: vocab_size={config.bow.vocab_size}")
    
    # Build vocabulary
    vocab_config = VocabConfig(
        max_vocab_size=config.bow.vocab_size,
        min_df=config.bow.min_doc_freq,
        max_df_ratio=config.bow.max_doc_freq_ratio
    )
    vocab_builder = VocabBuilder(config=vocab_config, dev_mode=config.dev_mode)
    
    vocab_builder.add_documents(texts, dataset_name=config.data.dataset)
    vocab_builder.build_vocab()
    
    # Generate BOW
    bow_generator = BOWGenerator(vocab_builder, dev_mode=config.dev_mode)
    bow_output = bow_generator.generate_bow(texts, dataset_name=config.data.dataset)
    
    vocab = vocab_builder.get_vocab_list()
    
    logger.info(f"BOW shape: {bow_output.bow_matrix.shape}, vocab_size: {len(vocab)}")
    return bow_output.bow_matrix, vocab


def generate_vocab_embeddings(
    vocab: List[str],
    config: PipelineConfig,
    logger: logging.Logger
) -> np.ndarray:
    """Generate vocabulary embeddings using Qwen"""
    from model.vocab_embedder import VocabEmbedder
    
    logger.info(f"Generating vocab embeddings for {len(vocab)} words")
    
    embedder = VocabEmbedder(
        model_path=config.embedding.model_path,
        device="cuda" if config.device == "cuda" else "cpu",
        batch_size=64,
        dev_mode=config.dev_mode
    )
    
    embeddings = embedder.embed_vocab(vocab)
    logger.info(f"Vocab embeddings shape: {embeddings.shape}")
    
    return embeddings


def load_doc_embeddings(
    config: PipelineConfig,
    logger: logging.Logger
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load document embeddings from result/{dataset}/{mode}/embeddings/"""
    embedding_dir = config.embeddings_dir
    # Support multiple embedding file naming conventions
    emb_candidates = [
        os.path.join(embedding_dir, f"{config.data.dataset}_{config.embedding.mode}_embeddings.npy"),
        os.path.join(embedding_dir, "embeddings.npy"),
    ]
    emb_path = emb_candidates[0]  # default for error message
    for candidate in emb_candidates:
        if os.path.exists(candidate):
            emb_path = candidate
            break
    
    label_candidates = [
        os.path.join(embedding_dir, f"{config.data.dataset}_{config.embedding.mode}_labels.npy"),
        os.path.join(embedding_dir, "labels.npy"),
    ]
    label_path = label_candidates[0]
    for candidate in label_candidates:
        if os.path.exists(candidate):
            label_path = candidate
            break
    
    logger.info(f"Loading doc embeddings from {emb_path}")
    embeddings = np.load(emb_path)
    
    labels = None
    if os.path.exists(label_path):
        try:
            labels = np.load(label_path, allow_pickle=True)
        except Exception as e:
            logger.warning(f"Failed to load labels: {e}")
    
    logger.info(f"Doc embeddings: {embeddings.shape}, labels: {labels.shape if labels is not None else 'None'}")
    return embeddings, labels


def train_etm(
    doc_embeddings: np.ndarray,
    bow_matrix: sparse.csr_matrix,
    vocab_embeddings: np.ndarray,
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device,
    local_rank: int = -1,
    world_size: int = 1
) -> Dict[str, Any]:
    """Train ETM model with optional DDP support
    
    Args:
        doc_embeddings: Document embeddings
        bow_matrix: Bag-of-words matrix
        vocab_embeddings: Vocabulary embeddings
        config: Pipeline configuration
        logger: Logger instance
        device: Torch device
        local_rank: Local rank for DDP (-1 for single GPU)
        world_size: Total number of GPUs for DDP
    
    Returns:
        Dict containing model, history, and other results
    """
    from model.theta.etm import ETM
    from data.dataloader import ETMDataset, create_dataloader
    from torch.utils.data import random_split
    
    use_ddp = local_rank >= 0 and world_size > 1
    
    if is_main_process(local_rank):
        logger.info(f"Device: {device}")
        logger.info(f"DDP enabled: {use_ddp}, world_size: {world_size}")
        logger.info(f"Config: num_topics={config.model.num_topics}, epochs={config.model.epochs}, batch_size={config.model.batch_size}")
    
    # Create dataset
    # For large datasets, keep BOW in sparse format to avoid massive memory usage
    use_sparse = doc_embeddings.shape[0] > 200000
    if use_sparse and is_main_process(local_rank):
        logger.info(f"Large dataset ({doc_embeddings.shape[0]:,} docs): keeping BOW sparse to save memory")
    dataset = ETMDataset(
        doc_embeddings=doc_embeddings,
        bow_matrix=bow_matrix,
        normalize_bow=True,
        dev_mode=config.dev_mode,
        keep_sparse=use_sparse
    )
    
    # Split data
    n_total = len(dataset)
    n_train = int(n_total * config.model.train_ratio)
    n_val = int(n_total * config.model.val_ratio)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Scale DataLoader workers for large datasets
    # With keep_sparse=True, BOW is CSR (~few hundred MB) so workers are safe.
    # But limit workers to avoid excessive memory from forked doc_embeddings tensor.
    dl_num_workers = config.model.num_workers
    dl_persistent = config.model.persistent_workers
    dl_pin_memory = config.model.pin_memory
    if n_total > 200000:
        dl_num_workers = min(dl_num_workers, 2)
        dl_persistent = False
        if is_main_process(local_rank):
            logger.info(f"Large dataset ({n_total:,} docs): using num_workers={dl_num_workers}, persistent=False")
    
    # Create samplers for DDP
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True
        )
    
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=config.model.batch_size, 
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor,
        sampler=train_sampler
    )
    val_loader = create_dataloader(
        val_dataset, 
        batch_size=config.model.batch_size, 
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor
    )
    test_loader = create_dataloader(
        test_dataset, 
        batch_size=config.model.batch_size, 
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor
    )
    
    if is_main_process(local_rank):
        logger.info(f"Data splits: train={n_train}, val={n_val}, test={n_test}")
    
    # Create model
    vocab_size = bow_matrix.shape[1]
    
    # Use pretrained embeddings only if train_word_embeddings is False
    # Otherwise use random initialization for better learning
    if config.model.train_word_embeddings:
        word_emb_tensor = None  # Use random init
        logger.info("Using random word embeddings (trainable)")
    else:
        word_emb_tensor = torch.tensor(vocab_embeddings, dtype=torch.float32)
        logger.info("Using pretrained word embeddings (frozen)")
    
    model = ETM(
        vocab_size=vocab_size,
        num_topics=config.model.num_topics,
        doc_embedding_dim=config.model.doc_embedding_dim,
        word_embedding_dim=config.model.word_embedding_dim,
        hidden_dim=config.model.hidden_dim,
        encoder_dropout=config.model.encoder_dropout,
        word_embeddings=word_emb_tensor,
        train_word_embeddings=config.model.train_word_embeddings,
        dev_mode=config.dev_mode
    ).to(device)
    
    # Wrap model with DDP if using multi-GPU
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process(local_rank):
            logger.info(f"Model wrapped with DistributedDataParallel on {world_size} GPUs")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(local_rank):
        logger.info(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay
    )
    
    scheduler = None
    if config.model.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.model.scheduler_factor,
            patience=config.model.scheduler_patience
        )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'recon_loss': [], 'kl_loss': [],
        'perplexity': []  # Track perplexity during training
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    if is_main_process(local_rank):
        logger.info("=" * 60)
        logger.info("Starting training...")
    
    for epoch in range(config.model.epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # KL annealing: linear warmup from kl_start to kl_end
        # Free bits in loss function handles posterior collapse, so no minimum here
        warmup_progress = min(1.0, (epoch + 1) / config.model.kl_warmup_epochs)
        kl_weight = config.model.kl_start + (config.model.kl_end - config.model.kl_start) * warmup_progress
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            doc_emb = batch['doc_embedding'].to(device)
            bow = batch['bow'].to(device)
            
            optimizer.zero_grad()
            output = model(doc_emb, bow, kl_weight=kl_weight)
            loss = output['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * doc_emb.size(0)
        
        train_loss /= n_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                doc_emb = batch['doc_embedding'].to(device)
                bow = batch['bow'].to(device)
                
                output = model(doc_emb, bow, kl_weight=kl_weight)
                val_loss += output['total_loss'].item() * doc_emb.size(0)
                val_recon += output['recon_loss'].item() * doc_emb.size(0)
                val_kl += output['kl_loss'].item() * doc_emb.size(0)
        
        val_loss /= n_val
        val_recon /= n_val
        val_kl /= n_val
        
        # Compute perplexity on validation set
        # Perplexity = exp(negative log-likelihood per word)
        # We compute it properly using the model's reconstruction
        with torch.no_grad():
            total_nll = 0.0
            total_words = 0
            for batch in val_loader:
                doc_emb = batch['doc_embedding'].to(device)
                bow = batch['bow'].to(device)
                
                # Get topic distribution and word distribution
                # Handle DDP wrapped model
                base_model = model.module if use_ddp else model
                theta_batch, _, _ = base_model.encoder(doc_emb)
                beta = base_model.decoder.get_beta()
                
                # Compute word probabilities: p(w|d) = sum_k theta_dk * beta_kw
                word_probs = torch.matmul(theta_batch, beta)  # (batch, vocab)
                word_probs = torch.clamp(word_probs, min=1e-10)
                
                # Negative log-likelihood per document
                log_probs = torch.log(word_probs)
                nll = -torch.sum(bow * log_probs, dim=1)  # (batch,)
                doc_lengths = bow.sum(dim=1)  # (batch,)
                
                total_nll += nll.sum().item()
                total_words += doc_lengths.sum().item()
            
            # Perplexity = exp(NLL / total_words)
            val_perplexity = np.exp(total_nll / total_words) if total_words > 0 else float('inf')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(val_recon)
        history['kl_loss'].append(val_kl)
        history['perplexity'].append(val_perplexity)
        
        # Check for improvement
        improved = val_loss < best_val_loss - config.model.min_delta
        if improved:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""
        
        # Learning rate scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        if is_main_process(local_rank):
            logger.info(
                f"Epoch {epoch+1:3d}/{config.model.epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Recon: {val_recon:.4f} | KL: {val_kl:.4f} | "
                f"PPL: {val_perplexity:.1f} | KL_w: {kl_weight:.3f} {marker}"
            )
        
        # Periodic topic quality check (every 10 epochs or at end) - only on main process
        if is_main_process(local_rank) and ((epoch + 1) % 10 == 0 or epoch == config.model.epochs - 1 or improved):
            with torch.no_grad():
                base_model = model.module if use_ddp else model
                beta = base_model.decoder.get_beta().cpu().numpy()
                # Check topic diversity: how different are topics from each other
                top_k = 10
                topic_top_words = []
                for k in range(beta.shape[0]):
                    top_indices = beta[k].argsort()[-top_k:][::-1]
                    topic_top_words.append(set(top_indices))
                
                # Compute pairwise Jaccard similarity
                total_overlap = 0
                n_pairs = 0
                for i in range(len(topic_top_words)):
                    for j in range(i+1, len(topic_top_words)):
                        intersection = len(topic_top_words[i] & topic_top_words[j])
                        union = len(topic_top_words[i] | topic_top_words[j])
                        total_overlap += intersection / union if union > 0 else 0
                        n_pairs += 1
                avg_overlap = total_overlap / n_pairs if n_pairs > 0 else 0
                diversity = 1 - avg_overlap
                
                # Check beta sharpness (how peaked are topic distributions)
                beta_max = beta.max(axis=1).mean()
                beta_entropy = -np.sum(beta * np.log(beta + 1e-10), axis=1).mean()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Topic Quality: diversity={diversity:.3f}, beta_max={beta_max:.4f}, entropy={beta_entropy:.2f}")
        
        # Early stopping
        if config.model.early_stopping and patience_counter >= config.model.patience:
            if is_main_process(local_rank):
                logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            doc_emb = batch['doc_embedding'].to(device)
            bow = batch['bow'].to(device)
            output = model(doc_emb, bow, kl_weight=1.0)
            test_loss += output['total_loss'].item() * doc_emb.size(0)
    test_loss /= n_test
    
    history['test_loss'] = test_loss
    history['best_val_loss'] = best_val_loss
    history['epochs_trained'] = epoch + 1
    
    if is_main_process(local_rank):
        logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}, Test loss: {test_loss:.4f}")
    
    # Return the base model (unwrap DDP if needed)
    base_model = model.module if use_ddp else model
    
    return {
        'model': base_model,
        'history': history,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss
    }


def save_results(
    model,
    history: Dict,
    vocab: List[str],
    doc_embeddings: np.ndarray,
    bow_matrix: sparse.csr_matrix,
    vocab_embeddings: np.ndarray,
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device,
    timestamps: Optional[np.ndarray] = None
) -> str:
    """Save training results to result/{dataset}/{mode}/ subdirectories"""
    # Create all output directories
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.bow_dir, exist_ok=True)
    os.makedirs(config.evaluation_dir, exist_ok=True)
    os.makedirs(config.visualization_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Saving results to {config.result_dir}")
    
    model.eval()
    with torch.no_grad():
        # Get theta for all documents (batch processing for large datasets)
        n_docs = doc_embeddings.shape[0]
        if n_docs > 100000:
            # Process in batches to avoid GPU OOM
            batch_size = 10000
            theta_parts = []
            for start in range(0, n_docs, batch_size):
                end = min(start + batch_size, n_docs)
                batch_emb = torch.tensor(doc_embeddings[start:end], dtype=torch.float32).to(device)
                batch_theta = model.get_theta(batch_emb).cpu().numpy()
                theta_parts.append(batch_theta)
                del batch_emb
            theta = np.concatenate(theta_parts, axis=0)
            del theta_parts
        else:
            doc_emb_tensor = torch.tensor(doc_embeddings, dtype=torch.float32).to(device)
            theta = model.get_theta(doc_emb_tensor).cpu().numpy()
        
        # Get beta
        beta = model.get_beta().cpu().numpy()
        
        # Get topic embeddings
        topic_emb = model.get_topic_embeddings().cpu().numpy()
        
        # Get topic words
        topic_words = model.get_topic_words(top_k=20, vocab=vocab)
    
    # Save BOW to bow_dir
    # Save as dense npy format
    bow_dense = bow_matrix.toarray() if sparse.issparse(bow_matrix) else bow_matrix
    np.save(os.path.join(config.bow_dir, "bow_matrix.npy"), bow_dense)
    np.save(os.path.join(config.bow_dir, "vocab_embeddings.npy"), vocab_embeddings)
    with open(os.path.join(config.bow_dir, "vocab.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))
    logger.info(f"BOW saved to {config.bow_dir}")
    
    # Save model outputs to model_dir
    np.save(os.path.join(config.model_dir, f"theta_{timestamp}.npy"), theta)
    np.save(os.path.join(config.model_dir, f"beta_{timestamp}.npy"), beta)
    np.save(os.path.join(config.model_dir, f"topic_embeddings_{timestamp}.npy"), topic_emb)
    
    # Save topic words
    topic_words_dict = {str(k): words for k, words in topic_words}
    with open(os.path.join(config.model_dir, f"topic_words_{timestamp}.json"), 'w') as f:
        json.dump(topic_words_dict, f, indent=2, ensure_ascii=False)
    
    # Save training history
    with open(os.path.join(config.model_dir, f"training_history_{timestamp}.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(config.model_dir, f"etm_model_{timestamp}.pt"))
    
    # Save config
    config.save(os.path.join(config.model_dir, f"config_{timestamp}.json"))
    
    # Save timestamps if available (for temporal analysis)
    if timestamps is not None and len(timestamps) > 0:
        ts_path = os.path.join(config.result_dir, "timestamps.npy")
        np.save(ts_path, timestamps)
        logger.info(f"Timestamps saved to {ts_path}")
    
    # Log top words
    logger.info("=" * 60)
    logger.info("Top 10 words per topic:")
    for topic_idx, words in topic_words[:10]:
        word_str = ", ".join([w for w, _ in words[:10]])
        logger.info(f"  Topic {topic_idx}: {word_str}")
    
    logger.info(f"Results saved with timestamp: {timestamp}")
    return timestamp


def run_evaluation(
    config: PipelineConfig,
    logger: logging.Logger,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """Run evaluation on trained model"""
    from evaluation.topic_metrics import compute_all_metrics
    from evaluation.metrics import TopicMetrics
    
    # Find latest results if no timestamp
    if timestamp is None:
        result_files = sorted(Path(config.model_dir).glob("theta_*.npy"), reverse=True)
        if not result_files:
            raise FileNotFoundError(f"No results found in {config.model_dir}")
        timestamp = result_files[0].stem.replace("theta_", "")
    
    logger.info(f"Evaluating results with timestamp: {timestamp}")
    
    # Load results from model_dir
    theta = np.load(os.path.join(config.model_dir, f"theta_{timestamp}.npy"))
    beta = np.load(os.path.join(config.model_dir, f"beta_{timestamp}.npy"))
    
    with open(os.path.join(config.model_dir, f"topic_words_{timestamp}.json"), 'r') as f:
        topic_words = json.load(f)
    
    # Load BOW matrix from bow_dir (or regenerate if not exists)
    bow_path = os.path.join(config.bow_dir, "bow_matrix.npy")
    if os.path.exists(bow_path):
        bow_matrix = np.load(bow_path)
        logger.info(f"Loaded BOW from {bow_path}")
    else:
        texts, _, _ = load_texts(config, logger)
        bow_matrix, vocab = generate_bow(texts, config, logger)
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_all_metrics(
        beta=beta,
        theta=theta,
        doc_term_matrix=bow_matrix,
        top_k_coherence=config.evaluation.top_k_coherence,
        top_k_diversity=config.evaluation.top_k_diversity
    )
    
    # Log metrics
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    logger.info(f"  Topic Coherence (NPMI): {metrics.get('topic_coherence_npmi_avg', metrics.get('topic_coherence_avg', 0)):.4f}")
    logger.info(f"  Topic Diversity (TD): {metrics['topic_diversity_td']:.4f}")
    logger.info(f"  Topic Diversity (iRBO): {metrics['topic_diversity_irbo']:.4f}")
    if 'perplexity' in metrics and metrics['perplexity'] is not None:
        logger.info(f"  Perplexity: {metrics['perplexity']:.2f}")
    if 'topic_coherence_cv_avg' in metrics and metrics['topic_coherence_cv_avg'] is not None:
        logger.info(f"  Topic Coherence (C_V): {metrics['topic_coherence_cv_avg']:.4f}")
    if 'topic_coherence_umass_avg' in metrics and metrics['topic_coherence_umass_avg'] is not None:
        logger.info(f"  Topic Coherence (UMass): {metrics['topic_coherence_umass_avg']:.4f}")
    if 'topic_exclusivity_avg' in metrics and metrics['topic_exclusivity_avg'] is not None:
        logger.info(f"  Topic Exclusivity: {metrics['topic_exclusivity_avg']:.4f}")
    logger.info("=" * 60)
    
    # Save metrics to evaluation_dir
    os.makedirs(config.evaluation_dir, exist_ok=True)
    metrics_path = os.path.join(config.evaluation_dir, f"metrics_{timestamp}.json")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    return metrics


def run_visualization(
    config: PipelineConfig,
    logger: logging.Logger,
    timestamp: Optional[str] = None,
    use_wordcloud: bool = True
):
    """
    Generate visualizations using the unified visualization generator.
    Delegates entirely to run_all_visualizations() which handles:
    - VisualizationGenerator (global + per-topic charts)
    - TopicVisualizer (additional charts: wordclouds, pyLDAvis, etc.)
    - Summary report
    """
    from visualization.topic_visualizer import load_etm_results
    from visualization import run_all_visualizations
    
    # Find latest results if no timestamp
    if timestamp is None:
        result_files = sorted(Path(config.model_dir).glob("theta_*.npy"), reverse=True)
        if not result_files:
            raise FileNotFoundError(f"No results found in {config.model_dir}")
        timestamp = result_files[0].stem.replace("theta_", "")
    
    logger.info(f"Generating visualizations for timestamp: {timestamp}")
    
    # Save topic words to dedicated folder (before visualization)
    try:
        results = load_etm_results(config.model_dir, timestamp)
        topic_words_dir = os.path.join(config.result_dir, "topic_words")
        os.makedirs(topic_words_dir, exist_ok=True)
        
        topic_words_path = os.path.join(topic_words_dir, f"topic_words_{timestamp}.json")
        with open(topic_words_path, 'w', encoding='utf-8') as f:
            json.dump(results['topic_words'], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved topic words to {topic_words_path}")
        
        topic_words_txt_path = os.path.join(topic_words_dir, f"topic_words_{timestamp}.txt")
        with open(topic_words_txt_path, 'w', encoding='utf-8') as f:
            for topic_id, words in results['topic_words']:
                word_list = [f"{word}({prob:.4f})" for word, prob in words]
                f.write(f"Topic {topic_id}: {', '.join(word_list)}\n")
        logger.info(f"Saved topic words text to {topic_words_txt_path}")
    except Exception as e:
        logger.warning(f"Failed to save topic words: {e}")
    
    # Determine visualization parameters based on directory structure
    if config.train_exp:
        # New exp structure: result_dir is already the train exp dir (contains model/ subdir)
        viz_result_dir = config.result_dir
        viz_model_size = None
        viz_dataset = config.data.dataset
        viz_mode = config.embedding.mode
        # Output directly under the train exp's visualization dir
        viz_output_dir = str(Path(config.result_dir) / 'visualization')
    else:
        # Legacy structure
        viz_result_dir = config.output_base_dir
        viz_model_size = config.model_size
        viz_dataset = config.data.dataset
        viz_mode = config.embedding.mode
        viz_output_dir = None  # Let run_all_visualizations determine it
    
    try:
        output_dir = run_all_visualizations(
            result_dir=viz_result_dir,
            dataset=viz_dataset,
            mode=viz_mode,
            model_size=viz_model_size,
            output_dir=viz_output_dir,
            language=config.visualization.language,
            dpi=config.visualization.dpi
        )
        logger.info(f"All visualizations saved to {output_dir}")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


def run_train(config: PipelineConfig, logger: logging.Logger, local_rank: int = -1, world_size: int = 1):
    """Run training pipeline with optional DDP support
    
    Args:
        config: Pipeline configuration
        logger: Logger instance
        local_rank: Local rank for DDP (-1 for single GPU)
        world_size: Total number of GPUs for DDP
    """
    device = setup_device(config, local_rank)
    
    # Load data (with optional timestamps for temporal analysis)
    texts, _, timestamps = load_texts(config, logger)
    
    # Check if BOW already exists (shared across modes)
    bow_path = os.path.join(config.bow_dir, "bow_matrix.npy")
    vocab_path = os.path.join(config.bow_dir, "vocab.txt")
    vocab_emb_path = os.path.join(config.bow_dir, "vocab_embeddings.npy")
    
    if os.path.exists(bow_path) and os.path.exists(vocab_path) and os.path.exists(vocab_emb_path):
        # Load existing BOW (shared across modes for fair comparison)
        logger.info(f"Loading existing BOW from {config.bow_dir}")
        bow_matrix = np.load(bow_path)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        vocab_embeddings = np.load(vocab_emb_path)
        logger.info(f"Loaded BOW: shape={bow_matrix.shape}, vocab_size={len(vocab)}")
        # Convert large dense BOW to sparse to save memory during training
        if bow_matrix.shape[0] > 200000:
            logger.info(f"Converting BOW to sparse format to save memory ({bow_matrix.nbytes / 1e9:.1f} GB dense)")
            bow_matrix = sparse.csr_matrix(bow_matrix)
            logger.info(f"Sparse BOW: nnz={bow_matrix.nnz:,}, density={bow_matrix.nnz / (bow_matrix.shape[0]*bow_matrix.shape[1]):.4f}")
    else:
        # Generate BOW (first time for this dataset)
        logger.info(f"Generating new BOW for dataset {config.data.dataset}")
        bow_matrix, vocab = generate_bow(texts, config, logger)
        vocab_embeddings = generate_vocab_embeddings(vocab, config, logger)
        
        # Save BOW immediately so other modes can reuse it
        os.makedirs(config.bow_dir, exist_ok=True)
        sparse.save_npz(bow_path, bow_matrix)
        np.save(vocab_emb_path, vocab_embeddings)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab))
        logger.info(f"BOW saved to {config.bow_dir} (shared across modes)")
    
    # Load document embeddings
    doc_embeddings, labels = load_doc_embeddings(config, logger)
    
    # Auto-detect doc_embedding_dim from actual data
    actual_dim = doc_embeddings.shape[1]
    if config.model.doc_embedding_dim != actual_dim:
        logger.info(f"Auto-adjusting doc_embedding_dim: {config.model.doc_embedding_dim} -> {actual_dim}")
        config.model.doc_embedding_dim = actual_dim
    
    # Train model
    results = train_etm(doc_embeddings, bow_matrix, vocab_embeddings, config, logger, device, local_rank, world_size)
    
    # Save results (including timestamps if available)
    timestamp = save_results(
        results['model'], results['history'], vocab,
        doc_embeddings, bow_matrix, vocab_embeddings,
        config, logger, device,
        timestamps=timestamps
    )
    
    return timestamp


def run_pipeline(config: PipelineConfig, logger: logging.Logger,
                 skip_viz: bool = False, skip_eval: bool = False):
    """Run full pipeline: train + evaluate + visualize"""
    logger.info("=" * 60)
    logger.info("Running full ETM pipeline")
    logger.info(f"Dataset: {config.data.dataset}, Mode: {config.embedding.mode}")
    logger.info("=" * 60)
    
    # Train
    timestamp = run_train(config, logger)
    
    # Evaluate
    if not skip_eval:
        run_evaluation(config, logger, timestamp)
    else:
        logger.info("[SKIP] Evaluation skipped")
    
    # Visualize
    if not skip_viz:
        run_visualization(config, logger, timestamp)
    else:
        logger.info("[SKIP] Visualization skipped")
    
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Results: {config.result_dir}")
    logger.info(f"  - Embeddings: {config.embeddings_dir}")
    logger.info(f"  - BOW: {config.bow_dir}")
    logger.info(f"  - Model: {config.model_dir}")
    logger.info(f"  - Evaluation: {config.evaluation_dir}")
    logger.info(f"  - Visualization: {config.visualization_dir}")
    logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = create_parser()
    
    # Add DDP arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (set by torchrun)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of GPUs for distributed training')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup DDP if using multiple GPUs
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    
    if local_rank >= 0 and world_size > 1:
        setup_ddp(local_rank, world_size)
    
    # Create config from args
    config = config_from_args(args)
    
    # Setup logging (only on main process for DDP)
    logger = setup_logging(config)
    
    try:
        if args.command == "train":
            run_train(config, logger, local_rank, world_size)
        elif args.command == "evaluate":
            run_evaluation(config, logger, getattr(args, 'timestamp', None))
        elif args.command == "visualize":
            use_wordcloud = not getattr(args, 'no_wordcloud', False)
            run_visualization(config, logger, getattr(args, 'timestamp', None), use_wordcloud)
        elif args.command == "pipeline":
            skip_viz = getattr(args, 'skip_viz', False)
            skip_eval = getattr(args, 'skip_eval', False)
            run_pipeline(config, logger, skip_viz=skip_viz, skip_eval=skip_eval)
        elif args.command == "clean":
            # Data cleaning
            from dataclean.main import main as clean_main
            sys.argv = ['main.py', 'convert', args.input, args.output, '--language', args.language]
            clean_main()
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup DDP
        cleanup_ddp()


if __name__ == "__main__":
    main()

"""
Engine B: Embedding Training Module

Implements:
1. Unsupervised training - SimCSE contrastive learning
2. Supervised training - Label-guided contrastive learning with LoRA

Training stages (from method):
- Stage 1: Zero-shot (already implemented in embedder.py)
- Stage 2: Unsupervised fine-tuning (SimCSE)
- Stage 3: Supervised fine-tuning (with labels)
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from tqdm import tqdm

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARNING] peft not available, LoRA training disabled")


# Environment variable defaults for DLC compatibility
_THETA_BASE = os.environ.get("THETA_BASE", "/root/autodl-tmp")
_THETA_MODEL_DIR = os.environ.get("THETA_MODEL_DIR", f"{_THETA_BASE}/embedding_models")
_THETA_RESULT_DIR = os.environ.get("THETA_RESULT_DIR", f"{_THETA_BASE}/result")


@dataclass
class TrainingConfig:
    """Configuration for embedding training"""
    # Model
    model_path: str = f"{_THETA_MODEL_DIR}/qwen3_embedding_0.6B"
    max_length: int = 512
    
    # Training
    epochs: int = 3
    batch_size: int = 8  # Reduced from 16 to prevent OOM
    gradient_accumulation_steps: int = 2  # Effective batch size = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # LoRA (for supervised/unsupervised fine-tuning)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Contrastive learning
    temperature: float = 0.05
    
    # Output
    output_dir: str = f"{_THETA_BASE}/embedding/outputs"
    checkpoint_dir: str = f"{_THETA_BASE}/embedding/checkpoints"
    result_dir: str = _THETA_RESULT_DIR


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning"""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        tokenizer=None,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'idx': idx
        }
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx])
        
        return item


class SimCSELoss(nn.Module):
    """
    SimCSE contrastive loss for unsupervised learning.
    
    Uses dropout as data augmentation - same text with different dropout
    produces positive pairs.
    """
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SimCSE loss.
        
        Args:
            embeddings1: First view embeddings (batch, dim)
            embeddings2: Second view embeddings (batch, dim)
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Compute similarity matrix
        batch_size = embeddings1.size(0)
        
        # Concatenate embeddings
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # (2*batch, dim)
        
        # Compute all pairwise similarities
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=embeddings.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=embeddings.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss using labels.
    
    Pulls together samples with same label, pushes apart different labels.
    """
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: Embeddings (batch, dim)
            labels: Labels (batch,)
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        batch_size = embeddings.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.t()).float()
        
        # Remove self-similarities from positives
        mask_self = torch.eye(batch_size, device=embeddings.device)
        mask_positive = mask_positive - mask_self
        
        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)
        
        # Compute mean of log-likelihood over positive pairs
        num_positives = mask_positive.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)
        
        mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / num_positives
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


class EmbeddingTrainer:
    """
    Trainer for embedding fine-tuning.
    
    Supports:
    - Unsupervised: SimCSE contrastive learning
    - Supervised: Label-guided contrastive learning
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        dev_mode: bool = False
    ):
        self.config = config or TrainingConfig()
        self.dev_mode = dev_mode
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.multi_gpu = False  # Will be set in _load_model if using multiple GPUs
        
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.result_dir, exist_ok=True)
        
        # Load model and tokenizer
        self._load_model()
        
        # Timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.dev_mode:
            print(f"[DEV] EmbeddingTrainer initialized")
            print(f"[DEV]   device={self.device}")
            print(f"[DEV]   use_lora={self.config.use_lora}")
    
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        
        # Check if we should use model parallelism (split across GPUs)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and str(self.device) == "cuda":
            print(f"Using model parallelism with device_map='auto' across {num_gpus} GPUs")
            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                device_map="auto"
            )
            self.multi_gpu = True
        else:
            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
        
        # Apply LoRA if enabled
        if self.config.use_lora and PEFT_AVAILABLE:
            print("Applying LoRA adapter...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Move to device (skip if already using device_map="auto")
        if not self.multi_gpu:
            self.model.to(self.device)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        if self.config.use_lora and PEFT_AVAILABLE:
            # Load LoRA weights
            from peft import PeftModel
            # First load base model
            base_model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            # Then load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
            self.model.to(self.device)
        else:
            # Load full model state dict
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        print("Checkpoint loaded successfully")
    
    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get embeddings from model"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings
    
    def train_unsupervised(
        self,
        texts: List[str],
        dataset_name: str
    ) -> Dict:
        """
        Train with unsupervised SimCSE.
        
        Args:
            texts: List of texts
            dataset_name: Name of dataset
            
        Returns:
            Training results
        """
        print(f"\n{'='*70}")
        print(f"Unsupervised Training (SimCSE) - {dataset_name}")
        print(f"{'='*70}")
        print(f"Samples: {len(texts)}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        # Create dataset
        dataset = ContrastiveDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Loss function
        criterion = SimCSELoss(temperature=self.config.temperature)
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(dataloader) * self.config.epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop
        self.model.train()
        history = []
        accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_start = time.time()
            
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass twice with different dropout (SimCSE)
                embeddings1 = self._get_embeddings(input_ids, attention_mask)
                embeddings2 = self._get_embeddings(input_ids, attention_mask)
                
                # Compute loss with gradient accumulation
                loss = criterion(embeddings1, embeddings2) / accum_steps
                loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accum_steps
                progress.set_postfix({'loss': loss.item() * accum_steps})
                
                # Clear GPU cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(dataloader)
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'time': epoch_time,
                'lr': scheduler.get_last_lr()[0]
            })
        
        # Save checkpoint
        self._save_checkpoint(dataset_name, "unsupervised")
        
        # Save history
        self._save_history(history, dataset_name, "unsupervised")
        
        return {
            'history': history,
            'final_loss': history[-1]['loss']
        }
    
    def train_supervised(
        self,
        texts: List[str],
        labels: np.ndarray,
        dataset_name: str
    ) -> Dict:
        """
        Train with supervised contrastive learning.
        
        Args:
            texts: List of texts
            labels: Label array
            dataset_name: Name of dataset
            
        Returns:
            Training results
        """
        print(f"\n{'='*70}")
        print(f"Supervised Training (Contrastive) - {dataset_name}")
        print(f"{'='*70}")
        print(f"Samples: {len(texts)}")
        print(f"Unique labels: {len(np.unique(labels))}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        # Create dataset
        dataset = ContrastiveDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Loss function
        criterion = SupervisedContrastiveLoss(temperature=self.config.temperature)
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(dataloader) * self.config.epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop
        self.model.train()
        history = []
        accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_start = time.time()
            
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                
                # Forward pass
                embeddings = self._get_embeddings(input_ids, attention_mask)
                
                # Compute loss with gradient accumulation
                loss = criterion(embeddings, batch_labels) / accum_steps
                loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accum_steps
                progress.set_postfix({'loss': loss.item() * accum_steps})
                
                # Clear GPU cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(dataloader)
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'time': epoch_time,
                'lr': scheduler.get_last_lr()[0]
            })
        
        # Save checkpoint
        self._save_checkpoint(dataset_name, "supervised")
        
        # Save history
        self._save_history(history, dataset_name, "supervised")
        
        return {
            'history': history,
            'final_loss': history[-1]['loss']
        }
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings using the trained model.
        
        Args:
            texts: List of texts
            batch_size: Batch size
            show_progress: Show progress bar
            
        Returns:
            Embeddings array (N, D)
        """
        self.model.eval()
        
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        with torch.no_grad():
            for batch_idx, i in enumerate(iterator):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=self.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get embeddings
                embeddings = self._get_embeddings(input_ids, attention_mask)
                
                # Normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                # Clear GPU cache periodically
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
        
        return np.vstack(all_embeddings)
    
    def _save_checkpoint(self, dataset_name: str, mode: str):
        """Save model checkpoint to dataset-specific directory"""
        # Create dataset-specific checkpoint directory
        dataset_ckpt_dir = os.path.join(self.config.checkpoint_dir, dataset_name)
        os.makedirs(dataset_ckpt_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            dataset_ckpt_dir,
            f"embedding_{mode}_{self.run_timestamp}"
        )
        
        if self.config.use_lora and PEFT_AVAILABLE:
            # Save only LoRA weights
            self.model.save_pretrained(checkpoint_path)
        else:
            # Save full model
            torch.save(self.model.state_dict(), f"{checkpoint_path}.pt")
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _save_history(self, history: List[Dict], dataset_name: str, mode: str):
        """Save training history to dataset-specific directory"""
        # Create dataset-specific result directory
        dataset_result_dir = os.path.join(self.config.result_dir, dataset_name, "embedding")
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        history_path = os.path.join(
            dataset_result_dir,
            f"{mode}_training_history_{self.run_timestamp}.json"
        )
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved history: {history_path}")
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        dataset_name: str,
        mode: str,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """
        Save embeddings to files.
        
        Args:
            embeddings: Embedding matrix
            dataset_name: Dataset name
            mode: Training mode
            labels: Optional labels
            
        Returns:
            Dictionary with file paths
        """
        # Create mode-specific directory under dataset folder
        mode_dir = os.path.join(self.config.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        
        base_name = f"{dataset_name}_{mode}"
        
        # Save embeddings to output dir
        emb_path = os.path.join(mode_dir, f"{base_name}_embeddings.npy")
        np.save(emb_path, embeddings)
        
        # Save labels if provided
        label_path = None
        if labels is not None:
            label_path = os.path.join(mode_dir, f"{base_name}_labels.npy")
            np.save(label_path, labels)
        
        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'mode': mode,
            'num_docs': embeddings.shape[0],
            'embedding_dim': embeddings.shape[1],
            'timestamp': self.run_timestamp,
            'has_labels': labels is not None
        }
        
        meta_path = os.path.join(mode_dir, f"{base_name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save to result directory with dataset-specific folder
        dataset_result_dir = os.path.join(self.config.result_dir, dataset_name, "embedding")
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        result_emb_path = os.path.join(
            dataset_result_dir,
            f"{mode}_embeddings_{self.run_timestamp}.npy"
        )
        np.save(result_emb_path, embeddings)
        
        # Save labels to result dir too
        if labels is not None:
            result_label_path = os.path.join(
                dataset_result_dir,
                f"{mode}_labels_{self.run_timestamp}.npy"
            )
            np.save(result_label_path, labels)
        
        # Save metadata to result dir
        result_meta_path = os.path.join(
            dataset_result_dir,
            f"{mode}_metadata_{self.run_timestamp}.json"
        )
        with open(result_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        paths = {
            'embeddings': emb_path,
            'labels': label_path,
            'metadata': meta_path,
            'result_embeddings': result_emb_path
        }
        
        print(f"Saved embeddings: {emb_path}")
        print(f"Saved to result: {dataset_result_dir}")
        
        return paths

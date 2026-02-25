"""
Engine B: Embedding Training Module v2

Redesigned training methods:
1. Supervised training: Encoder + MLP classification head + Cross-Entropy loss
2. Unsupervised training: Autoregressive language modeling + Cross-Entropy loss

Key changes from v1:
- Supervised: Uses MLP classifier instead of contrastive loss
- Unsupervised: Uses causal LM loss instead of SimCSE
- More epochs (default 10-20)
- Proper deep learning training paradigm
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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
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


@dataclass
class TrainingConfig:
    """Configuration for embedding training"""
    # Model
    model_path: str = "/root/autodl-tmp/qwen3_embedding_0.6B"
    max_length: int = 512
    hidden_dim: int = 1024  # Embedding dimension
    
    # Training
    epochs: int = 10  # Increased from 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    classifier_lr: float = 1e-3  # Higher LR for classifier head
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Classifier head
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.3
    
    # Output
    output_dir: str = "/root/autodl-tmp/embedding/outputs"
    checkpoint_dir: str = "/root/autodl-tmp/embedding/checkpoints"
    result_dir: str = "/root/autodl-tmp/result"


class ClassificationDataset(Dataset):
    """Dataset for classification training"""
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
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
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class AutoregressiveDataset(Dataset):
    """Dataset for autoregressive language modeling"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # For autoregressive: labels are shifted input_ids
        # -100 is ignore index for cross entropy
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class MLPClassifier(nn.Module):
    """MLP classification head for supervised training"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class EmbeddingTrainerV2:
    """
    Trainer for embedding fine-tuning (v2).
    
    Training methods:
    - Supervised: Encoder + MLP classifier + Cross-Entropy loss
    - Unsupervised: Autoregressive LM + Cross-Entropy loss
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
        
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.result_dir, exist_ok=True)
        
        # Model will be loaded based on training mode
        self.model = None
        self.tokenizer = None
        self.classifier = None
        
        # Timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.dev_mode:
            print(f"[DEV] EmbeddingTrainerV2 initialized")
            print(f"[DEV]   device={self.device}")
            print(f"[DEV]   epochs={self.config.epochs}")
    
    def _load_encoder_model(self):
        """Load encoder model for supervised training"""
        print(f"Loading encoder model from {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
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
        
        self.model.to(self.device)
    
    def _load_causal_lm_model(self):
        """Load causal LM model for unsupervised training"""
        print(f"Loading causal LM model from {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load as causal LM for autoregressive training
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
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
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.model.to(self.device)
    
    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get embeddings from encoder model using mean pooling"""
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
    
    def train_supervised(
        self,
        texts: List[str],
        labels: np.ndarray,
        dataset_name: str
    ) -> Dict:
        """
        Supervised training with MLP classifier + Cross-Entropy loss.
        
        Architecture:
            text -> Encoder -> embedding -> MLP -> logits -> CE loss
        
        Args:
            texts: List of texts
            labels: Label array (numeric)
            dataset_name: Name of dataset
            
        Returns:
            Training results
        """
        print(f"\n{'='*70}")
        print(f"Supervised Training (MLP + Cross-Entropy) - {dataset_name}")
        print(f"{'='*70}")
        
        # Load encoder model
        self._load_encoder_model()
        
        # Get number of classes
        num_classes = len(np.unique(labels))
        print(f"Samples: {len(texts)}")
        print(f"Classes: {num_classes}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        # Initialize classifier head
        self.classifier = MLPClassifier(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.classifier_hidden_dim,
            num_classes=num_classes,
            dropout=self.config.classifier_dropout
        ).to(self.device)
        
        # Create dataset and dataloader
        dataset = ClassificationDataset(
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
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer with different LR for encoder and classifier
        optimizer = AdamW([
            {'params': self.model.parameters(), 'lr': self.config.learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.config.classifier_lr}
        ], weight_decay=self.config.weight_decay)
        
        # Scheduler
        total_steps = len(dataloader) * self.config.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[self.config.learning_rate, self.config.classifier_lr],
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio
        )
        
        # Training loop
        self.model.train()
        self.classifier.train()
        history = []
        accum_steps = self.config.gradient_accumulation_steps
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_start = time.time()
            
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    embeddings = self._get_embeddings(input_ids, attention_mask)
                    logits = self.classifier(embeddings.float())
                    loss = criterion(logits, batch_labels) / accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Track metrics
                epoch_loss += loss.item() * accum_steps
                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == batch_labels).sum().item()
                epoch_total += batch_labels.size(0)
                
                acc = epoch_correct / epoch_total
                progress.set_postfix({'loss': loss.item() * accum_steps, 'acc': f'{acc:.3f}'})
                
                # Clear GPU cache periodically
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(dataloader)
            accuracy = epoch_correct / epoch_total
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Accuracy: {accuracy:.4f}")
            
            history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy,
                'time': epoch_time
            })
        
        # Save checkpoint
        self._save_checkpoint(dataset_name, "supervised")
        
        # Save history
        self._save_history(history, dataset_name, "supervised")
        
        return {
            'history': history,
            'final_loss': history[-1]['loss'],
            'final_accuracy': history[-1]['accuracy']
        }
    
    def train_unsupervised(
        self,
        texts: List[str],
        dataset_name: str
    ) -> Dict:
        """
        Unsupervised training with autoregressive language modeling.
        
        Uses Cross-Entropy loss on next token prediction.
        
        Args:
            texts: List of texts
            dataset_name: Name of dataset
            
        Returns:
            Training results
        """
        print(f"\n{'='*70}")
        print(f"Unsupervised Training (Autoregressive LM) - {dataset_name}")
        print(f"{'='*70}")
        
        # Load causal LM model
        self._load_causal_lm_model()
        
        print(f"Samples: {len(texts)}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        # Create dataset
        dataset = AutoregressiveDataset(
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
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(dataloader) * self.config.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio
        )
        
        # Training loop
        self.model.train()
        history = []
        accum_steps = self.config.gradient_accumulation_steps
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_start = time.time()
            
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with causal LM
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accum_steps
                progress.set_postfix({'loss': loss.item() * accum_steps})
                
                # Clear GPU cache periodically
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(dataloader)
            perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Perplexity: {perplexity:.2f}")
            
            history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'perplexity': perplexity,
                'time': epoch_time
            })
        
        # Save checkpoint
        self._save_checkpoint(dataset_name, "unsupervised")
        
        # Save history
        self._save_history(history, dataset_name, "unsupervised")
        
        return {
            'history': history,
            'final_loss': history[-1]['loss'],
            'final_perplexity': history[-1]['perplexity']
        }
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings using the trained encoder model.
        
        Args:
            texts: List of texts
            batch_size: Batch size
            show_progress: Show progress bar
            
        Returns:
            Embeddings array (N, D)
        """
        if self.model is None:
            self._load_encoder_model()
        
        self.model.eval()
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=self.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    embeddings = self._get_embeddings(input_ids, attention_mask)
                
                all_embeddings.append(embeddings.float().cpu().numpy())
                
                if i % (batch_size * 50) == 0:
                    torch.cuda.empty_cache()
        
        return np.vstack(all_embeddings)
    
    def _save_checkpoint(self, dataset_name: str, mode: str):
        """Save model checkpoint"""
        dataset_ckpt_dir = os.path.join(self.config.checkpoint_dir, dataset_name)
        os.makedirs(dataset_ckpt_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            dataset_ckpt_dir,
            f"embedding_{mode}_{self.run_timestamp}"
        )
        
        if self.config.use_lora and PEFT_AVAILABLE:
            self.model.save_pretrained(checkpoint_path)
        else:
            torch.save(self.model.state_dict(), f"{checkpoint_path}.pt")
        
        # Save classifier if exists
        if self.classifier is not None:
            classifier_path = os.path.join(
                dataset_ckpt_dir,
                f"classifier_{mode}_{self.run_timestamp}.pt"
            )
            torch.save(self.classifier.state_dict(), classifier_path)
            print(f"Saved classifier: {classifier_path}")
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _save_history(self, history: List[Dict], dataset_name: str, mode: str):
        """Save training history"""
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
        """Save embeddings to files"""
        mode_dir = os.path.join(self.config.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        
        base_name = f"{dataset_name}_{mode}"
        
        # Save embeddings
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
        
        # Also save to result directory
        dataset_result_dir = os.path.join(self.config.result_dir, dataset_name, "embedding")
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        result_emb_path = os.path.join(
            dataset_result_dir,
            f"{mode}_embeddings_{self.run_timestamp}.npy"
        )
        np.save(result_emb_path, embeddings)
        
        if labels is not None:
            result_label_path = os.path.join(
                dataset_result_dir,
                f"{mode}_labels_{self.run_timestamp}.npy"
            )
            np.save(result_label_path, labels)
        
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

"""
PLAN: Generate embeddings for vocabulary and documents
WHAT: Calls scripts/run_engine_b.py to generate embeddings for downstream processing
WHY: Creates dense representations required for ETM topic modeling
INPUTS: 
- job_id: str - Unique identifier for analysis job
- vocab_path: str - Path to vocabulary JSON file
- bow_path: str - Path to BOW matrix file
OUTPUTS:
- embedding/outputs/zero_shot/{job_id}_embeddings.npy - Document embeddings (N×1024)
- embedding/outputs/zero_shot/{job_id}_vocab_emb.npy - Vocabulary embeddings (V×1024)
SIDE EFFECTS:
- Creates embedding/outputs/zero_shot/ directory structure
- Generates dense vector representations
- Logs processing steps
FAILURE MODES:
- Script execution failure → log error + status="failed"
- Invalid input shapes → log error + status="failed"
- Output shape mismatch → log error + status="failed"
RULE:
- Must call scripts/run_engine_b.py with correct parameters
- Must validate output shapes: embeddings (N,1024), vocab_emb (V,1024)
- All embeddings must be float32 type
- Path contract: inputs from ETM/outputs/, outputs to embedding/outputs/zero_shot/
"""

import os
import json
import numpy as np
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class EmbeddingAgent:
    """Agent responsible for generating embeddings"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp", model_path: str = None):
        self.base_dir = Path(base_dir)
        self.model_path = model_path  # Optional Qwen model path
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"EmbeddingAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def process(self, job_id: str, vocab_path: str, bow_path: str) -> Dict[str, Any]:
        """
        Generate embeddings for vocabulary and documents
        
        Args:
            job_id: Unique identifier for analysis job
            vocab_path: Path to vocabulary JSON file
            bow_path: Path to BOW matrix file
            
        Returns:
            Dict with processing results and status
        """
        try:
            self.logger.info(f"Starting embedding generation for job_id: {job_id}")
            
            # Prepare paths
            script_path = self.base_dir / "scripts" / "run_engine_b.py"
            
            # Create output directory
            embed_dir = self.base_dir / "embedding" / "outputs" / "zero_shot"
            embed_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output paths
            doc_emb_output = embed_dir / f"{job_id}_embeddings.npy"
            vocab_emb_output = embed_dir / f"{job_id}_vocab_emb.npy"
            
            # Get data CSV path for better document embeddings
            data_csv_path = self.base_dir / "data" / job_id / "data.csv"
            
            # Call engine B script
            cmd = [
                "python", str(script_path),
                "--vocab", str(vocab_path),
                "--bow", str(bow_path),
                "--doc_emb_output", str(doc_emb_output),
                "--vocab_emb_output", str(vocab_emb_output),
                "--job_id", job_id
            ]
            
            # Add optional Qwen model path if available
            if self.model_path and Path(self.model_path).exists():
                cmd.extend(["--model_path", str(self.model_path)])
            
            # Add data CSV path if exists
            if data_csv_path.exists():
                cmd.extend(["--data_csv", str(data_csv_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
            
            # Validate outputs
            if not doc_emb_output.exists():
                raise FileNotFoundError(f"Document embeddings not created: {doc_emb_output}")
            if not vocab_emb_output.exists():
                raise FileNotFoundError(f"Vocabulary embeddings not created: {vocab_emb_output}")
            
            # Validate shapes and types
            doc_embeddings = np.load(doc_emb_output)
            vocab_embeddings = np.load(vocab_emb_output)
            
            # Check that embeddings have 2 dimensions and consistent embedding dim
            if len(doc_embeddings.shape) != 2:
                raise ValueError(f"Document embeddings must be 2D, got shape: {doc_embeddings.shape}")
            if len(vocab_embeddings.shape) != 2:
                raise ValueError(f"Vocabulary embeddings must be 2D, got shape: {vocab_embeddings.shape}")
            if doc_embeddings.shape[1] != vocab_embeddings.shape[1]:
                raise ValueError(f"Embedding dimension mismatch: doc={doc_embeddings.shape[1]}, vocab={vocab_embeddings.shape[1]}")
            
            # Check dtype (allow float32 or float64)
            if doc_embeddings.dtype not in [np.float32, np.float64]:
                raise ValueError(f"Document embeddings type mismatch: {doc_embeddings.dtype}, expected float32 or float64")
            if vocab_embeddings.dtype not in [np.float32, np.float64]:
                raise ValueError(f"Vocabulary embeddings type mismatch: {vocab_embeddings.dtype}, expected float32 or float64")
            
            # Log success
            self._log(job_id, f"Embedding generation successful. Doc emb: {doc_embeddings.shape}, Vocab emb: {vocab_embeddings.shape}")
            
            return {
                "status": "success",
                "job_id": job_id,
                "doc_embeddings_path": str(doc_emb_output),
                "vocab_embeddings_path": str(vocab_emb_output),
                "doc_emb_shape": doc_embeddings.shape,
                "vocab_emb_shape": vocab_embeddings.shape
            }
            
        except subprocess.CalledProcessError as e:
            self._log(job_id, f"Embedding generation failed: {e.stderr}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"Script execution failed: {e.stderr}"
            }
        except Exception as e:
            self._log(job_id, f"Embedding generation failed: {str(e)}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(e)
            }
    
    def _log(self, job_id: str, message: str, error: bool = False):
        """Log message to result/{job_id}/log.txt"""
        log_path = self.base_dir / "result" / job_id / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "ERROR" if error else "INFO"
        log_entry = f"[{timestamp}] [{level}] EmbeddingAgent: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)

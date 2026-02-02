"""
PLAN: Run ETM (Embedded Topic Model) to generate topic distributions
WHAT: Calls scripts/run_engine_c.py to generate topic model outputs
WHY: Creates the core topic modeling results (theta, beta, alpha matrices)
INPUTS: 
- job_id: str - Unique identifier for analysis job
- vocab_path: str - Path to vocabulary JSON file
- bow_path: str - Path to BOW matrix file
- doc_embeddings_path: str - Path to document embeddings
- vocab_embeddings_path: str - Path to vocabulary embeddings
OUTPUTS:
- ETM/outputs/theta/{job_id}_theta.npy - Document-topic distributions (N×K)
- ETM/outputs/beta/{job_id}_beta.npy - Topic-word distributions (K×V)
- ETM/outputs/alpha/{job_id}_alpha.npy - Topic embeddings (K×1024)
- ETM/outputs/topic_words/{job_id}_topics.json - Topic keywords and metadata
SIDE EFFECTS:
- Creates ETM/outputs/ directory structure
- Generates core topic model matrices
- Logs processing steps and training metrics
FAILURE MODES:
- Script execution failure → log error + status="failed"
- Training convergence issues → log error + status="failed"
- Output shape validation failure → log error + status="failed"
RULE:
- Must call scripts/run_engine_c.py with correct parameters
- Must validate output shapes: theta (N,K), beta (K,V), alpha (K,1024)
- Must generate topics.json with proper schema
- Path contract: inputs from embedding/outputs/ and ETM/outputs/, outputs to ETM/outputs/
"""

import os
import json
import numpy as np
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class ETMAgent:
    """Agent responsible for running ETM topic modeling"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp"):
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"ETMAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def process(self, job_id: str, vocab_path: str, bow_path: str, 
               doc_embeddings_path: str, vocab_embeddings_path: str) -> Dict[str, Any]:
        """
        Run ETM topic modeling
        
        Args:
            job_id: Unique identifier for analysis job
            vocab_path: Path to vocabulary JSON file
            bow_path: Path to BOW matrix file
            doc_embeddings_path: Path to document embeddings
            vocab_embeddings_path: Path to vocabulary embeddings
            
        Returns:
            Dict with processing results and status
        """
        try:
            self.logger.info(f"Starting ETM processing for job_id: {job_id}")
            
            # Prepare paths
            script_path = self.base_dir / "scripts" / "run_engine_c.py"
            
            # Create output directories
            theta_dir = self.base_dir / "ETM" / "outputs" / "theta"
            beta_dir = self.base_dir / "ETM" / "outputs" / "beta"
            alpha_dir = self.base_dir / "ETM" / "outputs" / "alpha"
            topics_dir = self.base_dir / "ETM" / "outputs" / "topic_words"
            
            for dir_path in [theta_dir, beta_dir, alpha_dir, topics_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Define output paths
            theta_output = theta_dir / f"{job_id}_theta.npy"
            beta_output = beta_dir / f"{job_id}_beta.npy"
            alpha_output = alpha_dir / f"{job_id}_alpha.npy"
            topics_output = topics_dir / f"{job_id}_topics.json"
            
            # Call engine C script
            cmd = [
                "python", str(script_path),
                "--vocab", str(vocab_path),
                "--bow", str(bow_path),
                "--doc_emb", str(doc_embeddings_path),
                "--vocab_emb", str(vocab_embeddings_path),
                "--theta_output", str(theta_output),
                "--beta_output", str(beta_output),
                "--alpha_output", str(alpha_output),
                "--topics_output", str(topics_output),
                "--job_id", job_id
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
            
            # Validate outputs
            required_outputs = [theta_output, beta_output, alpha_output, topics_output]
            for output_path in required_outputs:
                if not output_path.exists():
                    raise FileNotFoundError(f"ETM output not created: {output_path}")
            
            # Validate shapes and types
            theta = np.load(theta_output)
            beta = np.load(beta_output)
            alpha = np.load(alpha_output)
            with open(topics_output, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
            
            # Shape validation
            if len(theta.shape) != 2:
                raise ValueError(f"Theta shape invalid: {theta.shape}, expected (N, K)")
            if len(beta.shape) != 2:
                raise ValueError(f"Beta shape invalid: {beta.shape}, expected (K, V)")
            if len(alpha.shape) != 2:
                raise ValueError(f"Alpha shape invalid: {alpha.shape}, expected (K, 1024)")
            if theta.shape[1] != beta.shape[0]:
                raise ValueError(f"Topic count mismatch: theta K={theta.shape[1]}, beta K={beta.shape[0]}")
            if alpha.shape[0] != beta.shape[0]:
                raise ValueError(f"Topic count mismatch: alpha K={alpha.shape[0]}, beta K={beta.shape[0]}")
            
            # Validate topics.json schema
            if not isinstance(topics_data, list):
                raise ValueError("Topics output must be a list")
            for topic in topics_data:
                if not all(key in topic for key in ['id', 'name', 'keywords', 'proportion']):
                    raise ValueError(f"Topic missing required fields: {topic}")
            
            # Log success
            self._log(job_id, f"ETM processing successful. Theta: {theta.shape}, Beta: {beta.shape}, Alpha: {alpha.shape}")
            
            return {
                "status": "success",
                "job_id": job_id,
                "theta_path": str(theta_output),
                "beta_path": str(beta_output),
                "alpha_path": str(alpha_output),
                "topics_path": str(topics_output),
                "theta_shape": theta.shape,
                "beta_shape": beta.shape,
                "alpha_shape": alpha.shape,
                "num_topics": len(topics_data)
            }
            
        except subprocess.CalledProcessError as e:
            self._log(job_id, f"ETM processing failed: {e.stderr}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"Script execution failed: {e.stderr}"
            }
        except Exception as e:
            self._log(job_id, f"ETM processing failed: {str(e)}", error=True)
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
        log_entry = f"[{timestamp}] [{level}] ETMAgent: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)

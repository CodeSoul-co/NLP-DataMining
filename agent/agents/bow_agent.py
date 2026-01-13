"""
PLAN: Generate BOW representation and vocabulary for topic modeling
WHAT: Calls scripts/run_engine_a.py to generate vocabulary and BOW matrix
WHY: Creates input representations required for embedding and ETM processing
INPUTS: 
- job_id: str - Unique identifier for analysis job
- data_path: str - Path to validated CSV file
OUTPUTS:
- ETM/outputs/vocab/{job_id}_vocab.json - Vocabulary mapping
- ETM/outputs/bow/{job_id}_bow.npz - BOW matrix (N×V)
SIDE EFFECTS:
- Creates ETM/outputs/ directory structure
- Generates vocabulary and BOW representations
- Logs processing steps
FAILURE MODES:
- Script execution failure → log error + status="failed"
- Invalid input data → log error + status="failed"
- Output shape mismatch → log error + status="failed"
RULE:
- Must call scripts/run_engine_a.py with correct parameters
- Must validate output shapes: bow.npz shape (N,V) matches vocab.json length
- Path contract: inputs from data/{job_id}/, outputs to ETM/outputs/
- Must ensure vocabulary and BOW consistency
"""

import os
import json
import numpy as np
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class BowAgent:
    """Agent responsible for generating BOW representation and vocabulary"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp"):
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"BowAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def process(self, job_id: str) -> Dict[str, Any]:
        """
        Generate BOW representation and vocabulary
        
        Args:
            job_id: Unique identifier for analysis job
            
        Returns:
            Dict with processing results and status
        """
        try:
            self.logger.info(f"Starting BOW generation for job_id: {job_id}")
            
            # Prepare paths
            data_path = self.base_dir / "data" / job_id / "data.csv"
            script_path = self.base_dir / "scripts" / "run_engine_a.py"
            
            # Create output directories
            vocab_dir = self.base_dir / "ETM" / "outputs" / "vocab"
            bow_dir = self.base_dir / "ETM" / "outputs" / "bow"
            vocab_dir.mkdir(parents=True, exist_ok=True)
            bow_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output paths
            vocab_output = vocab_dir / f"{job_id}_vocab.json"
            bow_output = bow_dir / f"{job_id}_bow.npz"
            
            # Call engine A script
            cmd = [
                "python", str(script_path),
                "--input", str(data_path),
                "--vocab_output", str(vocab_output),
                "--bow_output", str(bow_output),
                "--job_id", job_id
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
            
            # Validate outputs
            if not vocab_output.exists():
                raise FileNotFoundError(f"Vocabulary output not created: {vocab_output}")
            if not bow_output.exists():
                raise FileNotFoundError(f"BOW output not created: {bow_output}")
            
            # Validate shapes
            with open(vocab_output, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            bow_data = np.load(bow_output)
            bow_matrix = bow_data['bow']  # Assuming 'bow' key in npz file
            
            # Get vocab list from vocab_data (matching THETA-main format)
            vocab_list = vocab_data.get('vocab', vocab_data)  # Support both formats
            vocab_size = len(vocab_list) if isinstance(vocab_list, list) else vocab_data.get('vocab_size', len(vocab_data))
            
            if vocab_size != bow_matrix.shape[1]:
                raise ValueError(f"Shape mismatch: vocab length {vocab_size} != BOW columns {bow_matrix.shape[1]}")
            
            # Log success
            self._log(job_id, f"BOW generation successful. Shape: {bow_matrix.shape}, Vocab size: {vocab_size}")
            
            return {
                "status": "success",
                "job_id": job_id,
                "vocab_path": str(vocab_output),
                "bow_path": str(bow_output),
                "bow_shape": bow_matrix.shape,
                "vocab_size": vocab_size
            }
            
        except subprocess.CalledProcessError as e:
            self._log(job_id, f"BOW generation failed: {e.stderr}", error=True)
            return {
                "status": "failed",
                "job_id": job_id,
                "error": f"Script execution failed: {e.stderr}"
            }
        except Exception as e:
            self._log(job_id, f"BOW generation failed: {str(e)}", error=True)
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
        log_entry = f"[{timestamp}] [{level}] BowAgent: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)

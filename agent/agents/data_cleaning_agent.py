"""
PLAN: Data validation and configuration setup for topic model analysis pipeline
WHAT: Validates input data and ensures proper configuration for downstream processing
WHY: Ensures data quality and consistency before expensive processing steps
INPUTS: 
- job_id: str - Unique identifier for the analysis job
- data_path: str - Path to input CSV file (data/{job_id}/data.csv)
OUTPUTS:
- result/{job_id}/config.yaml - Configuration parameters snapshot
- result/{job_id}/log.txt - Processing log entries
SIDE EFFECTS:
- Creates result directory structure
- Validates data format and schema
- Writes configuration to disk
FAILURE MODES:
- Invalid data format → log error + status="failed"
- Missing required columns → log error + status="failed"
- File not found → log error + status="failed"
RULE:
- Must validate CSV schema before processing
- Must create result/{job_id}/ directory if not exists
- Must write config.yaml with all parameters
- Must append to log.txt with timestamps
- Shape validation: data.csv must have at least 1 row and 1 column
- Path contract: inputs from data/{job_id}/, outputs to result/{job_id}/
"""

import os
import yaml
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class DataCleaningAgent:
    """Agent responsible for data validation and configuration setup"""
    
    def __init__(self, base_dir: str = "/root/autodl-tmp"):
        self.base_dir = Path(base_dir)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"DataCleaningAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        return logger
    
    def process(self, job_id: str) -> Dict[str, Any]:
        """
        Process data cleaning and validation
        
        Args:
            job_id: Unique identifier for the analysis job
            
        Returns:
            Dict with processing results and status
        """
        try:
            self.logger.info(f"Starting data cleaning for job_id: {job_id}")
            
            # Validate input data exists
            data_path = self.base_dir / "data" / job_id / "data.csv"
            if not data_path.exists():
                raise FileNotFoundError(f"Input data not found: {data_path}")
            
            # Validate data format
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError("Input data is empty")
            
            # Create result directory
            result_dir = self.base_dir / "result" / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Create configuration
            config = {
                "job_id": job_id,
                "data_path": str(data_path),
                "data_shape": df.shape,
                "columns": list(df.columns),
                "processing_time": datetime.now().isoformat(),
                "agent": "DataCleaningAgent"
            }
            
            # Write configuration
            config_path = result_dir / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # Log success
            self._log(job_id, f"Data validation successful. Shape: {df.shape}")
            
            return {
                "status": "success",
                "job_id": job_id,
                "data_shape": df.shape,
                "config_path": str(config_path)
            }
            
        except Exception as e:
            self._log(job_id, f"Data cleaning failed: {str(e)}", error=True)
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
        log_entry = f"[{timestamp}] [{level}] DataCleaningAgent: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)

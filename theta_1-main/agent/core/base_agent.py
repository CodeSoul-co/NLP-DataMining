"""
Base Agent
Base class for all agents, defines common interfaces and methods.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Base Agent Class
    
    All agents should inherit from this class and implement the process method.
    """
    
    def __init__(self, base_dir: str = "/root/autodl-tmp", name: str = "BaseAgent"):
        self.base_dir = Path(base_dir)
        self.name = name
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this agent"""
        logger = logging.getLogger(f"{self.name}_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # Avoid adding duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def process(self, job_id: str, **kwargs) -> Dict[str, Any]:
        """
        Abstract method for processing tasks
        
        Args:
            job_id: Job identifier
            **kwargs: Additional parameters
            
        Returns:
            Processing result dictionary
        """
        pass
    
    def _log(self, job_id: str, message: str, error: bool = False):
        """
        Log message to file and console
        
        Args:
            job_id: Job identifier
            message: Log message
            error: Whether this is an error log
        """
        log_path = self.base_dir / "result" / job_id / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "ERROR" if error else "INFO"
        log_entry = f"[{timestamp}] [{level}] {self.name}: {message}\n"
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if error:
            self.logger.error(message)
        else:
            self.logger.info(message)
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file"""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_json(self, data: Dict[str, Any], path: Path):
        """Save data to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists"""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_result_dir(self, job_id: str) -> Path:
        """Get job result directory"""
        return self._ensure_dir(self.base_dir / "result" / job_id)
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        result_path = self.get_result_dir(job_id) / "analysis_result.json"
        return self._load_json(result_path)

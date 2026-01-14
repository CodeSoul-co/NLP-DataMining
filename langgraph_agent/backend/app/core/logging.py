"""
Logging Configuration
Structured logging for the THETA Agent System
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import settings


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class AgentLogger:
    """
    Specialized logger for agent execution with step tracking
    """
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.logger = get_logger(f"agent.{task_id}")
        self.steps: list = []
        self.start_time = datetime.now()
    
    def log_step(self, step_name: str, status: str, message: str, **kwargs):
        """Log an agent step"""
        step_info = {
            "step": step_name,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.steps.append(step_info)
        
        log_msg = f"[{self.task_id}] {step_name}: {status} - {message}"
        
        if status == "error":
            self.logger.error(log_msg)
        elif status == "warning":
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
        
        return step_info
    
    def get_summary(self) -> dict:
        """Get execution summary"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        return {
            "task_id": self.task_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_steps": len(self.steps),
            "steps": self.steps
        }

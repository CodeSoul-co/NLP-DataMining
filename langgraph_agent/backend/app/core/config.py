"""
Application Configuration
Centralized settings for the THETA Agent System
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "THETA"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Paths
    BASE_DIR: Path = Path("/root/autodl-tmp")
    ETM_DIR: Path = Field(default_factory=lambda: Path("/root/autodl-tmp/ETM"))
    DATA_DIR: Path = Field(default_factory=lambda: Path("/root/autodl-tmp/data"))
    RESULT_DIR: Path = Field(default_factory=lambda: Path("/root/autodl-tmp/result"))
    EMBEDDING_DIR: Path = Field(default_factory=lambda: Path("/root/autodl-tmp/embedding"))
    QWEN_MODEL_PATH: Path = Field(default_factory=lambda: Path("/root/autodl-tmp/qwen3_embedding_0.6B"))
    
    # GPU Configuration
    GPU_ID: int = 1  # Use GPU 1, avoid GPU 0
    DEVICE: str = "cuda"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    
    # Agent Configuration
    DEFAULT_NUM_TOPICS: int = 20
    DEFAULT_VOCAB_SIZE: int = 5000
    DEFAULT_EPOCHS: int = 50
    DEFAULT_BATCH_SIZE: int = 64
    
    # Checkpointer
    CHECKPOINT_DIR: Path = Field(default_factory=lambda: Path("/root/autodl-tmp/langgraph_agent/checkpoints"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def get_result_path(self, dataset: str, mode: str) -> Path:
        """Get result directory for a specific dataset and mode"""
        return self.RESULT_DIR / dataset / mode
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        if not self.DATA_DIR.exists():
            return []
        return [d.name for d in self.DATA_DIR.iterdir() if d.is_dir()]
    
    def get_available_results(self) -> List[dict]:
        """Get list of available result directories"""
        results = []
        if not self.RESULT_DIR.exists():
            return results
        for dataset_dir in self.RESULT_DIR.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name != "README.md":
                for mode_dir in dataset_dir.iterdir():
                    if mode_dir.is_dir():
                        results.append({
                            "dataset": dataset_dir.name,
                            "mode": mode_dir.name,
                            "path": str(mode_dir)
                        })
        return results


settings = Settings()

# Ensure directories exist
settings.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

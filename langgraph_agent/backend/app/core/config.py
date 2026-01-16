"""
Application Configuration
Centralized settings for the THETA Agent System
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


def get_project_root() -> Path:
    """Get the project root directory"""
    # 从 config.py 向上找到项目根目录
    # config.py -> core -> app -> backend -> langgraph_agent -> THETA
    return Path(__file__).parent.parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "THETA"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 模拟模式 - 在没有完整 ETM 环境时使用
    SIMULATION_MODE: bool = True  # 设为 False 启用真实训练
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Paths - 动态获取项目根目录
    @property
    def BASE_DIR(self) -> Path:
        return get_project_root()
    
    @property
    def ETM_DIR(self) -> Path:
        return self.BASE_DIR / "ETM"
    
    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "data"
    
    @property
    def RESULT_DIR(self) -> Path:
        return self.BASE_DIR / "result"
    
    @property
    def EMBEDDING_DIR(self) -> Path:
        return self.BASE_DIR / "embedding"
    
    @property
    def QWEN_MODEL_PATH(self) -> Path:
        return self.BASE_DIR / "qwen3_embedding_0.6B"
    
    # GPU Configuration
    GPU_ID: int = 0  # 使用 GPU 0
    DEVICE: str = "cuda"
    
    # CORS - 支持从环境变量读取，逗号分隔
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000", 
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ],
        description="Allowed CORS origins (comma-separated in env var)"
    )
    
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    
    # Agent Configuration
    DEFAULT_NUM_TOPICS: int = 20
    DEFAULT_VOCAB_SIZE: int = 5000
    DEFAULT_EPOCHS: int = 50
    DEFAULT_BATCH_SIZE: int = 64
    
    # Checkpointer
    @property
    def CHECKPOINT_DIR(self) -> Path:
        return self.BASE_DIR / "langgraph_agent" / "checkpoints"
    
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
try:
    settings.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass  # 忽略目录创建错误

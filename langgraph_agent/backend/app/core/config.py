"""
Application Configuration
Centralized settings for the THETA Agent System
"""

import os
from pathlib import Path
from typing import Optional, List, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


def get_project_root() -> Path:
    """Get the project root directory"""
    env_root = os.environ.get('THETA_PROJECT_ROOT')
    if env_root:
        return Path(env_root)
    
    # From config.py -> core -> app -> backend -> langgraph_agent -> THETA
    return Path(__file__).parent.parent.parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # ==================== Application ====================
    APP_NAME: str = "THETA"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SIMULATION_MODE: bool = False  # True = 模拟模式（无 GPU/PAI 时）
    
    # ==================== Server ====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # ==================== Database (PostgreSQL) ====================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/theta",
        description="PostgreSQL connection URL"
    )
    
    # ==================== Redis ====================
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # ==================== Paths ====================
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
    def CHECKPOINT_DIR(self) -> Path:
        return self.BASE_DIR / "langgraph_agent" / "checkpoints"
    
    # ==================== GPU Configuration ====================
    GPU_ID: int = 0
    DEVICE: str = "cuda"
    
    # ==================== CORS ====================
    CORS_ORIGINS: Union[str, List[str]] = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Allowed CORS origins (comma-separated)"
    )
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    # ==================== WebSocket ====================
    WS_HEARTBEAT_INTERVAL: int = 30
    
    # ==================== Agent Configuration ====================
    DEFAULT_NUM_TOPICS: int = 20
    DEFAULT_VOCAB_SIZE: int = 5000
    DEFAULT_EPOCHS: int = 50
    DEFAULT_BATCH_SIZE: int = 64
    
    # ==================== Security / Authentication ====================
    SECRET_KEY: str = Field(
        default="theta-secure-key-change-in-production-2025",
        description="Secret key for JWT encoding"
    )
    ACCESS_TOKEN_EXPIRE_DAYS: int = 30
    
    # ==================== Qwen API (Chat) ====================
    QWEN_API_KEY: Optional[str] = Field(default=None, description="Qwen API Key")
    QWEN_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen-turbo"
    
    # ==================== 阿里云 OSS ====================
    OSS_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    OSS_ACCESS_KEY_SECRET: Optional[str] = Field(default=None)
    OSS_BUCKET_NAME: Optional[str] = Field(default=None)
    OSS_ENDPOINT: Optional[str] = Field(default=None, description="如 oss-cn-hongkong.aliyuncs.com")
    OSS_INTERNAL_ENDPOINT: Optional[str] = Field(default=None, description="VPC 内网端点")
    
    @property
    def OSS_ENABLED(self) -> bool:
        return bool(
            self.OSS_ACCESS_KEY_ID and self.OSS_ACCESS_KEY_SECRET
            and self.OSS_BUCKET_NAME and self.OSS_ENDPOINT
        )
    
    # ==================== 阿里云 PAI-DLC (训练) ====================
    PAI_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    PAI_ACCESS_KEY_SECRET: Optional[str] = Field(default=None)
    PAI_REGION: str = Field(default="cn-hongkong", description="如 cn-hongkong, cn-shanghai")
    PAI_WORKSPACE_ID: Optional[str] = Field(default=None)
    PAI_RESOURCE_GROUP_ID: Optional[str] = Field(default=None)
    PAI_TRAINING_IMAGE: str = Field(
        default="registry.cn-hongkong.aliyuncs.com/theta/etm-training:latest",
        description="训练镜像地址"
    )
    
    @property
    def PAI_ENABLED(self) -> bool:
        return bool(
            self.PAI_ACCESS_KEY_ID and self.PAI_ACCESS_KEY_SECRET
            and self.PAI_WORKSPACE_ID
        )
    
    # ==================== 阿里云 PAI-EAS (推理) ====================
    EAS_ENDPOINT: Optional[str] = Field(default=None, description="EAS 服务端点")
    EAS_TOKEN: Optional[str] = Field(default=None, description="EAS 服务 Token")
    
    @property
    def EAS_ENABLED(self) -> bool:
        return bool(self.EAS_ENDPOINT and self.EAS_TOKEN)
    
    # ==================== Model Config ====================
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    # ==================== Helper Methods ====================
    def get_result_path(self, dataset: str, mode: str) -> Path:
        return self.RESULT_DIR / dataset / mode
    
    def get_available_datasets(self) -> List[str]:
        if not self.DATA_DIR.exists():
            return []
        return [d.name for d in self.DATA_DIR.iterdir() if d.is_dir()]
    
    def get_available_results(self) -> List[dict]:
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
    pass

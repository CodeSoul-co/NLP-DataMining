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
    
    # ==================== 阿里云统一凭证 ====================
    # 优先使用 ALIBABA_CLOUD_* 变量名（与 theta_1-main、dlc_client 统一）
    # 同时兼容旧的 OSS_ACCESS_KEY_ID / PAI_ACCESS_KEY_ID
    ALIBABA_CLOUD_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    ALIBABA_CLOUD_ACCESS_KEY_SECRET: Optional[str] = Field(default=None)
    ALIBABA_CLOUD_REGION: str = Field(default="cn-shanghai", description="如 cn-shanghai, cn-hongkong")

    # 旧变量名兼容
    OSS_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    OSS_ACCESS_KEY_SECRET: Optional[str] = Field(default=None)
    PAI_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    PAI_ACCESS_KEY_SECRET: Optional[str] = Field(default=None)

    @property
    def CLOUD_ACCESS_KEY_ID(self) -> Optional[str]:
        return self.ALIBABA_CLOUD_ACCESS_KEY_ID or self.OSS_ACCESS_KEY_ID or self.PAI_ACCESS_KEY_ID

    @property
    def CLOUD_ACCESS_KEY_SECRET(self) -> Optional[str]:
        return self.ALIBABA_CLOUD_ACCESS_KEY_SECRET or self.OSS_ACCESS_KEY_SECRET or self.PAI_ACCESS_KEY_SECRET

    # ==================== 阿里云 OSS ====================
    OSS_BUCKET_NAME: Optional[str] = Field(default=None, description="OSS Bucket 名称")
    OSS_BUCKET: Optional[str] = Field(default=None, description="兼容 theta_1-main 的 OSS_BUCKET")
    OSS_ENDPOINT: Optional[str] = Field(default=None, description="如 oss-cn-shanghai.aliyuncs.com")
    OSS_INTERNAL_ENDPOINT: Optional[str] = Field(default=None, description="VPC 内网端点")

    @property
    def RESOLVED_OSS_BUCKET(self) -> Optional[str]:
        return self.OSS_BUCKET_NAME or self.OSS_BUCKET

    @property
    def OSS_ENABLED(self) -> bool:
        if self.SIMULATION_MODE:
            return False
        return bool(
            self.CLOUD_ACCESS_KEY_ID and self.CLOUD_ACCESS_KEY_SECRET
            and self.RESOLVED_OSS_BUCKET and self.OSS_ENDPOINT
        )
    
    # ==================== 阿里云 PAI-DLC (训练) ====================
    PAI_REGION: str = Field(default="cn-shanghai", description="如 cn-shanghai, cn-hongkong")
    PAI_WORKSPACE_ID: Optional[str] = Field(default=None, description="兼容旧变量名")
    DLC_WORKSPACE_ID: Optional[str] = Field(default=None, description="DLC 工作空间 ID")
    PAI_RESOURCE_GROUP_ID: Optional[str] = Field(default=None)
    OSS_DATASET_ID: Optional[str] = Field(default=None, description="PAI 数据集 ID，用于挂载 OSS 到 /mnt")
    DLC_INSTANCE_TYPE: str = Field(
        default="ecs.gn7i-c8g1.2xlarge",
        description="DLC 实例规格，如 ecs.gn7i-c8g1.2xlarge (A10 GPU)"
    )
    PAI_TRAINING_IMAGE: str = Field(
        default="dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/pytorch-training:2.7-gpu-py312-cu128-ubuntu24.04",
        description="DLC 训练镜像地址"
    )

    @property
    def RESOLVED_WORKSPACE_ID(self) -> Optional[str]:
        return self.DLC_WORKSPACE_ID or self.PAI_WORKSPACE_ID

    @property
    def RESOLVED_REGION(self) -> str:
        return self.ALIBABA_CLOUD_REGION or self.PAI_REGION

    @property
    def PAI_ENABLED(self) -> bool:
        return bool(
            self.CLOUD_ACCESS_KEY_ID and self.CLOUD_ACCESS_KEY_SECRET
            and self.RESOLVED_WORKSPACE_ID
        )

    @property
    def DLC_ENABLED(self) -> bool:
        """OSS + DLC 完整训练链路是否可用"""
        if self.SIMULATION_MODE:
            return False
        return self.OSS_ENABLED and self.PAI_ENABLED and bool(self.OSS_DATASET_ID)
    
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

    def get_user_result_path(self, user_id: int, dataset: str, mode: str) -> Path:
        """用户隔离的结果路径: result/{user_id}/{dataset}/{mode}/"""
        return self.RESULT_DIR / str(user_id) / dataset / mode
    
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

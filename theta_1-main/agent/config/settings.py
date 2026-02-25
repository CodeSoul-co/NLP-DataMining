"""
Settings Module
系统配置管理
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Settings:
    """系统配置"""
    
    # 基础路径
    base_dir: str = "/root/autodl-tmp"
    result_dir: str = "result"
    
    # 数据路径
    data_dir: str = "data"
    embedding_dir: str = "result/0.6B"
    bow_dir: str = "result/0.6B"
    
    # 模型路径
    model_dir: str = "models"
    qwen_model_path: str = "/root/autodl-tmp/models/Qwen2.5-0.6B"
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载配置"""
        return cls(
            base_dir=os.environ.get("THETA_ROOT", "/root/autodl-tmp"),
            api_host=os.environ.get("API_HOST", "0.0.0.0"),
            api_port=int(os.environ.get("API_PORT", "8000")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            qwen_model_path=os.environ.get("QWEN_MODEL_PATH", "/root/autodl-tmp/models/Qwen2.5-0.6B")
        )
    
    @property
    def base_path(self) -> Path:
        return Path(self.base_dir)
    
    @property
    def result_path(self) -> Path:
        return self.base_path / self.result_dir
    
    @property
    def data_path(self) -> Path:
        return self.base_path / self.data_dir
    
    @property
    def model_path(self) -> Path:
        return self.base_path / self.model_dir


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings():
    """重置全局配置"""
    global _settings
    _settings = None

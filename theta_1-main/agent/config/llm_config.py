"""
LLM Configuration Manager
统一管理LLM API配置，支持多种Provider

支持的Provider:
- qwen (阿里云通义千问) - 默认
- openai (OpenAI API)
- local (本地模型)
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LLMConfig:
    """LLM配置类"""
    
    # Provider配置
    provider: str = "qwen"  # qwen, openai, local
    model: str = "qwen-plus"
    
    # API配置
    api_key: str = ""
    base_url: str = ""
    
    # 生成参数
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 4096
    
    # 超时配置
    timeout: int = 120
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMConfigManager:
    """
    LLM配置管理器
    
    优先级（从高到低）：
    1. 显式传入的配置
    2. 环境变量
    3. 默认值
    """
    
    # 默认配置
    DEFAULT_CONFIGS = {
        "qwen": {
            "provider": "qwen",
            "model": "qwen-plus",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 4096,
        },
        "openai": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 4096,
        },
        "local": {
            "provider": "local",
            "model": "qwen2.5-7b",
            "base_url": "http://localhost:8000/v1",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 4096,
        }
    }
    
    # 环境变量映射
    ENV_MAPPING = {
        "api_key": ["DASHSCOPE_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"],
        "base_url": ["DASHSCOPE_BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL"],
        "model": ["QWEN_MODEL", "OPENAI_MODEL", "LLM_MODEL"],
        "temperature": ["QWEN_TEMPERATURE", "LLM_TEMPERATURE"],
        "top_p": ["QWEN_TOP_P", "LLM_TOP_P"],
    }
    
    @classmethod
    def get_config(
        cls, 
        provider: str = "qwen",
        override: Optional[Dict[str, Any]] = None
    ) -> LLMConfig:
        """
        获取LLM配置
        
        Args:
            provider: LLM提供商 (qwen, openai, local)
            override: 覆盖配置
            
        Returns:
            LLMConfig实例
        """
        # 获取默认配置
        if provider not in cls.DEFAULT_CONFIGS:
            provider = "qwen"
        
        config_dict = cls.DEFAULT_CONFIGS[provider].copy()
        
        # 从环境变量读取
        for key, env_vars in cls.ENV_MAPPING.items():
            for env_var in env_vars:
                value = os.environ.get(env_var)
                if value:
                    # 类型转换
                    if key in ["temperature", "top_p"]:
                        config_dict[key] = float(value)
                    elif key == "max_tokens":
                        config_dict[key] = int(value)
                    else:
                        config_dict[key] = value
                    break
        
        # 应用覆盖配置
        if override:
            config_dict.update(override)
        
        return LLMConfig(**config_dict)
    
    @classmethod
    def get_qwen_config(cls, override: Optional[Dict[str, Any]] = None) -> LLMConfig:
        """获取Qwen配置的快捷方法"""
        return cls.get_config("qwen", override)
    
    @classmethod
    def get_openai_config(cls, override: Optional[Dict[str, Any]] = None) -> LLMConfig:
        """获取OpenAI配置的快捷方法"""
        return cls.get_config("openai", override)
    
    @classmethod
    def validate_config(cls, config: LLMConfig) -> tuple:
        """
        验证配置是否有效
        
        Returns:
            (is_valid: bool, error_message: str)
        """
        if not config.api_key:
            return False, "API key is required. Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable."
        
        if not config.base_url:
            return False, "Base URL is required."
        
        if not config.model:
            return False, "Model name is required."
        
        return True, "OK"


# 便捷函数
def get_llm_config(provider: str = "qwen", **kwargs) -> LLMConfig:
    """获取LLM配置的便捷函数"""
    return LLMConfigManager.get_config(provider, kwargs if kwargs else None)


def get_default_llm_config() -> Dict[str, Any]:
    """获取默认LLM配置字典（用于向后兼容）"""
    config = LLMConfigManager.get_qwen_config()
    return config.to_dict()

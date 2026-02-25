"""
API Configuration Module
Centralized configuration for API endpoints, authentication, and service settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class LLMAPIConfig:
    """LLM API configuration"""
    
    # Provider: deepseek, qwen, openai
    provider: str = "deepseek"
    
    # Model settings
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    
    # API endpoints
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_env(cls) -> "LLMAPIConfig":
        """Load configuration from environment variables"""
        provider = os.environ.get("LLM_PROVIDER", "deepseek")
        
        # Select API key and base_url based on provider
        if provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        elif provider == "qwen":
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            model = os.environ.get("QWEN_MODEL", "qwen-plus")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        else:
            api_key = os.environ.get("LLM_API_KEY", "")
            base_url = os.environ.get("LLM_BASE_URL", "")
            model = os.environ.get("LLM_MODEL", "")
        
        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2000")),
            timeout=int(os.environ.get("LLM_TIMEOUT", "60")),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key[:8] + "..." if self.api_key else "",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }


@dataclass
class ServerConfig:
    """FastAPI server configuration"""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    
    # CORS settings
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_methods: list = field(default_factory=lambda: ["*"])
    cors_headers: list = field(default_factory=lambda: ["*"])
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables"""
        return cls(
            host=os.environ.get("API_HOST", "0.0.0.0"),
            port=int(os.environ.get("API_PORT", "8000")),
            debug=os.environ.get("API_DEBUG", "false").lower() == "true",
            reload=os.environ.get("API_RELOAD", "false").lower() == "true",
            workers=int(os.environ.get("API_WORKERS", "1")),
        )


@dataclass
class StorageConfig:
    """Storage paths configuration"""
    
    base_dir: str = "/root/autodl-tmp"
    result_dir: str = "result"
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Load configuration from environment variables"""
        base = os.environ.get("THETA_ROOT", "/root/autodl-tmp")
        return cls(
            base_dir=base,
            result_dir=os.environ.get("RESULT_DIR", "result"),
            data_dir=os.environ.get("DATA_DIR", "data"),
            model_dir=os.environ.get("MODEL_DIR", "models"),
            log_dir=os.environ.get("LOG_DIR", "logs"),
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
    
    @property
    def log_path(self) -> Path:
        return self.base_path / self.log_dir


@dataclass
class APIConfig:
    """
    Main API configuration class
    Aggregates all configuration sections
    """
    
    llm: LLMAPIConfig = field(default_factory=LLMAPIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Application metadata
    app_name: str = "Topic Model Agent API"
    app_version: str = "1.0.0"
    app_description: str = "API for topic model analysis and interpretation"
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load all configuration from environment variables"""
        return cls(
            llm=LLMAPIConfig.from_env(),
            server=ServerConfig.from_env(),
            storage=StorageConfig.from_env(),
            app_name=os.environ.get("APP_NAME", "Topic Model Agent API"),
            app_version=os.environ.get("APP_VERSION", "1.0.0"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (safe for logging)"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "llm": self.llm.to_dict(),
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "debug": self.server.debug,
            },
            "storage": {
                "base_dir": self.storage.base_dir,
                "result_dir": self.storage.result_dir,
            },
        }


# Global configuration instance
_api_config: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """Get global API configuration instance"""
    global _api_config
    if _api_config is None:
        _api_config = APIConfig.from_env()
    return _api_config


def reset_api_config():
    """Reset global configuration (useful for testing)"""
    global _api_config
    _api_config = None


def load_api_config_from_file(config_path: str) -> APIConfig:
    """
    Load configuration from a YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        APIConfig instance
    """
    import json
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if path.suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config files")
    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    # Build config from data
    config = APIConfig()
    
    if "llm" in data:
        for key, value in data["llm"].items():
            if hasattr(config.llm, key):
                setattr(config.llm, key, value)
    
    if "server" in data:
        for key, value in data["server"].items():
            if hasattr(config.server, key):
                setattr(config.server, key, value)
    
    if "storage" in data:
        for key, value in data["storage"].items():
            if hasattr(config.storage, key):
                setattr(config.storage, key, value)
    
    return config

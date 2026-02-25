"""
LangChain LLM Integration

Provides a unified ChatModel interface for multiple LLM providers (DeepSeek, Qwen, OpenAI).
Reads configuration from environment variables and .env file.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def _load_env():
    """Load .env file from agent directory if exists."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env()


def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    Create a LangChain ChatModel based on provider configuration.

    Priority: explicit args > environment variables > defaults.

    Args:
        provider: LLM provider (deepseek, qwen, openai). Default from LLM_PROVIDER env.
        model: Model name. Default from provider-specific env var.
        temperature: Sampling temperature. Default from LLM_TEMPERATURE env.
        max_tokens: Max output tokens. Default from LLM_MAX_TOKENS env.
        api_key: API key. Default from provider-specific env var.
        base_url: API base URL. Default from provider-specific env var.

    Returns:
        LangChain BaseChatModel instance.
    """
    from langchain_openai import ChatOpenAI

    provider = (provider or os.environ.get("LLM_PROVIDER", "deepseek")).lower()
    temperature = temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", "0.7"))
    max_tokens = max_tokens or int(os.environ.get("LLM_MAX_TOKENS", "2000"))
    timeout = int(os.environ.get("LLM_TIMEOUT", "120"))

    if provider == "deepseek":
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    elif provider == "qwen":
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = model or os.environ.get("QWEN_MODEL", "qwen-plus")
    elif provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    else:
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        model = model or os.environ.get("LLM_MODEL", "")

    if not api_key:
        raise ValueError(
            f"API key not set for provider '{provider}'. "
            f"Set the appropriate environment variable (e.g. DEEPSEEK_API_KEY, DASHSCOPE_API_KEY, OPENAI_API_KEY)."
        )

    logger.info(f"Creating ChatModel: provider={provider}, model={model}, base_url={base_url[:40]}...")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )

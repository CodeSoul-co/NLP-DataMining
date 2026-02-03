"""
LLM Utils
Utility functions for LLM API calls.
Supports DeepSeek, Qwen, and OpenAI compatible APIs.
"""

import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def get_llm_client():
    """
    Get OpenAI-compatible client based on provider configuration.
    
    Returns:
        OpenAI client instance
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    elif provider == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    else:
        api_key = os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get("LLM_BASE_URL", "")
    
    if not api_key:
        raise ValueError(f"API key not set for provider: {provider}")
    
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm_api(
    messages: List[Dict[str, str]],
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: int = 60
) -> str:
    """
    Call LLM API using OpenAI-compatible interface.
    
    Args:
        messages: Message list [{"role": "system/user/assistant", "content": "..."}]
        model: Model name (default from env)
        api_key: API key (default from env)
        base_url: API base URL (default from env)
        temperature: Temperature parameter
        max_tokens: Maximum tokens
        timeout: Timeout in seconds
        
    Returns:
        LLM response text
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    # Get provider from environment
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    
    # Set defaults based on provider
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
        model = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    else:
        api_key = api_key or os.environ.get("LLM_API_KEY", "")
        base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        model = model or os.environ.get("LLM_MODEL", "")
    
    if not api_key:
        raise ValueError(f"API key not provided for provider: {provider}")
    
    # Use OpenAI client for API call
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        raise Exception(f"LLM API error: {e}")


def format_messages(
    system_prompt: str,
    user_message: str,
    history: List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Format message list for LLM API.
    
    Args:
        system_prompt: System prompt
        user_message: User message
        history: Conversation history
        
    Returns:
        Formatted message list
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_message})
    
    return messages


def truncate_history(
    history: List[Dict[str, str]], 
    max_turns: int = 10
) -> List[Dict[str, str]]:
    """
    Truncate conversation history, keeping only recent turns.
    
    Args:
        history: Conversation history
        max_turns: Maximum number of turns to keep
        
    Returns:
        Truncated conversation history
    """
    if not history:
        return []
    
    # Each turn contains user and assistant messages
    max_messages = max_turns * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history

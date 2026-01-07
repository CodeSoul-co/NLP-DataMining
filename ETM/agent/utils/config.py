"""
Agent Configuration

Defines Agent configuration parameters, including model paths, parameter settings, etc.
"""

import os
from typing import Dict, Any, Optional


class AgentConfig:
    """
    Agent configuration class that defines Agent configuration parameters.
    """
    
    def __init__(
        self,
        etm_model_path: str,
        vocab_path: str,
        embedding_model_path: str = "/root/autodl-tmp/qwen3_embedding_0.6B",
        embedding_dim: int = 1024,
        device: Optional[str] = None,
        max_history_length: int = 10,
        max_topic_history_length: int = 5,
        use_faiss: bool = True,
        llm_model_name: str = "gpt-3.5-turbo",
        llm_api_key: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        register_default_tools: bool = True
    ):
        """
        Initialize Agent configuration.
        
        Args:
            etm_model_path: ETM model path
            vocab_path: Vocabulary path
            embedding_model_path: Qwen embedding model path
            embedding_dim: Embedding dimension
            device: Device ('cuda', 'cpu', or None for auto-selection)
            max_history_length: Maximum conversation history length
            max_topic_history_length: Maximum topic history length
            use_faiss: Whether to use FAISS for vector retrieval
            llm_model_name: Large language model name
            llm_api_key: Large language model API key
            llm_api_base: Large language model API base URL
            register_default_tools: Whether to register default tools
        """
        self.etm_model_path = etm_model_path
        self.vocab_path = vocab_path
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.device = device
        self.max_history_length = max_history_length
        self.max_topic_history_length = max_topic_history_length
        self.use_faiss = use_faiss
        self.llm_model_name = llm_model_name
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.register_default_tools = register_default_tools
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            "etm_model_path": self.etm_model_path,
            "vocab_path": self.vocab_path,
            "embedding_model_path": self.embedding_model_path,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "max_history_length": self.max_history_length,
            "max_topic_history_length": self.max_topic_history_length,
            "use_faiss": self.use_faiss,
            "llm_model_name": self.llm_model_name,
            "llm_api_key": self.llm_api_key,
            "llm_api_base": self.llm_api_base,
            "register_default_tools": self.register_default_tools
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            Configuration instance
        """
        return cls(
            etm_model_path=os.environ.get("ETM_MODEL_PATH", ""),
            vocab_path=os.environ.get("VOCAB_PATH", ""),
            embedding_model_path=os.environ.get("EMBEDDING_MODEL_PATH", "/root/autodl-tmp/qwen3_embedding_0.6B"),
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", "1024")),
            device=os.environ.get("DEVICE", None),
            max_history_length=int(os.environ.get("MAX_HISTORY_LENGTH", "10")),
            max_topic_history_length=int(os.environ.get("MAX_TOPIC_HISTORY_LENGTH", "5")),
            use_faiss=os.environ.get("USE_FAISS", "True").lower() == "true",
            llm_model_name=os.environ.get("LLM_MODEL_NAME", "gpt-3.5-turbo"),
            llm_api_key=os.environ.get("LLM_API_KEY", None),
            llm_api_base=os.environ.get("LLM_API_BASE", None),
            register_default_tools=os.environ.get("REGISTER_DEFAULT_TOOLS", "True").lower() == "true"
        )

"""
Topic Model Analysis Agents

This module contains all agents for the topic model analysis pipeline.
Each agent has specific responsibilities and follows strict path conventions.

Directory Structure:
- core/: Core agent implementations
- prompts/: LLM prompt templates
- utils/: Utility functions
- config/: Configuration management
- tests/: Test cases
"""

# Backward compatible imports - from core subdirectory
from .core.data_cleaning_agent import DataCleaningAgent
from .core.bow_agent import BowAgent
from .core.embedding_agent import EmbeddingAgent
from .core.etm_agent import ETMAgent
from .core.visualization_agent import VisualizationAgent
from .core.text_qa_agent import TextQAAgent
from .core.vision_qa_agent import VisionQAAgent
from .core.report_agent import ReportAgent
from .core.orchestrator_agent import OrchestratorAgent
from .core.result_interpreter_agent import ResultInterpreterAgent
from .config.llm_config import LLMConfig, LLMConfigManager, get_llm_config, get_default_llm_config

# LangChain agent (lazy imports to avoid startup cost)
from .langchain_agent import THETAAgent, get_agent, reset_agent
from .langchain_llm import get_chat_model

# Submodule imports
from . import core
from . import prompts
from . import utils
from . import config

__all__ = [
    # Agents
    "DataCleaningAgent",
    "BowAgent", 
    "EmbeddingAgent",
    "ETMAgent",
    "VisualizationAgent",
    "TextQAAgent",
    "VisionQAAgent",
    "ReportAgent",
    "OrchestratorAgent",
    "ResultInterpreterAgent",
    # LangChain Agent
    "THETAAgent",
    "get_agent",
    "reset_agent",
    "get_chat_model",
    # Config
    "LLMConfig",
    "LLMConfigManager",
    "get_llm_config",
    "get_default_llm_config",
    # Submodules
    "core",
    "prompts",
    "utils",
    "config"
]

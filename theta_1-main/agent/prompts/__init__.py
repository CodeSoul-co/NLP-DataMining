"""
Prompts Module
Contains all LLM prompt templates.

Two ways to use prompts:
1. Import from Python dictionaries (hardcoded)
2. Load from txt files (recommended, easy to modify)

txt file usage example:
    from agent.prompts import load_prompt, PromptLoader
    
    # Method 1: Use convenience function
    system_prompt = load_prompt("qa_system", "zh")
    
    # Method 2: Use loader instance
    loader = PromptLoader()
    system_prompt = loader.get("qa_system", "zh")
"""

# Python dictionary prompts (backward compatible)
from .metric_prompts import METRIC_INTERPRETATION_PROMPTS, SINGLE_METRIC_PROMPTS, QUALITY_TEXT, RELATION_TEXT
from .topic_prompts import TOPIC_ANALYSIS_PROMPTS
from .qa_prompts import QA_SYSTEM_PROMPTS, CONVERSATION_PROMPTS
from .report_prompts import REPORT_GENERATION_PROMPTS

# txt file loader (recommended)
from .prompt_loader import PromptLoader, get_prompt_loader, load_prompt, reload_prompts

__all__ = [
    # Loader (recommended)
    "PromptLoader",
    "get_prompt_loader",
    "load_prompt",
    "reload_prompts",
    # Python dictionaries (backward compatible)
    "METRIC_INTERPRETATION_PROMPTS",
    "SINGLE_METRIC_PROMPTS",
    "QUALITY_TEXT",
    "RELATION_TEXT",
    "TOPIC_ANALYSIS_PROMPTS",
    "QA_SYSTEM_PROMPTS",
    "CONVERSATION_PROMPTS",
    "REPORT_GENERATION_PROMPTS"
]

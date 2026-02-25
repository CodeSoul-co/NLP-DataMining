"""
LangChain Agent for THETA Topic Model Pipeline

Provides a ReAct agent that can:
- Execute pipeline steps (clean, prepare, train, evaluate, visualize)
- Query and interpret results
- Answer questions about topic modeling

Uses LangGraph for agent orchestration with tool calling.
"""

import os
import logging
from typing import Optional, Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from .langchain_llm import get_chat_model
from .langchain_tools import ALL_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are THETA Agent, an intelligent assistant for the THETA topic model analysis pipeline.

You help users with:
1. **Data Management**: List datasets, clean raw data, prepare BOW/embeddings
2. **Model Training**: Train THETA (with Qwen embeddings) and 11 baseline models (LDA, HDP, STM, BTM, NVDM, GSM, ProdLDA, CTM, ETM, DTM, BERTopic)
3. **Evaluation**: Evaluate models with 7 metrics (TD, iRBO, NPMI, C_V, UMass, Exclusivity, PPL)
4. **Visualization**: Generate 20+ chart types for trained models
5. **Result Interpretation**: Explain topics, metrics, and provide insights

**Important guidelines:**
- All scripts are non-interactive (pure command-line). Always provide all required parameters.
- For THETA: `--dataset` is required. Default model_size=0.6B, mode=zero_shot.
- For baselines: `--dataset` and `--models` are required.
- Data must be prepared (03_prepare_data.sh) before training.
- Use `list_datasets` to discover available data, `list_experiments` to find existing experiments.
- When interpreting results, focus on **content semantics** not algorithm details.
- Answer in the same language the user uses (Chinese → Chinese, English → English).

**Pipeline order:**
1. clean_data → 2. prepare_data → 3. train_theta / train_baseline → 4. evaluate_model → 5. visualize → 6. compare_models
"""


class THETAAgent:
    """LangChain-based agent for THETA topic model pipeline."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the THETA agent.

        Args:
            provider: LLM provider (deepseek, qwen, openai). Default from env.
            model: Model name. Default from env.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            api_key: API key override.
            base_url: Base URL override.
        """
        self.llm = get_chat_model(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        self.tools = ALL_TOOLS
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )
        # Session-based message history
        self._sessions: Dict[str, List] = {}
        logger.info("THETAAgent initialized with LangChain ReAct agent")

    def chat(
        self,
        message: str,
        session_id: str = "default",
    ) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: User message.
            session_id: Session ID for multi-turn conversation.

        Returns:
            Agent response text.
        """
        # Build message history
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        history = self._sessions[session_id]
        input_messages = list(history) + [HumanMessage(content=message)]

        try:
            result = self.agent.invoke({"messages": input_messages})
            response_messages = result.get("messages", [])

            # Extract the final AI response
            ai_response = ""
            for msg in reversed(response_messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    ai_response = msg.content
                    break

            if not ai_response:
                # Fallback: get last message content
                if response_messages:
                    ai_response = str(response_messages[-1].content)
                else:
                    ai_response = "No response generated."

            # Update session history (keep last 20 messages to avoid context overflow)
            history.append(HumanMessage(content=message))
            history.append(AIMessage(content=ai_response))
            if len(history) > 40:
                self._sessions[session_id] = history[-20:]

            return ai_response

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            return f"Agent error: {str(e)}"

    def chat_stream(
        self,
        message: str,
        session_id: str = "default",
    ):
        """
        Stream a response from the agent.

        Args:
            message: User message.
            session_id: Session ID.

        Yields:
            Chunks of the response as dicts with 'type' and 'content' keys.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        history = self._sessions[session_id]
        input_messages = list(history) + [HumanMessage(content=message)]

        try:
            full_response = ""
            for chunk in self.agent.stream({"messages": input_messages}):
                # chunk is a dict like {"agent": {"messages": [...]}} or {"tools": {"messages": [...]}}
                for node_name, node_output in chunk.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield {"type": "tool_call", "content": f"Calling: {tc['name']}({tc['args']})"}
                            elif msg.content:
                                full_response = msg.content
                                yield {"type": "text", "content": msg.content}

            # Update history
            history.append(HumanMessage(content=message))
            history.append(AIMessage(content=full_response))
            if len(history) > 40:
                self._sessions[session_id] = history[-20:]

        except Exception as e:
            logger.error(f"Agent stream error: {e}", exc_info=True)
            yield {"type": "error", "content": str(e)}

    def clear_session(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """List active session IDs."""
        return list(self._sessions.keys())


# Singleton instance
_agent_instance: Optional[THETAAgent] = None


def get_agent(**kwargs) -> THETAAgent:
    """Get or create the singleton THETAAgent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = THETAAgent(**kwargs)
    return _agent_instance


def reset_agent():
    """Reset the singleton agent instance."""
    global _agent_instance
    _agent_instance = None

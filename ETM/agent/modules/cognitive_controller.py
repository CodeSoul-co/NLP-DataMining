"""
Cognitive Controller Module

Integrates topic information with large language models to implement intent understanding,
context management, reasoning planning, and tool selection.
This module is the decision center of the Agent, responsible for coordinating various components.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from pathlib import Path

# Add project root directory to path
sys.path.append(str(Path(__file__).parents[2]))

# Import other modules
from agent.modules.topic_aware import TopicAwareModule
from agent.memory.memory_system import MemorySystem
from agent.utils.tool_registry import ToolRegistry
from agent.utils.llm_client import LLMClient


class CognitiveController:
    """
    Cognitive controller module that integrates topic information with large language models.
    
    Features:
    1. Intent understanding: Understand user intent combined with topic information
    2. Context management: Maintain conversation state and topic evolution
    3. Reasoning planning: Perform reasoning and planning based on topic relevance
    4. Tool selection: Select appropriate tools based on topic and intent
    """
    
    def __init__(
        self,
        topic_module: TopicAwareModule,
        memory_system: MemorySystem,
        llm_client: LLMClient,
        tool_registry: Optional[ToolRegistry] = None,
        dev_mode: bool = False
    ):
        """
        Initialize cognitive controller module.
        
        Args:
            topic_module: Topic-aware module
            memory_system: Memory system
            llm_client: Large language model client
            tool_registry: Tool registry (optional)
            dev_mode: Whether to enable development mode (print debug information)
        """
        self.topic_module = topic_module
        self.memory = memory_system
        self.llm = llm_client
        self.tools = tool_registry or ToolRegistry()
        self.dev_mode = dev_mode
        
        if self.dev_mode:
            print(f"[CognitiveController] Initialized successfully")
    
    def process_input(
        self,
        user_input: str,
        session_id: str,
        use_tools: bool = True,
        use_topic_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Process user input and generate response.
        
        Args:
            user_input: User input text
            session_id: Session ID
            use_tools: Whether to use tools
            use_topic_enhancement: Whether to use topic enhancement
            
        Returns:
            Dictionary containing response and metadata
        """
        # Get session context
        context = self.memory.get_context(session_id)
        
        # Get topic distribution
        topic_dist = self.topic_module.get_topic_distribution(user_input)
        dominant_topics = self.topic_module.get_dominant_topics(topic_dist)
        
        # Detect topic shift
        topic_shifted = False
        if context.get("last_topic_dist") is not None:
            topic_shifted = self.topic_module.detect_topic_shift(
                context["last_topic_dist"],
                topic_dist
            )
        
        # Build enhanced prompt
        if use_topic_enhancement:
            topic_info = self._get_topic_context(dominant_topics)
            enhanced_prompt = self._build_prompt(user_input, topic_info, context, topic_shifted)
        else:
            enhanced_prompt = user_input
        
        # Select relevant tools
        available_tools = None
        if use_tools:
            available_tools = self._select_relevant_tools(dominant_topics)
        
        # LLM reasoning
        response = self.llm.generate(
            prompt=enhanced_prompt,
            context=context,
            tools=available_tools
        )
        
        # Handle tool calls
        if use_tools and "tool_calls" in response:
            response = self._handle_tool_calls(response, session_id)
        
        # Update memory
        self.memory.add(
            session_id=session_id,
            user_input=user_input,
            response=response["content"],
            topic_dist=topic_dist,
            metadata={
                "dominant_topics": dominant_topics,
                "topic_shifted": topic_shifted
            }
        )
        
        # Return result
        result = {
            "content": response["content"],
            "topic_dist": topic_dist.tolist(),
            "dominant_topics": dominant_topics,
            "topic_shifted": topic_shifted
        }
        
        if "thinking" in response:
            result["thinking"] = response["thinking"]
        
        return result
    
    def _get_topic_context(
        self,
        dominant_topics: List[Tuple[int, float]]
    ) -> Dict[str, Any]:
        """
        Get topic context information.
        
        Args:
            dominant_topics: List of dominant topics [(topic_idx, weight), ...]
            
        Returns:
            Topic context information
        """
        topic_context = {
            "topics": []
        }
        
        for topic_idx, weight in dominant_topics:
            topic_words = self.topic_module.get_topic_words(topic_idx)
            
            topic_context["topics"].append({
                "id": topic_idx,
                "weight": float(weight),
                "keywords": [word for word, _ in topic_words],
                "keyword_weights": [float(w) for _, w in topic_words]
            })
        
        return topic_context
    
    def _build_prompt(
        self,
        user_input: str,
        topic_info: Dict[str, Any],
        context: Dict[str, Any],
        topic_shifted: bool
    ) -> str:
        """
        Build enhanced prompt.
        
        Args:
            user_input: User input
            topic_info: Topic information
            context: Conversation context
            topic_shifted: Whether topic has changed
            
        Returns:
            Enhanced prompt
        """
        # Base prompt
        prompt = user_input
        
        # Add topic information
        if topic_info["topics"]:
            prompt += "\n\nRelevant topic context:"
            
            for topic in topic_info["topics"]:
                keywords_str = ", ".join(topic["keywords"][:5])
                prompt += f"\n- Topic {topic['id']} (weight: {topic['weight']:.2f}): {keywords_str}"
        
        # If topic has changed, add note
        if topic_shifted and context.get("conversation_history"):
            prompt += "\n\nNote: The conversation topic seems to have changed."
        
        return prompt
    
    def _select_relevant_tools(
        self,
        dominant_topics: List[Tuple[int, float]]
    ) -> List[Dict[str, Any]]:
        """
        Select relevant tools based on topics.
        
        Args:
            dominant_topics: List of dominant topics
            
        Returns:
            List of tools
        """
        # If no tools, return empty list
        if not self.tools or not self.tools.get_all_tools():
            return []
        
        # Get all tools
        all_tools = self.tools.get_all_tools()
        
        # If no topics, return all tools
        if not dominant_topics:
            return all_tools
        
        # Select tools based on topics
        # More complex tool selection logic can be implemented here
        # Currently simply returns all tools
        return all_tools
    
    def _handle_tool_calls(
        self,
        response: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Handle tool calls.
        
        Args:
            response: LLM response
            session_id: Session ID
            
        Returns:
            Processed response
        """
        if not response.get("tool_calls"):
            return response
        
        tool_results = []
        
        for tool_call in response["tool_calls"]:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            
            # Call tool
            result = self.tools.call_tool(tool_name, tool_args)
            
            tool_results.append({
                "name": tool_name,
                "result": result
            })
        
        # Add tool results to context
        self.memory.add_tool_results(session_id, tool_results)
        
        # Generate final response using tool results
        final_response = self.llm.generate_with_tool_results(
            original_response=response,
            tool_results=tool_results,
            context=self.memory.get_context(session_id)
        )
        
        return final_response
    
    def plan(
        self,
        user_input: str,
        topic_dist: np.ndarray,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate plan based on user input and topic distribution.
        
        Args:
            user_input: User input
            topic_dist: Topic distribution
            context: Conversation context
            
        Returns:
            Plan
        """
        # Get dominant topics
        dominant_topics = self.topic_module.get_dominant_topics(topic_dist)
        
        # Build planning prompt
        planning_prompt = self._build_planning_prompt(user_input, dominant_topics, context)
        
        # Generate plan
        plan = self.llm.generate_plan(planning_prompt, context)
        
        return plan
    
    def _build_planning_prompt(
        self,
        user_input: str,
        dominant_topics: List[Tuple[int, float]],
        context: Dict[str, Any]
    ) -> str:
        """
        Build planning prompt.
        
        Args:
            user_input: User input
            dominant_topics: Dominant topics
            context: Conversation context
            
        Returns:
            Planning prompt
        """
        prompt = f"Please create a detailed response plan for the following user request:\n\n{user_input}\n\n"
        
        # Add topic information
        if dominant_topics:
            prompt += "Related topics:\n"
            
            for topic_idx, weight in dominant_topics:
                topic_words = self.topic_module.get_topic_words(topic_idx)
                words_str = ", ".join([word for word, _ in topic_words[:5]])
                prompt += f"- Topic {topic_idx} (weight: {weight:.2f}): {words_str}\n"
        
        # Add context information
        if context.get("conversation_history"):
            prompt += "\nConversation history summary:\n"
            # Conversation history summary can be added here
        
        prompt += "\nPlease provide the following:\n"
        prompt += "1. User intent analysis\n"
        prompt += "2. Information to retrieve\n"
        prompt += "3. Response steps\n"
        prompt += "4. Tools to use\n"
        
        return prompt
    
    def execute_plan(
        self,
        plan: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Execute plan.
        
        Args:
            plan: Plan
            session_id: Session ID
            
        Returns:
            Execution result
        """
        # Get context
        context = self.memory.get_context(session_id)
        
        # Execute steps in the plan
        results = []
        
        for step in plan.get("steps", []):
            step_type = step.get("type")
            step_params = step.get("params", {})
            
            if step_type == "tool_call":
                # Call tool
                tool_name = step_params.get("name")
                tool_args = step_params.get("arguments", {})
                
                result = self.tools.call_tool(tool_name, tool_args)
                results.append({"type": "tool_result", "result": result})
                
            elif step_type == "llm_call":
                # Call LLM
                prompt = step_params.get("prompt")
                
                response = self.llm.generate(prompt, context)
                results.append({"type": "llm_result", "result": response})
        
        # Generate final response
        final_response = self.llm.generate_from_plan_results(
            plan=plan,
            results=results,
            context=context
        )
        
        # Update memory
        self.memory.add_plan_execution(
            session_id=session_id,
            plan=plan,
            results=results,
            final_response=final_response
        )
        
        return final_response


# Test code
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cognitive controller module")
    parser.add_argument("--etm_model", type=str, required=True, help="ETM model path")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary path")
    parser.add_argument("--input", type=str, required=True, help="Test input")
    parser.add_argument("--dev_mode", action="store_true", help="Development mode")
    
    args = parser.parse_args()
    
    # Initialize components
    from agent.modules.topic_aware import TopicAwareModule
    from agent.memory.memory_system import MemorySystem
    from agent.utils.llm_client import LLMClient
    
    topic_module = TopicAwareModule(
        etm_model_path=args.etm_model,
        vocab_path=args.vocab,
        dev_mode=args.dev_mode
    )
    
    memory_system = MemorySystem()
    llm_client = LLMClient()
    
    # Initialize cognitive controller module
    controller = CognitiveController(
        topic_module=topic_module,
        memory_system=memory_system,
        llm_client=llm_client,
        dev_mode=args.dev_mode
    )
    
    # Process input
    session_id = "test_session"
    result = controller.process_input(args.input, session_id)
    
    print(f"Response: {result['content']}")
    print(f"Dominant topics: {result['dominant_topics']}")
    print(f"Topic shifted: {result['topic_shifted']}")

"""
Memory System

Manages conversation history, topic evolution, and knowledge cache.
This module is responsible for storing and retrieving Agent's memory.
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from collections import defaultdict

# Add project root directory to path
sys.path.append(str(Path(__file__).parents[2]))


class MemorySystem:
    """
    Memory system that manages conversation history, topic evolution, and knowledge cache.
    
    Features:
    1. Short-term memory: Stores recent conversation history
    2. Long-term memory: Stores important information and knowledge
    3. Topic memory: Tracks topic evolution
    """
    
    def __init__(
        self,
        max_history_length: int = 10,
        max_topic_history_length: int = 5,
        dev_mode: bool = False
    ):
        """
        Initialize memory system.
        
        Args:
            max_history_length: Maximum conversation history length
            max_topic_history_length: Maximum topic history length
            dev_mode: Whether to enable development mode (print debug information)
        """
        self.max_history_length = max_history_length
        self.max_topic_history_length = max_topic_history_length
        self.dev_mode = dev_mode
        
        # Session memory
        self.sessions = defaultdict(lambda: {
            "conversation_history": [],
            "topic_history": [],
            "tool_results": [],
            "plan_history": [],
            "knowledge_cache": {},
            "metadata": {}
        })
        
        if self.dev_mode:
            print(f"[MemorySystem] Initialized successfully")
    
    def add(
        self,
        session_id: str,
        user_input: str,
        response: str,
        topic_dist: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add conversation record to memory.
        
        Args:
            session_id: Session ID
            user_input: User input
            response: System response
            topic_dist: Topic distribution
            metadata: Metadata
        """
        # Get session
        session = self.sessions[session_id]
        
        # Add conversation record
        timestamp = time.time()
        
        conversation_entry = {
            "timestamp": timestamp,
            "user_input": user_input,
            "response": response,
            "metadata": metadata or {}
        }
        
        session["conversation_history"].append(conversation_entry)
        
        # Limit conversation history length
        if len(session["conversation_history"]) > self.max_history_length:
            session["conversation_history"] = session["conversation_history"][-self.max_history_length:]
        
        # Add topic record (if provided)
        if topic_dist is not None:
            topic_entry = {
                "timestamp": timestamp,
                "topic_dist": topic_dist.tolist(),
                "metadata": metadata or {}
            }
            
            session["topic_history"].append(topic_entry)
            
            # Limit topic history length
            if len(session["topic_history"]) > self.max_topic_history_length:
                session["topic_history"] = session["topic_history"][-self.max_topic_history_length:]
            
            # Update last topic distribution
            session["metadata"]["last_topic_dist"] = topic_dist.tolist()
    
    def add_tool_results(
        self,
        session_id: str,
        tool_results: List[Dict[str, Any]]
    ) -> None:
        """
        Add tool call results to memory.
        
        Args:
            session_id: Session ID
            tool_results: Tool call results
        """
        # Get session
        session = self.sessions[session_id]
        
        # Add tool results
        timestamp = time.time()
        
        for result in tool_results:
            tool_entry = {
                "timestamp": timestamp,
                "name": result.get("name", "unknown_tool"),
                "result": result.get("result"),
                "metadata": result.get("metadata", {})
            }
            
            session["tool_results"].append(tool_entry)
    
    def add_plan_execution(
        self,
        session_id: str,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]],
        final_response: Dict[str, Any]
    ) -> None:
        """
        Add plan execution record to memory.
        
        Args:
            session_id: Session ID
            plan: Plan
            results: Execution results
            final_response: Final response
        """
        # Get session
        session = self.sessions[session_id]
        
        # Add plan execution record
        timestamp = time.time()
        
        plan_entry = {
            "timestamp": timestamp,
            "plan": plan,
            "results": results,
            "final_response": final_response
        }
        
        session["plan_history"].append(plan_entry)
    
    def add_knowledge(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> None:
        """
        Add knowledge to cache.
        
        Args:
            session_id: Session ID
            key: Knowledge key
            value: Knowledge value
        """
        # Get session
        session = self.sessions[session_id]
        
        # Add knowledge
        session["knowledge_cache"][key] = value
    
    def get_context(
        self,
        session_id: str,
        include_tool_results: bool = True,
        include_plan_history: bool = False
    ) -> Dict[str, Any]:
        """
        Get session context.
        
        Args:
            session_id: Session ID
            include_tool_results: Whether to include tool call results
            include_plan_history: Whether to include plan history
            
        Returns:
            Session context
        """
        # Get session
        session = self.sessions[session_id]
        
        # Build context
        context = {
            "conversation_history": session["conversation_history"],
            "topic_history": session["topic_history"],
            "knowledge_cache": session["knowledge_cache"],
            "metadata": session["metadata"]
        }
        
        # Add tool call results (if needed)
        if include_tool_results:
            context["tool_results"] = session["tool_results"]
        
        # Add plan history (if needed)
        if include_plan_history:
            context["plan_history"] = session["plan_history"]
        
        return context
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            session_id: Session ID
            limit: Number of records to return
            
        Returns:
            Conversation history
        """
        # Get session
        session = self.sessions[session_id]
        
        # Get conversation history
        history = session["conversation_history"]
        
        # Limit number of records
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def get_topic_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get topic history.
        
        Args:
            session_id: Session ID
            limit: Number of records to return
            
        Returns:
            Topic history
        """
        # Get session
        session = self.sessions[session_id]
        
        # Get topic history
        history = session["topic_history"]
        
        # Limit number of records
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def get_last_topic_dist(
        self,
        session_id: str
    ) -> Optional[np.ndarray]:
        """
        Get last topic distribution.
        
        Args:
            session_id: Session ID
            
        Returns:
            Topic distribution
        """
        # Get session
        session = self.sessions[session_id]
        
        # Get last topic distribution
        last_topic_dist = session["metadata"].get("last_topic_dist")
        
        if last_topic_dist is not None:
            return np.array(last_topic_dist)
        
        return None
    
    def get_knowledge(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get knowledge cache.
        
        Args:
            session_id: Session ID
            key: Knowledge key
            default: Default value
            
        Returns:
            Knowledge value
        """
        # Get session
        session = self.sessions[session_id]
        
        # Get knowledge
        return session["knowledge_cache"].get(key, default)
    
    def clear_session(
        self,
        session_id: str
    ) -> None:
        """
        Clear session.
        
        Args:
            session_id: Session ID
        """
        # Clear session
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def save(
        self,
        path: str,
        session_id: Optional[str] = None
    ) -> None:
        """
        Save memory to file.
        
        Args:
            path: Save path
            session_id: Session ID (if None, save all sessions)
        """
        # Create directory
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare data
        if session_id is not None:
            # Save single session
            if session_id in self.sessions:
                data = {session_id: self.sessions[session_id]}
            else:
                data = {}
        else:
            # Save all sessions
            data = dict(self.sessions)
        
        # Save to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(
        cls,
        path: str,
        dev_mode: bool = False
    ) -> 'MemorySystem':
        """
        Load memory from file.
        
        Args:
            path: Load path
            dev_mode: Whether to enable development mode
            
        Returns:
            Memory system instance
        """
        # Create instance
        instance = cls(dev_mode=dev_mode)
        
        # Load data
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Set sessions
        for session_id, session_data in data.items():
            instance.sessions[session_id] = session_data
        
        return instance


# Test code
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test memory system")
    parser.add_argument("--dev_mode", action="store_true", help="Development mode")
    
    args = parser.parse_args()
    
    # Initialize memory system
    memory = MemorySystem(dev_mode=args.dev_mode)
    
    # Test adding conversation record
    session_id = "test_session"
    
    # Simulate topic distribution
    topic_dist = np.array([0.1, 0.2, 0.5, 0.1, 0.1])
    
    # Add conversation record
    memory.add(
        session_id=session_id,
        user_input="Hello, I want to learn about climate change.",
        response="Climate change refers to long-term changes in the Earth's climate system, including temperature, precipitation, and wind patterns.",
        topic_dist=topic_dist,
        metadata={"dominant_topics": [(2, 0.5)]}
    )
    
    # Add knowledge
    memory.add_knowledge(
        session_id=session_id,
        key="user_interests",
        value=["climate change", "renewable energy"]
    )
    
    # Get context
    context = memory.get_context(session_id)
    
    print("Context:")
    print(f"  Conversation history: {len(context['conversation_history'])} entries")
    print(f"  Topic history: {len(context['topic_history'])} entries")
    print(f"  Knowledge cache: {context['knowledge_cache']}")
    
    # Get last topic distribution
    last_topic_dist = memory.get_last_topic_dist(session_id)
    print(f"Last topic distribution: {last_topic_dist}")
    
    # Save memory
    memory.save("test_memory.json", session_id)
    print("Memory saved to test_memory.json")

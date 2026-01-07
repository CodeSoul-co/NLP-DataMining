"""
Topic-Aware Agent

Integrates topic-aware module, cognitive control module, knowledge representation module,
and memory system to implement an intelligent Agent based on ETM.
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path

# Add project root directory to path
sys.path.append(str(Path(__file__).parents[2]))

# Import other modules
from agent.modules.topic_aware import TopicAwareModule
from agent.modules.cognitive_controller import CognitiveController
from agent.modules.knowledge_module import KnowledgeModule
from agent.memory.memory_system import MemorySystem
from agent.utils.llm_client import LLMClient
from agent.utils.tool_registry import ToolRegistry
from agent.utils.config import AgentConfig

from engine_c.etm import ETM
from embedding.embedder import QwenEmbedder


class TopicAwareAgent:
    """
    Topic-Aware Agent that integrates ETM model with large language models for intelligent interaction.
    
    Architecture:
    1. Topic-aware module: Maps text to topic space
    2. Cognitive control module: Integrates topic information with large language models
    3. Knowledge representation module: Stores and retrieves knowledge
    4. Memory system: Manages conversation history and topic evolution
    """
    
    def __init__(
        self,
        config: AgentConfig,
        dev_mode: bool = False
    ):
        """
        Initialize Topic-Aware Agent.
        
        Args:
            config: Agent configuration
            dev_mode: Whether to enable development mode (print debug information)
        """
        self.config = config
        self.dev_mode = dev_mode
        
        if self.dev_mode:
            print(f"[TopicAwareAgent] Initializing with config: {config}")
        
        # Set device
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        if self.dev_mode:
            print(f"[TopicAwareAgent] Using device: {self.device}")
        
        # Initialize components
        self.etm = self._load_etm_model()
        self.embedder = self._init_embedder()
        self.topic_module = self._init_topic_module()
        self.memory = self._init_memory_system()
        self.knowledge_module = self._init_knowledge_module()
        self.llm_client = self._init_llm_client()
        self.tool_registry = self._init_tool_registry()
        self.cognitive_controller = self._init_cognitive_controller()
        
        if self.dev_mode:
            print(f"[TopicAwareAgent] Initialized successfully")
    
    def _load_etm_model(self) -> ETM:
        """Load ETM model"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Loading ETM model from {self.config.etm_model_path}")
            
            etm = ETM.load_model(self.config.etm_model_path, self.device)
            etm.eval()  # Set to evaluation mode
            return etm
        except Exception as e:
            raise RuntimeError(f"Failed to load ETM model: {e}")
    
    def _init_embedder(self) -> QwenEmbedder:
        """Initialize Qwen embedding model"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing embedder with model: {self.config.embedding_model_path}")
            
            return QwenEmbedder(
                model_path=self.config.embedding_model_path,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedder: {e}")
    
    def _init_topic_module(self) -> TopicAwareModule:
        """Initialize topic-aware module"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing topic module")
            
            return TopicAwareModule(
                etm_model_path=self.config.etm_model_path,
                vocab_path=self.config.vocab_path,
                embedding_model_path=self.config.embedding_model_path,
                device=self.device,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize topic module: {e}")
    
    def _init_memory_system(self) -> MemorySystem:
        """Initialize memory system"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing memory system")
            
            return MemorySystem(
                max_history_length=self.config.max_history_length,
                max_topic_history_length=self.config.max_topic_history_length,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize memory system: {e}")
    
    def _init_knowledge_module(self) -> KnowledgeModule:
        """Initialize knowledge representation module"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing knowledge module")
            
            return KnowledgeModule(
                topic_module=self.topic_module,
                embedder=self.embedder,
                vector_dim=self.config.embedding_dim,
                use_faiss=self.config.use_faiss,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize knowledge module: {e}")
    
    def _init_llm_client(self) -> LLMClient:
        """Initialize large language model client"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing LLM client")
            
            return LLMClient(
                model_name=self.config.llm_model_name,
                api_key=self.config.llm_api_key,
                api_base=self.config.llm_api_base,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM client: {e}")
    
    def _init_tool_registry(self) -> ToolRegistry:
        """Initialize tool registry"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing tool registry")
            
            registry = ToolRegistry()
            
            # Register default tools
            if self.config.register_default_tools:
                registry.register_default_tools()
            
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tool registry: {e}")
    
    def _init_cognitive_controller(self) -> CognitiveController:
        """Initialize cognitive control module"""
        try:
            if self.dev_mode:
                print(f"[TopicAwareAgent] Initializing cognitive controller")
            
            return CognitiveController(
                topic_module=self.topic_module,
                memory_system=self.memory,
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                dev_mode=self.dev_mode
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize cognitive controller: {e}")
    
    def process(
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
        return self.cognitive_controller.process_input(
            user_input=user_input,
            session_id=session_id,
            use_tools=use_tools,
            use_topic_enhancement=use_topic_enhancement
        )
    
    def add_document(
        self,
        document: Dict[str, Any],
        embedding: Optional[np.ndarray] = None,
        topic_dist: Optional[np.ndarray] = None
    ) -> int:
        """
        Add document to knowledge base.
        
        Args:
            document: Document content
            embedding: Document embedding vector (optional)
            topic_dist: Document topic distribution (optional)
            
        Returns:
            Document ID
        """
        return self.knowledge_module.add_document(document, embedding, topic_dist)
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[np.ndarray]] = None,
        topic_dists: Optional[List[np.ndarray]] = None
    ) -> List[int]:
        """
        Batch add documents to knowledge base.
        
        Args:
            documents: List of documents
            embeddings: List of embedding vectors (optional)
            topic_dists: List of topic distributions (optional)
            
        Returns:
            List of document IDs
        """
        return self.knowledge_module.add_documents(documents, embeddings, topic_dists)
    
    def query_knowledge(
        self,
        query_text: str,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Query knowledge base.
        
        Args:
            query_text: Query text
            top_k: Number of documents to return
            use_hybrid: Whether to use hybrid query
            
        Returns:
            List of document ID, similarity, and document content
        """
        if use_hybrid:
            return self.knowledge_module.hybrid_query(query_text, top_k=top_k)
        else:
            return self.knowledge_module.query_by_text(query_text, top_k=top_k)
    
    def get_topic_distribution(
        self,
        text: str
    ) -> np.ndarray:
        """
        Get topic distribution for text.
        
        Args:
            text: Input text
            
        Returns:
            Topic distribution vector
        """
        return self.topic_module.get_topic_distribution(text)
    
    def get_dominant_topics(
        self,
        topic_dist: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Get dominant topics.
        
        Args:
            topic_dist: Topic distribution vector
            top_k: Number of topics to return
            
        Returns:
            List of topic index and weight
        """
        return self.topic_module.get_dominant_topics(topic_dist, top_k)
    
    def get_topic_words(
        self,
        topic_idx: int,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get keywords for a topic.
        
        Args:
            topic_idx: Topic index
            top_k: Number of keywords to return
            
        Returns:
            List of keyword and weight
        """
        return self.topic_module.get_topic_words(topic_idx, top_k)
    
    def register_tool(
        self,
        name: str,
        func: callable,
        description: str
    ) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
        """
        self.tool_registry.register_tool(name, func, description)
    
    def save(
        self,
        path: str,
        save_knowledge: bool = True,
        save_memory: bool = True
    ) -> None:
        """
        Save Agent state to file.
        
        Args:
            path: Save path
            save_knowledge: Whether to save knowledge base
            save_memory: Whether to save memory system
        """
        # Create directory
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save knowledge base
        if save_knowledge:
            knowledge_path = os.path.join(path, "knowledge.json")
            self.knowledge_module.save(knowledge_path)
        
        # Save memory system
        if save_memory:
            memory_path = os.path.join(path, "memory.json")
            self.memory.save(memory_path)
    
    @classmethod
    def load(
        cls,
        path: str,
        dev_mode: bool = False
    ) -> 'TopicAwareAgent':
        """
        Load Agent state from file.
        
        Args:
            path: Load path
            dev_mode: Whether to enable development mode
            
        Returns:
            Agent instance
        """
        # Load configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = AgentConfig.from_dict(config_dict)
        
        # Create instance
        instance = cls(config, dev_mode)
        
        # Load knowledge base
        knowledge_path = os.path.join(path, "knowledge.json")
        if os.path.exists(knowledge_path):
            instance.knowledge_module = KnowledgeModule.load(
                path=knowledge_path,
                topic_module=instance.topic_module,
                embedder=instance.embedder,
                use_faiss=config.use_faiss,
                dev_mode=dev_mode
            )
        
        # Load memory system
        memory_path = os.path.join(path, "memory.json")
        if os.path.exists(memory_path):
            instance.memory = MemorySystem.load(
                path=memory_path,
                dev_mode=dev_mode
            )
            
            # Re-initialize cognitive control module
            instance.cognitive_controller = CognitiveController(
                topic_module=instance.topic_module,
                memory_system=instance.memory,
                llm_client=instance.llm_client,
                tool_registry=instance.tool_registry,
                dev_mode=dev_mode
            )
        
        return instance


# Test code
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Topic-Aware Agent")
    parser.add_argument("--etm_model", type=str, required=True, help="ETM model path")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary path")
    parser.add_argument("--input", type=str, required=True, help="Test input")
    parser.add_argument("--dev_mode", action="store_true", help="Development mode")
    
    args = parser.parse_args()
    
    # Create configuration
    from agent.utils.config import AgentConfig
    
    config = AgentConfig(
        etm_model_path=args.etm_model,
        vocab_path=args.vocab,
        embedding_model_path="/root/autodl-tmp/qwen3_embedding_0.6B",
        llm_model_name="gpt-3.5-turbo"  # Example, replace with actual model in production
    )
    
    # Initialize Agent
    agent = TopicAwareAgent(config, dev_mode=args.dev_mode)
    
    # Process input
    session_id = "test_session"
    result = agent.process(args.input, session_id)
    
    print(f"Response: {result['content']}")
    print(f"Dominant topics: {result['dominant_topics']}")
    print(f"Topic shifted: {result['topic_shifted']}")

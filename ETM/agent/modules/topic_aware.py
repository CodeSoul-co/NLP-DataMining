"""
Topic-Aware Module

Leverages ETM's topic modeling capabilities to implement topic identification, tracking, and expansion.
This module is a core component of the Agent, responsible for mapping user input to topic space.
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

from engine_c.etm import ETM
from embedding.embedder import QwenEmbedder


class TopicAwareModule:
    """
    Topic-aware module that uses ETM model to map text to topic space.
    
    Features:
    1. Topic identification: Map user input to topic space (theta)
    2. Topic tracking: Track conversation topic changes
    3. Topic expansion: Expand related knowledge based on topic similarity
    4. Topic filtering: Filter information based on topic relevance
    """
    
    def __init__(
        self,
        etm_model_path: str,
        vocab_path: str,
        embedding_model_path: str = "/root/autodl-tmp/qwen3_embedding_0.6B",
        device: str = None,
        threshold: float = 0.1,  # Topic significance threshold
        dev_mode: bool = False
    ):
        """
        Initialize topic-aware module.
        
        Args:
            etm_model_path: ETM model path
            vocab_path: Vocabulary path
            embedding_model_path: Qwen embedding model path
            device: Device ('cuda', 'cpu', or None for auto-selection)
            threshold: Topic significance threshold
            dev_mode: Whether to enable development mode (print debug information)
        """
        self.dev_mode = dev_mode
        self.threshold = threshold
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if self.dev_mode:
            print(f"[TopicAwareModule] Using device: {self.device}")
            print(f"[TopicAwareModule] Loading ETM model from {etm_model_path}")
        
        # Load ETM model
        self.etm = self._load_etm_model(etm_model_path)
        
        # Load vocabulary
        self.vocab = self._load_vocab(vocab_path)
        
        # Initialize Qwen embedding model
        self.embedder = self._init_embedder(embedding_model_path)
        
        if self.dev_mode:
            print(f"[TopicAwareModule] Initialized successfully")
            print(f"[TopicAwareModule] Vocabulary size: {len(self.vocab)}")
            print(f"[TopicAwareModule] Number of topics: {self.etm.num_topics}")
    
    def _load_etm_model(self, model_path: str) -> ETM:
        """Load ETM model"""
        try:
            etm = ETM.load_model(model_path, self.device)
            etm.eval()  # Set to evaluation mode
            return etm
        except Exception as e:
            raise RuntimeError(f"Failed to load ETM model: {e}")
    
    def _load_vocab(self, vocab_path: str) -> List[str]:
        """Load vocabulary"""
        try:
            if vocab_path.endswith('_list.json'):
                # Direct list format
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_list = json.load(f)
                return vocab_list
            else:
                # word2idx format
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    word2idx = json.load(f)
                
                # Convert to ordered list
                vocab_size = len(word2idx)
                vocab_list = [''] * vocab_size
                for word, idx in word2idx.items():
                    vocab_list[int(idx)] = word
                
                return vocab_list
        except Exception as e:
            raise RuntimeError(f"Failed to load vocabulary: {e}")
    
    def _init_embedder(self, model_path: str) -> QwenEmbedder:
        """Initialize Qwen embedding model"""
        try:
            # Assuming QwenEmbedder is already implemented
            # If not, can be implemented directly using transformers library
            return QwenEmbedder(
                model_path=model_path,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedder: {e}")
    
    def get_topic_distribution(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Get topic distribution for text.
        
        Args:
            text: Input text
            normalize: Whether to normalize topic distribution
            
        Returns:
            Topic distribution vector (num_topics,)
        """
        # Get Qwen embedding for text
        embedding = self.embedder.embed_text(text)
        
        # Convert to tensor and add batch dimension
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Use ETM encoder to get topic distribution
        with torch.no_grad():
            theta = self.etm.get_theta(embedding_tensor)
        
        # Convert to numpy array
        theta_np = theta.squeeze().cpu().numpy()
        
        return theta_np
    
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
            List of topic index and weight [(topic_idx, weight), ...]
        """
        # Get top k topics
        top_indices = np.argsort(-topic_dist)[:top_k]
        
        # Filter out topics with weight below threshold
        dominant_topics = [
            (idx, topic_dist[idx]) 
            for idx in top_indices 
            if topic_dist[idx] > self.threshold
        ]
        
        return dominant_topics
    
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
            List of keyword and weight [(word, weight), ...]
        """
        # Use ETM to get topic words
        topic_words = self.etm.get_topic_words(top_k=top_k, vocab=self.vocab)
        
        # Return words for specified topic
        return topic_words[topic_idx][1]
    
    def get_topic_similarity(
        self,
        topic_dist1: np.ndarray,
        topic_dist2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two topic distributions.
        
        Args:
            topic_dist1: First topic distribution
            topic_dist2: Second topic distribution
            
        Returns:
            Cosine similarity (0-1)
        """
        # Calculate cosine similarity
        dot_product = np.dot(topic_dist1, topic_dist2)
        norm1 = np.linalg.norm(topic_dist1)
        norm2 = np.linalg.norm(topic_dist2)
        
        # Avoid division by zero
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def detect_topic_shift(
        self,
        prev_topic_dist: np.ndarray,
        curr_topic_dist: np.ndarray,
        threshold: float = 0.7
    ) -> bool:
        """
        Detect whether topic has changed significantly.
        
        Args:
            prev_topic_dist: Previous topic distribution
            curr_topic_dist: Current topic distribution
            threshold: Similarity threshold, below which is considered topic change
            
        Returns:
            Whether topic has changed
        """
        similarity = self.get_topic_similarity(prev_topic_dist, curr_topic_dist)
        return similarity < threshold
    
    def get_topic_context(
        self,
        topic_indices: List[int],
        words_per_topic: int = 5
    ) -> Dict[str, Any]:
        """
        Get topic context information for prompt enhancement.
        
        Args:
            topic_indices: List of topic indices
            words_per_topic: Number of keywords to return per topic
            
        Returns:
            Topic context information
        """
        context = {
            "topics": []
        }
        
        for topic_idx in topic_indices:
            topic_words = self.get_topic_words(topic_idx, top_k=words_per_topic)
            
            context["topics"].append({
                "id": topic_idx,
                "keywords": [word for word, _ in topic_words],
                "weights": [float(weight) for _, weight in topic_words]
            })
        
        return context
    
    def enrich_prompt(
        self,
        prompt: str,
        topic_dist: np.ndarray,
        top_k_topics: int = 2,
        words_per_topic: int = 5
    ) -> str:
        """
        Enhance prompt using topic information.
        
        Args:
            prompt: Original prompt
            topic_dist: Topic distribution
            top_k_topics: Number of topics to use
            words_per_topic: Number of keywords per topic
            
        Returns:
            Enhanced prompt
        """
        # Get dominant topics
        dominant_topics = self.get_dominant_topics(topic_dist, top_k=top_k_topics)
        
        if not dominant_topics:
            return prompt
        
        # Build topic context
        topic_context = "Relevant topic context:\n"
        
        for topic_idx, weight in dominant_topics:
            topic_words = self.get_topic_words(topic_idx, top_k=words_per_topic)
            words_str = ", ".join([word for word, _ in topic_words])
            topic_context += f"- Topic {topic_idx} (weight: {weight:.2f}): {words_str}\n"
        
        # Enhance prompt
        enhanced_prompt = f"{prompt}\n\n{topic_context}"
        
        return enhanced_prompt


# Test code
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test topic-aware module")
    parser.add_argument("--etm_model", type=str, required=True, help="ETM model path")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary path")
    parser.add_argument("--text", type=str, required=True, help="Test text")
    parser.add_argument("--dev_mode", action="store_true", help="Development mode")
    
    args = parser.parse_args()
    
    # Initialize module
    topic_module = TopicAwareModule(
        etm_model_path=args.etm_model,
        vocab_path=args.vocab,
        dev_mode=args.dev_mode
    )
    
    # Get topic distribution
    topic_dist = topic_module.get_topic_distribution(args.text)
    print(f"Topic distribution: {topic_dist}")
    
    # Get dominant topics
    dominant_topics = topic_module.get_dominant_topics(topic_dist)
    print(f"Dominant topics: {dominant_topics}")
    
    # Get topic words
    for topic_idx, weight in dominant_topics:
        topic_words = topic_module.get_topic_words(topic_idx)
        print(f"Topic {topic_idx} (weight: {weight:.2f}): {topic_words}")
    
    # Enhance prompt
    enhanced_prompt = topic_module.enrich_prompt(args.text, topic_dist)
    print(f"\nEnhanced prompt:\n{enhanced_prompt}")

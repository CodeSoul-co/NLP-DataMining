"""
Topic Model Evaluation Metrics

Includes:
- Topic Coherence (NPMI-based)
- Topic Diversity
- Perplexity
"""

import numpy as np
from scipy import sparse
from typing import List, Dict, Tuple, Optional
from collections import Counter


def compute_topic_coherence(
    beta: np.ndarray,
    bow_matrix: sparse.csr_matrix,
    vocab: List[str],
    top_k: int = 10,
    method: str = 'npmi'
) -> Tuple[float, List[float]]:
    """
    Compute topic coherence using NPMI (Normalized Pointwise Mutual Information).
    
    Args:
        beta: Topic-word distribution, shape (K, V)
        bow_matrix: BOW matrix for computing co-occurrence, shape (N, V)
        vocab: Vocabulary list
        top_k: Number of top words per topic
        method: Coherence method ('npmi' or 'pmi')
        
    Returns:
        (average_coherence, per_topic_coherence)
    """
    num_topics = beta.shape[0]
    num_docs = bow_matrix.shape[0]
    
    # Convert BOW to binary (word presence)
    binary_bow = (bow_matrix > 0).astype(np.float32)
    
    # Document frequency for each word
    doc_freq = np.array(binary_bow.sum(axis=0)).flatten()
    
    # Compute co-occurrence matrix (V x V)
    # co_occur[i,j] = number of docs containing both word i and word j
    co_occur = binary_bow.T @ binary_bow
    if sparse.issparse(co_occur):
        co_occur = co_occur.toarray()
    
    topic_coherences = []
    
    for k in range(num_topics):
        # Get top words for this topic
        top_indices = np.argsort(beta[k])[-top_k:][::-1]
        
        coherence = 0.0
        count = 0
        
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                w1, w2 = top_indices[i], top_indices[j]
                
                # Document frequencies
                d_w1 = doc_freq[w1]
                d_w2 = doc_freq[w2]
                d_w1_w2 = co_occur[w1, w2]
                
                if d_w1 > 0 and d_w2 > 0 and d_w1_w2 > 0:
                    # PMI = log(P(w1,w2) / (P(w1) * P(w2)))
                    #     = log((d_w1_w2 / N) / ((d_w1/N) * (d_w2/N)))
                    #     = log(d_w1_w2 * N / (d_w1 * d_w2))
                    pmi = np.log((d_w1_w2 * num_docs) / (d_w1 * d_w2) + 1e-10)
                    
                    if method == 'npmi':
                        # NPMI = PMI / -log(P(w1,w2))
                        #      = PMI / -log(d_w1_w2 / N)
                        npmi = pmi / (-np.log(d_w1_w2 / num_docs + 1e-10) + 1e-10)
                        coherence += npmi
                    else:
                        coherence += pmi
                    
                    count += 1
        
        if count > 0:
            coherence /= count
        
        topic_coherences.append(coherence)
    
    avg_coherence = np.mean(topic_coherences)
    
    return avg_coherence, topic_coherences


def compute_topic_diversity(
    beta: np.ndarray,
    top_k: int = 25
) -> float:
    """
    Compute topic diversity as the percentage of unique words in top-k words across all topics.
    
    Higher diversity = less redundant topics.
    
    Args:
        beta: Topic-word distribution, shape (K, V)
        top_k: Number of top words per topic
        
    Returns:
        Diversity score (0 to 1)
    """
    num_topics = beta.shape[0]
    
    all_words = set()
    total_words = 0
    
    for k in range(num_topics):
        top_indices = np.argsort(beta[k])[-top_k:]
        all_words.update(top_indices.tolist())
        total_words += top_k
    
    diversity = len(all_words) / total_words
    
    return diversity


class TopicMetrics:
    """
    Comprehensive topic model evaluation.
    
    Computes and stores all evaluation metrics.
    """
    
    def __init__(
        self,
        beta: np.ndarray,
        theta: np.ndarray,
        bow_matrix: sparse.csr_matrix,
        vocab: List[str],
        dev_mode: bool = False
    ):
        """
        Initialize metrics calculator.
        
        Args:
            beta: Topic-word distribution (K, V)
            theta: Document-topic distribution (N, K)
            bow_matrix: BOW matrix (N, V)
            vocab: Vocabulary list
            dev_mode: Print debug information
        """
        self.beta = beta
        self.theta = theta
        self.bow_matrix = bow_matrix
        self.vocab = vocab
        self.dev_mode = dev_mode
        
        self.num_topics = beta.shape[0]
        self.vocab_size = beta.shape[1]
        self.num_docs = theta.shape[0]
        
        # Cached metrics
        self._coherence = None
        self._diversity = None
        self._topic_sizes = None
    
    def compute_all(self, top_k: int = 10) -> Dict:
        """
        Compute all metrics.
        
        Args:
            top_k: Number of top words for coherence/diversity
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Topic coherence
        avg_coh, per_topic_coh = compute_topic_coherence(
            self.beta, self.bow_matrix, self.vocab, top_k
        )
        metrics['coherence'] = avg_coh
        metrics['coherence_per_topic'] = per_topic_coh
        self._coherence = avg_coh
        
        # Topic diversity
        diversity = compute_topic_diversity(self.beta, top_k)
        metrics['diversity'] = diversity
        self._diversity = diversity
        
        # Topic sizes (average theta per topic)
        topic_sizes = self.theta.mean(axis=0)
        metrics['topic_sizes'] = topic_sizes.tolist()
        self._topic_sizes = topic_sizes
        
        # Topic entropy (how spread out the topic distribution is)
        theta_entropy = -np.sum(self.theta * np.log(self.theta + 1e-10), axis=1)
        metrics['avg_doc_entropy'] = float(np.mean(theta_entropy))
        
        # Topic concentration (how concentrated topics are)
        max_theta = np.max(self.theta, axis=1)
        metrics['avg_topic_concentration'] = float(np.mean(max_theta))
        
        if self.dev_mode:
            print(f"[DEV] Metrics computed:")
            print(f"[DEV]   Coherence: {avg_coh:.4f}")
            print(f"[DEV]   Diversity: {diversity:.4f}")
            print(f"[DEV]   Avg doc entropy: {metrics['avg_doc_entropy']:.4f}")
        
        return metrics
    
    def get_top_words(self, top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get top words for each topic.
        
        Args:
            top_k: Number of top words
            
        Returns:
            List of [(word, prob), ...] for each topic
        """
        topics = []
        for k in range(self.num_topics):
            top_indices = np.argsort(self.beta[k])[-top_k:][::-1]
            words = [(self.vocab[idx], float(self.beta[k, idx])) for idx in top_indices]
            topics.append(words)
        return topics
    
    def get_topic_summary(self, top_k: int = 10) -> str:
        """
        Get formatted topic summary.
        
        Args:
            top_k: Number of top words per topic
            
        Returns:
            Formatted string with topic descriptions
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TOPIC SUMMARY")
        lines.append("=" * 60)
        
        top_words = self.get_top_words(top_k)
        
        for k, words in enumerate(top_words):
            size = self._topic_sizes[k] if self._topic_sizes is not None else 0
            word_str = ", ".join([w for w, _ in words])
            lines.append(f"Topic {k:2d} (size={size:.3f}): {word_str}")
        
        lines.append("=" * 60)
        
        if self._coherence is not None:
            lines.append(f"Coherence: {self._coherence:.4f}")
        if self._diversity is not None:
            lines.append(f"Diversity: {self._diversity:.4f}")
        
        return "\n".join(lines)
    
    def save_results(self, output_path: str):
        """
        Save metrics and topic summary to file.
        
        Args:
            output_path: Output file path
        """
        import json
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        metrics = self.compute_all()
        top_words = self.get_top_words()
        
        results = {
            'metrics': {k: convert_to_serializable(v) for k, v in metrics.items() 
                       if not isinstance(v, np.ndarray) or len(v) <= 100},
            'topics': [
                {'id': k, 'words': [{'word': w, 'prob': float(p)} for w, p in words]}
                for k, words in enumerate(top_words)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved results to {output_path}")

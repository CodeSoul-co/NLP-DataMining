"""
Advanced Topic Model Evaluation Metrics

Includes:
- Topic Stability (across multiple runs)
- External Coherence (using gensim)
- Perplexity
- Topic Exclusivity
- pyLDAvis preparation
"""

import numpy as np
import scipy.sparse as sparse
from typing import List, Dict, Tuple, Optional, Union
import logging
from collections import Counter
import json

logger = logging.getLogger(__name__)


def compute_topic_exclusivity(
    beta: np.ndarray,
    top_k: int = 10
) -> Tuple[float, List[float]]:
    """
    Compute topic exclusivity - how exclusive top words are to each topic.
    
    Higher exclusivity means topics have more unique, non-overlapping words.
    
    Args:
        beta: Topic-word distribution (K x V)
        top_k: Number of top words per topic
        
    Returns:
        (average_exclusivity, per_topic_exclusivity)
    """
    num_topics = beta.shape[0]
    
    # Get top words for each topic
    top_words_per_topic = []
    for k in range(num_topics):
        top_indices = np.argsort(beta[k])[-top_k:]
        top_words_per_topic.append(set(top_indices))
    
    # Count word occurrences across topics
    word_topic_count = Counter()
    for words in top_words_per_topic:
        word_topic_count.update(words)
    
    # Compute exclusivity for each topic
    exclusivity_scores = []
    for k in range(num_topics):
        topic_words = top_words_per_topic[k]
        # Exclusivity = average of 1/count for each word
        exclusivity = np.mean([1.0 / word_topic_count[w] for w in topic_words])
        exclusivity_scores.append(exclusivity)
    
    avg_exclusivity = np.mean(exclusivity_scores)
    return avg_exclusivity, exclusivity_scores


def compute_topic_stability(
    beta_runs: List[np.ndarray],
    top_k: int = 10
) -> Tuple[float, np.ndarray]:
    """
    Compute topic stability across multiple training runs.
    
    Uses Jaccard similarity to measure how consistent topics are.
    
    Args:
        beta_runs: List of beta matrices from different runs
        top_k: Number of top words per topic
        
    Returns:
        (average_stability, stability_matrix)
    """
    if len(beta_runs) < 2:
        logger.warning("Need at least 2 runs to compute stability")
        return 1.0, np.array([[1.0]])
    
    num_runs = len(beta_runs)
    num_topics = beta_runs[0].shape[0]
    
    # Get top words for each topic in each run
    top_words_all_runs = []
    for beta in beta_runs:
        run_top_words = []
        for k in range(num_topics):
            top_indices = set(np.argsort(beta[k])[-top_k:])
            run_top_words.append(top_indices)
        top_words_all_runs.append(run_top_words)
    
    # Compute pairwise stability between runs
    stability_scores = []
    
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            # Find best matching topics between runs
            run_stability = []
            
            for k in range(num_topics):
                # Find most similar topic in other run
                best_jaccard = 0.0
                words_i = top_words_all_runs[i][k]
                
                for l in range(num_topics):
                    words_j = top_words_all_runs[j][l]
                    jaccard = len(words_i & words_j) / len(words_i | words_j)
                    best_jaccard = max(best_jaccard, jaccard)
                
                run_stability.append(best_jaccard)
            
            stability_scores.append(np.mean(run_stability))
    
    avg_stability = np.mean(stability_scores)
    return avg_stability, np.array(stability_scores)


def compute_perplexity(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix: Union[np.ndarray, sparse.csr_matrix]
) -> float:
    """
    Compute perplexity of the topic model.
    
    Perplexity = exp(-1/N * sum(log(p(w|d))))
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V)
        
    Returns:
        Perplexity score (lower is better)
    """
    if sparse.issparse(bow_matrix):
        bow_dense = bow_matrix.toarray()
    else:
        bow_dense = bow_matrix
    
    # Normalize BOW to get word distributions
    doc_lengths = bow_dense.sum(axis=1, keepdims=True)
    doc_lengths[doc_lengths == 0] = 1  # Avoid division by zero
    
    # Compute word probabilities: p(w|d) = sum_k theta[d,k] * beta[k,w]
    word_probs = theta @ beta  # (N x V)
    
    # Add small constant to avoid log(0)
    word_probs = np.clip(word_probs, 1e-10, 1.0)
    
    # Compute log likelihood
    log_likelihood = np.sum(bow_dense * np.log(word_probs))
    total_words = bow_dense.sum()
    
    # Perplexity
    perplexity = np.exp(-log_likelihood / total_words)
    
    return float(perplexity)


def compute_topic_quality_score(
    coherence: float,
    diversity: float,
    exclusivity: float,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> float:
    """
    Compute overall topic quality score.
    
    Combines coherence, diversity, and exclusivity into a single score.
    
    Args:
        coherence: Topic coherence score (higher is better)
        diversity: Topic diversity score (0-1, higher is better)
        exclusivity: Topic exclusivity score (0-1, higher is better)
        weights: Weights for (coherence, diversity, exclusivity)
        
    Returns:
        Combined quality score
    """
    # Normalize coherence to 0-1 range (assuming NPMI ranges from -1 to 1)
    coherence_norm = (coherence + 1) / 2
    coherence_norm = np.clip(coherence_norm, 0, 1)
    
    w_coh, w_div, w_exc = weights
    quality = w_coh * coherence_norm + w_div * diversity + w_exc * exclusivity
    
    return float(quality)


def prepare_pyldavis_data(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix: Union[np.ndarray, sparse.csr_matrix],
    vocab: List[str]
) -> Dict:
    """
    Prepare data for pyLDAvis visualization.
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V)
        vocab: Vocabulary list
        
    Returns:
        Dictionary with pyLDAvis-compatible data
    """
    if sparse.issparse(bow_matrix):
        bow_dense = bow_matrix.toarray()
    else:
        bow_dense = bow_matrix
    
    # Topic-term distribution (already have as beta)
    topic_term_dists = beta
    
    # Document-topic distribution (already have as theta)
    doc_topic_dists = theta
    
    # Document lengths
    doc_lengths = bow_dense.sum(axis=1)
    
    # Term frequency across corpus
    term_frequency = bow_dense.sum(axis=0)
    
    # Topic weights (proportion of corpus assigned to each topic)
    topic_weights = theta.mean(axis=0)
    
    return {
        'topic_term_dists': topic_term_dists.tolist(),
        'doc_topic_dists': doc_topic_dists.tolist(),
        'doc_lengths': doc_lengths.tolist(),
        'vocab': vocab,
        'term_frequency': term_frequency.tolist(),
        'topic_weights': topic_weights.tolist()
    }


def generate_pyldavis_html(
    theta: np.ndarray,
    beta: np.ndarray,
    bow_matrix: Union[np.ndarray, sparse.csr_matrix],
    vocab: List[str],
    output_path: str
) -> str:
    """
    Generate pyLDAvis HTML visualization.
    
    Args:
        theta: Document-topic distribution (N x K)
        beta: Topic-word distribution (K x V)
        bow_matrix: BOW matrix (N x V)
        vocab: Vocabulary list
        output_path: Path to save HTML file
        
    Returns:
        Path to saved HTML file
    """
    try:
        import pyLDAvis
    except ImportError:
        logger.warning("pyLDAvis not installed. Install with: pip install pyLDAvis")
        return None
    
    if sparse.issparse(bow_matrix):
        bow_dense = bow_matrix.toarray()
    else:
        bow_dense = bow_matrix
    
    # Prepare data
    doc_lengths = bow_dense.sum(axis=1)
    term_frequency = bow_dense.sum(axis=0)
    
    # Create pyLDAvis data
    vis_data = pyLDAvis.prepare(
        topic_term_dists=beta,
        doc_topic_dists=theta,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency,
        sort_topics=False
    )
    
    # Save to HTML
    pyLDAvis.save_html(vis_data, output_path)
    logger.info(f"pyLDAvis visualization saved to {output_path}")
    
    return output_path


def compute_topic_word_scores(
    beta: np.ndarray,
    vocab: List[str],
    top_k: int = 20
) -> List[Dict]:
    """
    Compute detailed word scores for each topic.
    
    Args:
        beta: Topic-word distribution (K x V)
        vocab: Vocabulary list
        top_k: Number of top words per topic
        
    Returns:
        List of topic dictionaries with word scores
    """
    num_topics = beta.shape[0]
    topics = []
    
    for k in range(num_topics):
        top_indices = np.argsort(beta[k])[-top_k:][::-1]
        
        words = []
        for idx in top_indices:
            words.append({
                'word': vocab[idx],
                'probability': float(beta[k, idx]),
                'rank': len(words) + 1
            })
        
        topics.append({
            'topic_id': k,
            'words': words,
            'total_probability': float(sum(w['probability'] for w in words))
        })
    
    return topics


def compute_document_topic_summary(
    theta: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute summary statistics for document-topic distributions.
    
    Args:
        theta: Document-topic distribution (N x K)
        labels: Optional document labels
        
    Returns:
        Summary statistics dictionary
    """
    num_docs, num_topics = theta.shape
    
    # Dominant topic for each document
    dominant_topics = np.argmax(theta, axis=1)
    dominant_probs = np.max(theta, axis=1)
    
    # Topic distribution statistics
    topic_counts = np.bincount(dominant_topics, minlength=num_topics)
    topic_proportions = topic_counts / num_docs
    
    # Entropy of document distributions
    doc_entropy = -np.sum(theta * np.log(theta + 1e-10), axis=1)
    
    summary = {
        'num_documents': num_docs,
        'num_topics': num_topics,
        'topic_document_counts': topic_counts.tolist(),
        'topic_proportions': topic_proportions.tolist(),
        'avg_dominant_prob': float(np.mean(dominant_probs)),
        'std_dominant_prob': float(np.std(dominant_probs)),
        'avg_doc_entropy': float(np.mean(doc_entropy)),
        'std_doc_entropy': float(np.std(doc_entropy))
    }
    
    # Per-label statistics if labels provided
    if labels is not None:
        unique_labels = np.unique(labels)
        label_stats = {}
        
        for label in unique_labels:
            mask = labels == label
            label_theta = theta[mask]
            label_dominant = dominant_topics[mask]
            
            label_stats[str(label)] = {
                'count': int(mask.sum()),
                'dominant_topic_distribution': np.bincount(label_dominant, minlength=num_topics).tolist(),
                'avg_topic_distribution': label_theta.mean(axis=0).tolist()
            }
        
        summary['label_statistics'] = label_stats
    
    return summary


def compute_all_advanced_metrics(
    beta: np.ndarray,
    theta: np.ndarray,
    bow_matrix: Union[np.ndarray, sparse.csr_matrix],
    vocab: List[str],
    labels: Optional[np.ndarray] = None,
    top_k: int = 10
) -> Dict:
    """
    Compute all advanced metrics.
    
    Args:
        beta: Topic-word distribution (K x V)
        theta: Document-topic distribution (N x K)
        bow_matrix: BOW matrix (N x V)
        vocab: Vocabulary list
        labels: Optional document labels
        top_k: Number of top words for metrics
        
    Returns:
        Dictionary with all metrics
    """
    logger.info("Computing advanced metrics...")
    
    metrics = {}
    
    # Topic exclusivity
    avg_exc, per_topic_exc = compute_topic_exclusivity(beta, top_k)
    metrics['topic_exclusivity_avg'] = avg_exc
    metrics['topic_exclusivity_per_topic'] = per_topic_exc
    logger.info(f"  Topic Exclusivity: {avg_exc:.4f}")
    
    # Perplexity
    perplexity = compute_perplexity(theta, beta, bow_matrix)
    metrics['perplexity'] = perplexity
    logger.info(f"  Perplexity: {perplexity:.2f}")
    
    # Document-topic summary
    doc_summary = compute_document_topic_summary(theta, labels)
    metrics['document_topic_summary'] = doc_summary
    
    # Topic word scores
    topic_scores = compute_topic_word_scores(beta, vocab, top_k * 2)
    metrics['topic_word_scores'] = topic_scores
    
    return metrics


def save_evaluation_report(
    metrics: Dict,
    output_path: str,
    include_details: bool = True
):
    """
    Save comprehensive evaluation report.
    
    Args:
        metrics: Dictionary with all metrics
        output_path: Path to save report
        include_details: Whether to include detailed per-topic metrics
    """
    report = {
        'summary': {
            'topic_coherence': metrics.get('topic_coherence_avg', None),
            'topic_diversity_td': metrics.get('topic_diversity_td', None),
            'topic_diversity_irbo': metrics.get('topic_diversity_irbo', None),
            'topic_exclusivity': metrics.get('topic_exclusivity_avg', None),
            'perplexity': metrics.get('perplexity', None)
        }
    }
    
    if include_details:
        report['per_topic_coherence'] = metrics.get('topic_coherence_per_topic', [])
        report['per_topic_exclusivity'] = metrics.get('topic_exclusivity_per_topic', [])
        report['topic_significance'] = metrics.get('topic_significance', [])
        report['document_topic_summary'] = metrics.get('document_topic_summary', {})
        report['topic_word_scores'] = metrics.get('topic_word_scores', [])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation report saved to {output_path}")


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    
    num_topics = 10
    vocab_size = 1000
    num_docs = 500
    
    # Generate random beta (topic-word distribution)
    beta = np.random.dirichlet(np.ones(vocab_size) * 0.1, size=num_topics)
    
    # Generate random theta (document-topic distribution)
    theta = np.random.dirichlet(np.ones(num_topics) * 0.5, size=num_docs)
    
    # Generate random BOW matrix
    bow_matrix = sparse.random(num_docs, vocab_size, density=0.1, format='csr')
    bow_matrix.data = np.random.randint(1, 10, size=len(bow_matrix.data))
    
    # Generate random vocab
    vocab = [f"word_{i}" for i in range(vocab_size)]
    
    # Compute metrics
    metrics = compute_all_advanced_metrics(beta, theta, bow_matrix, vocab)
    
    print("\nAdvanced Metrics:")
    print(f"  Exclusivity: {metrics['topic_exclusivity_avg']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")

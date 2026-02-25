"""
Evaluation metrics for ETM

Includes:
- Topic Coherence (NPMI, PMI)
- Topic Diversity (TD, iRBO)
- Topic Exclusivity
- Topic Stability
- Perplexity
- pyLDAvis preparation
"""

from .metrics import TopicMetrics, compute_topic_coherence, compute_topic_diversity
from .topic_metrics import (
    compute_topic_coherence_npmi,
    compute_topic_diversity_td,
    compute_topic_diversity_inverted_rbo,
    compute_topic_significance,
    compute_all_metrics
)
from .unified_evaluator import UnifiedEvaluator, evaluate_model

__all__ = [
    'TopicMetrics', 
    'compute_topic_coherence', 
    'compute_topic_diversity',
    'compute_topic_coherence_npmi',
    'compute_topic_diversity_td',
    'compute_topic_diversity_inverted_rbo',
    'compute_topic_significance',
    'compute_all_metrics',
    'UnifiedEvaluator',
    'evaluate_model'
]

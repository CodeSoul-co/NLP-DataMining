"""
Visualization utilities for ETM.

Includes:
- Unified data loader for all visualizations
- Topic word visualization (bar charts, word clouds)
- Topic similarity heatmap
- Document-topic distribution (t-SNE, PCA, UMAP)
- Topic proportions
- Temporal topic analysis (document volume, topic evolution, Sankey diagrams)
- Topic embedding space visualization (ETM paper style)
- Document-topic UMAP (BERTopic style)
- Model evaluation and comparison
- pyLDAvis interactive visualization
"""

# Unified Data Loader (use this first!)
from .data_loader import (
    VisualizationDataLoader,
    load_etm_data
)

from .topic_visualizer import (
    TopicVisualizer,
    load_etm_results,
    visualize_etm_results,
    generate_pyldavis_visualization,
    generate_pyldavis_notebook
)
from .temporal_analysis import (
    TemporalTopicAnalyzer,
    analyze_temporal_topics
)

# Optional imports with graceful fallback
try:
    from .dimension_analysis import (
        DimensionAnalyzer,
        analyze_dimension_topics
    )
    _HAS_DIMENSION = True
except ImportError:
    _HAS_DIMENSION = False

try:
    from .topic_embedding_space import TopicEmbeddingSpaceVisualizer
    _HAS_EMBEDDING_SPACE = True
except ImportError:
    _HAS_EMBEDDING_SPACE = False

try:
    from .document_topic_umap import DocumentTopicUMAPVisualizer
    _HAS_DOC_UMAP = True
except ImportError:
    _HAS_DOC_UMAP = False

try:
    from .dynamic_topic_evolution import DynamicTopicEvolutionVisualizer
    _HAS_DYNAMIC = True
except ImportError:
    _HAS_DYNAMIC = False

__all__ = [
    # Data Loader (primary interface)
    'VisualizationDataLoader',
    'load_etm_data',
    # Topic Visualizer
    'TopicVisualizer',
    'load_etm_results',
    'visualize_etm_results',
    'generate_pyldavis_visualization',
    'generate_pyldavis_notebook',
    # Temporal Analysis
    'TemporalTopicAnalyzer',
    'analyze_temporal_topics',
]

# Add optional exports
if _HAS_DIMENSION:
    __all__.extend(['DimensionAnalyzer', 'analyze_dimension_topics'])
if _HAS_EMBEDDING_SPACE:
    __all__.append('TopicEmbeddingSpaceVisualizer')
if _HAS_DOC_UMAP:
    __all__.append('DocumentTopicUMAPVisualizer')
if _HAS_DYNAMIC:
    __all__.append('DynamicTopicEvolutionVisualizer')

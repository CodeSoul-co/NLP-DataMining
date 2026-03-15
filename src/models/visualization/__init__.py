"""
ETM Visualization Module
========================

Unified visualization toolkit for Embedded Topic Model (ETM) results.

Quick Start:
-----------
    # Method 1: One-line visualization (recommended)
    from visualization.run_visualization import run_all_visualizations
    run_all_visualizations(result_dir, dataset, mode, model_size='0.6B')
    
    # Method 2: Command line
    python -m visualization.run_visualization --result_dir /path/to/result --dataset socialTwitter --mode zero_shot

Core Components:
---------------
- run_visualization: Unified entry point for all visualizations
- visualization_generator: Core visualization generator class
- data_loader: Unified data loader for ETM results
- topic_visualizer: Topic-specific visualizations

Generated Charts:
----------------
Global Charts:
- topic_table.png: Topic identification table
- topic_network.png: Topic correlation network
- doc_clusters.png: Document clustering (t-SNE)
- clustering_heatmap.png: Hierarchical clustering heatmap
- topic_similarity.png: Topic similarity matrix
- topic_wordclouds.png: Word clouds for all topics
- training_history.png: Training loss curves
- metrics.png: Evaluation metrics

Per-Topic Charts:
- word_importance.png: Top words bar chart
- evolution.png: Topic evolution over time (if timestamps available)
"""

# Primary interface - use this for one-line visualization
from .run_visualization import (
    run_all_visualizations,
    load_visualization_data,
)

# Core visualization generator
from .visualization_generator import (
    VisualizationGenerator,
    load_model_data,
    load_complete_data,
)

# Unified Data Loader (optional)
try:
    from .data_loader import (
        VisualizationDataLoader,
        load_etm_data
    )
    _HAS_DATA_LOADER = True
except ImportError:
    _HAS_DATA_LOADER = False
    VisualizationDataLoader = None
    load_etm_data = None

# Topic Visualizer
from .topic_visualizer import (
    TopicVisualizer,
    load_etm_results,
    visualize_etm_results,
)

# Optional imports with graceful fallback
try:
    from .temporal_analysis import (
        TemporalTopicAnalyzer,
        analyze_temporal_topics
    )
    _HAS_TEMPORAL = True
except ImportError:
    _HAS_TEMPORAL = False

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

__all__ = [
    # Primary interface (recommended)
    'run_all_visualizations',
    'load_visualization_data',
    # Core generator
    'VisualizationGenerator',
    'load_model_data',
    'load_complete_data',
    # Data Loader
    'VisualizationDataLoader',
    'load_etm_data',
    # Topic Visualizer
    'TopicVisualizer',
    'load_etm_results',
    'visualize_etm_results',
]

# Add optional exports
if _HAS_TEMPORAL:
    __all__.extend(['TemporalTopicAnalyzer', 'analyze_temporal_topics'])
if _HAS_EMBEDDING_SPACE:
    __all__.append('TopicEmbeddingSpaceVisualizer')
if _HAS_DOC_UMAP:
    __all__.append('DocumentTopicUMAPVisualizer')

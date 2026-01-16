"""
Visualization utilities for ETM.

Includes:
- Topic word visualization (bar charts, word clouds)
- Topic similarity heatmap
- Document-topic distribution (t-SNE, PCA)
- Topic proportions
- Temporal topic analysis (document volume, topic evolution, Sankey diagrams)
- Dimension/spatial analysis (region-topic heatmaps)
- pyLDAvis interactive visualization
"""

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
from .dimension_analysis import (
    DimensionAnalyzer,
    analyze_dimension_topics
)

__all__ = [
    # Topic Visualizer
    'TopicVisualizer',
    'load_etm_results',
    'visualize_etm_results',
    'generate_pyldavis_visualization',
    'generate_pyldavis_notebook',
    # Temporal Analysis
    'TemporalTopicAnalyzer',
    'analyze_temporal_topics',
    # Dimension Analysis
    'DimensionAnalyzer',
    'analyze_dimension_topics'
]

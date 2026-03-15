"""
THETA Models Module
Contains all topic modeling algorithms: ETM, LDA, HDP, STM, BTM, CTM, DTM, etc.

Components:
- bow: Vocabulary and BOW generation
- model: Topic model implementations (ETM, baselines)
- data: Data loading utilities
- evaluation: Topic metrics
- visualization: Topic visualization
- data_pipeline: CSV scanning, mapping, matrix generation
"""

__version__ = "1.0.0"

from .config import *

"""
Data-related Schemas
Schemas for datasets, results, and visualizations
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DatasetInfo(BaseModel):
    """Information about a dataset"""
    name: str
    path: str
    size: Optional[int] = None  # Number of documents
    columns: Optional[List[str]] = None
    has_labels: bool = False
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "socialTwitter",
                "path": "/root/autodl-tmp/data/socialTwitter",
                "size": 10000,
                "has_labels": True,
                "language": "english"
            }
        }


class ResultInfo(BaseModel):
    """Information about a training result"""
    dataset: str
    mode: str
    timestamp: str
    path: str
    
    # Model info
    num_topics: Optional[int] = None
    vocab_size: Optional[int] = None
    epochs_trained: Optional[int] = None
    
    # Metrics
    metrics: Optional[Dict[str, float]] = None
    
    # Available files
    has_model: bool = False
    has_theta: bool = False
    has_beta: bool = False
    has_topic_words: bool = False
    has_visualizations: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset": "socialTwitter",
                "mode": "zero_shot",
                "timestamp": "20240115_123456",
                "num_topics": 20,
                "metrics": {
                    "topic_coherence_avg": 0.58,
                    "topic_diversity_td": 0.72
                }
            }
        }


class VisualizationInfo(BaseModel):
    """Information about a visualization file"""
    name: str
    path: str
    type: str  # image, html, json
    size: Optional[int] = None
    created_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "topic_words_socialTwitter_zero_shot.png",
                "path": "/root/autodl-tmp/result/socialTwitter/zero_shot/visualization/topic_words.png",
                "type": "image"
            }
        }


class ProjectInfo(BaseModel):
    """Project overview information"""
    name: str = "THETA"
    version: str = "1.0.0"
    datasets_count: int = 0
    results_count: int = 0
    active_tasks: int = 0
    
    # System info
    gpu_available: bool = False
    gpu_id: int = 1
    
    # Recent activity
    recent_results: Optional[List[ResultInfo]] = None


class FileUploadResponse(BaseModel):
    """Response for file upload"""
    success: bool
    filename: str
    path: str
    message: str


class MetricsResponse(BaseModel):
    """Detailed metrics response"""
    dataset: str
    mode: str
    timestamp: str
    
    # Topic quality metrics
    topic_coherence_avg: Optional[float] = None
    topic_coherence_per_topic: Optional[List[float]] = None
    topic_diversity_td: Optional[float] = None
    topic_diversity_irbo: Optional[float] = None
    
    # Training metrics
    best_val_loss: Optional[float] = None
    test_loss: Optional[float] = None
    epochs_trained: Optional[int] = None
    
    # Additional metrics
    additional: Optional[Dict[str, Any]] = None

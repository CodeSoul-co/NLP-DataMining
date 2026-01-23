"""
Dataset Model
Database model for uploaded datasets
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Text, DateTime, BigInteger, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from ..core.database import Base


class Dataset(Base):
    """Dataset model for uploaded data files"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Dataset info
    name = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Storage info
    storage_type = Column(String(20), default="oss")  # local, oss
    storage_path = Column(String(500), nullable=False)  # OSS key or local path
    file_size = Column(BigInteger, nullable=False)
    file_type = Column(String(50), nullable=False)  # csv, json, xlsx
    
    # Data info
    row_count = Column(Integer, nullable=True)
    columns = Column(JSON, nullable=True)  # ["col1", "col2", ...]
    text_column = Column(String(255), nullable=True)  # detected or specified text column
    
    # Preprocessing status
    is_preprocessed = Column(Boolean, default=False)
    embedding_path = Column(String(500), nullable=True)  # OSS path to embeddings
    vocab_path = Column(String(500), nullable=True)  # OSS path to vocabulary
    bow_path = Column(String(500), nullable=True)  # OSS path to BOW matrix
    
    # 扩展元数据（避免与 SQLAlchemy Base.metadata 冲突）
    extra_meta = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="datasets")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "original_filename": self.original_filename,
            "description": self.description,
            "storage_type": self.storage_type,
            "storage_path": self.storage_path,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "row_count": self.row_count,
            "columns": self.columns,
            "text_column": self.text_column,
            "is_preprocessed": self.is_preprocessed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TrainingResult(Base):
    """Training result model"""
    __tablename__ = "training_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    
    # Result type
    result_type = Column(String(50), nullable=False)  # etm, detm
    
    # Model info
    num_topics = Column(Integer, nullable=False)
    model_path = Column(String(500), nullable=True)  # OSS path to model weights
    
    # Metrics
    metrics = Column(JSON, nullable=True)
    # {
    #     "coherence": 0.45,
    #     "perplexity": 120.5,
    #     "diversity": 0.8,
    #     "topic_quality": [...]
    # }
    
    # Topic words
    topic_words = Column(JSON, nullable=True)
    # [
    #     {"topic_id": 0, "words": [{"word": "政策", "weight": 0.05}, ...]},
    #     ...
    # ]
    
    # Visualization data paths
    visualization_paths = Column(JSON, nullable=True)
    # {
    #     "topic_distribution": "oss://...",
    #     "word_cloud": "oss://...",
    #     "similarity_matrix": "oss://..."
    # }
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "result_type": self.result_type,
            "num_topics": self.num_topics,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "topic_words": self.topic_words,
            "visualization_paths": self.visualization_paths,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

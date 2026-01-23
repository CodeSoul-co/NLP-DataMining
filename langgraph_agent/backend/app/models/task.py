"""
Task Model
Database model for training tasks
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from ..core.database import Base


class TaskStatus(str, enum.Enum):
    """Task status enum"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, enum.Enum):
    """Task type enum"""
    ETM = "etm"
    DETM = "detm"
    EMBEDDING = "embedding"
    PREPROCESSING = "preprocessing"


class Task(Base):
    """Task model for training jobs"""
    __tablename__ = "tasks"

    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Task info
    name = Column(String(255), nullable=True)
    task_type = Column(SQLEnum(TaskType), default=TaskType.ETM)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, index=True)
    
    # Dataset
    dataset_name = Column(String(255), nullable=False)
    text_column = Column(String(255), nullable=True)
    
    # Training config
    config = Column(JSON, nullable=False, default=dict)
    # Example config:
    # {
    #     "num_topics": 20,
    #     "vocab_size": 5000,
    #     "epochs": 50,
    #     "batch_size": 64,
    #     "embedding_model": "qwen"
    # }
    
    # Progress tracking
    progress = Column(Float, default=0.0)
    current_step = Column(String(255), nullable=True)
    
    # Results
    result = Column(JSON, nullable=True)
    # Example result:
    # {
    #     "metrics": {"coherence": 0.45, "perplexity": 120.5},
    #     "output_path": "/result/dataset/etm/",
    #     "topic_words": [...],
    #     "pai_job_id": "dlc-xxx"
    # }
    
    # Error info
    error_message = Column(Text, nullable=True)
    
    # PAI job tracking
    pai_job_id = Column(String(128), nullable=True, index=True)
    pai_job_status = Column(String(64), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="tasks")
    logs = relationship("TaskLog", back_populates="task", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "task_type": self.task_type.value if self.task_type else None,
            "status": self.status.value if self.status else None,
            "dataset_name": self.dataset_name,
            "text_column": self.text_column,
            "config": self.config,
            "progress": self.progress,
            "current_step": self.current_step,
            "result": self.result,
            "error_message": self.error_message,
            "pai_job_id": self.pai_job_id,
            "pai_job_status": self.pai_job_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TaskLog(Base):
    """Task log model"""
    __tablename__ = "task_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    
    level = Column(String(20), default="INFO")  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    extra_meta = Column(JSON, nullable=True)  # 避免与 Base.metadata 冲突

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    task = relationship("Task", back_populates="logs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "level": self.level,
            "message": self.message,
            "metadata": self.extra_meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

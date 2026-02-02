"""
Agent State and Request/Response Schemas
Defines the state object that flows through the LangGraph nodes
"""

from typing import TypedDict, Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Individual step status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentState(TypedDict, total=False):
    """
    State object passed between LangGraph nodes
    Contains all information needed for the ETM pipeline
    """
    # Task identification
    task_id: str
    
    # Input configuration
    dataset: str
    mode: str  # zero_shot / supervised / unsupervised
    
    # Paths
    data_path: str
    result_dir: str
    embeddings_dir: str
    bow_dir: str
    model_dir: str
    evaluation_dir: str
    visualization_dir: str
    
    # Model hyperparameters
    num_topics: int
    vocab_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    
    # Execution state
    current_step: str
    status: str
    error_message: Optional[str]
    
    # Step completion flags
    preprocess_completed: bool
    embedding_completed: bool
    training_completed: bool
    evaluation_completed: bool
    visualization_completed: bool
    
    # Intermediate results (matrix dimensions for dev mode)
    bow_shape: Optional[tuple]
    vocab_size_actual: Optional[int]
    doc_embeddings_shape: Optional[tuple]
    theta_shape: Optional[tuple]
    beta_shape: Optional[tuple]
    
    # Final results
    metrics: Optional[Dict[str, float]]
    topic_words: Optional[Dict[str, List[str]]]
    training_history: Optional[Dict[str, List[float]]]
    visualization_paths: Optional[List[str]]
    
    # Timestamps
    created_at: str
    updated_at: str
    completed_at: Optional[str]
    
    # Execution log
    logs: List[Dict[str, Any]]


class TaskRequest(BaseModel):
    """Request to start a new ETM pipeline task"""
    dataset: str = Field(..., description="Dataset name (e.g., socialTwitter)")
    mode: str = Field(
        default="zero_shot",
        description="Embedding mode: zero_shot, supervised, unsupervised"
    )
    num_topics: int = Field(default=20, ge=5, le=100, description="Number of topics")
    vocab_size: int = Field(default=5000, ge=1000, le=20000, description="Vocabulary size")
    epochs: int = Field(default=50, ge=10, le=500, description="Training epochs")
    batch_size: int = Field(default=64, ge=16, le=256, description="Batch size")
    learning_rate: float = Field(default=0.002, gt=0, le=0.1, description="Learning rate")
    hidden_dim: int = Field(default=512, ge=128, le=2048, description="Hidden dimension")
    dev_mode: bool = Field(default=False, description="Enable debug logging")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset": "socialTwitter",
                "mode": "zero_shot",
                "num_topics": 20,
                "vocab_size": 5000,
                "epochs": 50
            }
        }


class StepUpdate(BaseModel):
    """Real-time step update sent via WebSocket"""
    task_id: str
    step: str
    status: StepStatus
    message: str
    progress: Optional[float] = Field(None, ge=0, le=100)
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskResponse(BaseModel):
    """Response for task status query"""
    task_id: str
    status: TaskStatus
    current_step: Optional[str] = None
    progress: float = Field(default=0, ge=0, le=100)
    
    # Results (populated when completed)
    metrics: Optional[Dict[str, float]] = None
    topic_words: Optional[Dict[str, List[str]]] = None
    visualization_paths: Optional[List[str]] = None
    
    # Timing
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Error info
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_20240115_123456",
                "status": "completed",
                "current_step": "visualization",
                "progress": 100,
                "metrics": {
                    "topic_coherence_avg": 0.58,
                    "topic_diversity_td": 0.72
                }
            }
        }


class ChatMessage(BaseModel):
    """Chat message for the conversational interface"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Request for chat interaction"""
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response from chat interaction"""
    message: str
    action: Optional[str] = None  # e.g., "start_task", "show_results"
    task_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

"""
Project Model
用户项目 - 关联用户与数据集/任务，持久化到数据库
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from ..core.database import Base


class Project(Base):
    """用户项目 - 与用户关联，跨设备同步"""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # 项目信息
    name = Column(String(255), nullable=False)
    dataset_name = Column(String(255), nullable=True, index=True)  # 关联的数据集名（上传后填充）

    # 分析配置
    mode = Column(String(50), default="zero_shot")  # zero_shot, unsupervised, supervised
    num_topics = Column(Integer, default=20)

    # 状态
    status = Column(String(50), default="draft")  # draft, uploading, running, completed, error
    pipeline_status = Column(String(50), nullable=True)  # running, completed, error
    task_id = Column(String(128), nullable=True, index=True)  # 关联的训练任务 ID

    # 扩展信息
    extra = Column(JSON, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="projects")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "dataset_name": self.dataset_name,
            "mode": self.mode,
            "num_topics": self.num_topics,
            "status": self.status,
            "pipeline_status": self.pipeline_status,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

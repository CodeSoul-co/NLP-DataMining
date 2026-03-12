"""
UserDataset Model
Tracks ownership of locally-stored datasets (DATA_DIR/{dataset_name}/)
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from ..core.database import Base


class UserDataset(Base):
    """Maps user_id to dataset_name for local file storage isolation"""
    __tablename__ = "user_datasets"
    __table_args__ = (UniqueConstraint("user_id", "dataset_name", name="uq_user_dataset"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", backref="user_datasets")

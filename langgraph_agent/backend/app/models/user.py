"""
User Model
Database model for users with SQLAlchemy
"""

import hashlib
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import relationship
from passlib.context import CryptContext

from ..core.database import Base
from ..core.logging import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prehash_password(password: str) -> str:
    """
    Pre-hash password with SHA-256 before bcrypt.
    This allows passwords of any length while staying within bcrypt's 72 byte limit.
    Returns base64 encoded hash (44 characters, well under 72 bytes).
    """
    password_bytes = password.encode('utf-8')
    sha256_hash = hashlib.sha256(password_bytes).digest()
    return base64.b64encode(sha256_hash).decode('ascii')


class User(Base):
    """User model"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        prehashed = _prehash_password(plain_password)
        return pwd_context.verify(prehashed, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        prehashed = _prehash_password(password)
        return pwd_context.hash(prehashed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


# ========== UserDB wrapper for backward compatibility ==========
# This maintains the same interface as the old SQLite-based implementation

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class UserDB:
    """User database operations - wrapper for backward compatibility"""
    
    def __init__(self, session_maker=None):
        self._session_maker = session_maker

    def set_session_maker(self, session_maker):
        """Set the async session maker"""
        self._session_maker = session_maker

    async def _get_session(self) -> AsyncSession:
        """Get an async session"""
        if self._session_maker is None:
            from ..core.database import async_session_maker
            self._session_maker = async_session_maker
        return self._session_maker()

    async def initialize(self):
        """Initialize database tables (handled by database.init_db)"""
        pass  # No longer needed, handled by SQLAlchemy

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> User:
        """Create a new user"""
        hashed_password = User.get_password_hash(password)
        
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True
        )
        
        async with await self._get_session() as session:
            session.add(user)
            try:
                await session.commit()
                await session.refresh(user)
                return user
            except Exception as e:
                await session.rollback()
                error_str = str(e).lower()
                if "unique" in error_str or "duplicate" in error_str:
                    if "username" in error_str:
                        raise ValueError(f"Username '{username}' already exists")
                    elif "email" in error_str:
                        raise ValueError(f"Email '{email}' already exists")
                raise

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        async with await self._get_session() as session:
            result = await session.execute(
                select(User).where(User.username == username, User.is_active == True)
            )
            return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        async with await self._get_session() as session:
            result = await session.execute(
                select(User).where(User.email == email, User.is_active == True)
            )
            return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        async with await self._get_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id, User.is_active == True)
            )
            return result.scalar_one_or_none()

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user by username/email and password"""
        # Try username first
        user = await self.get_user_by_username(username)
        if not user:
            # Try email
            user = await self.get_user_by_email(username)
        
        if not user:
            return None
        
        if not User.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        async with await self._get_session() as session:
            user.last_login = datetime.utcnow()
            session.add(user)
            await session.commit()
        
        return user

    async def update_user(
        self,
        user_id: int,
        email: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Optional[User]:
        """Update user profile"""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        async with await self._get_session() as session:
            if email is not None and email != user.email:
                # Check if email already exists
                existing = await self.get_user_by_email(email)
                if existing and existing.id != user_id:
                    raise ValueError(f"Email '{email}' already in use")
                user.email = email
            
            if full_name is not None:
                user.full_name = full_name
            
            session.add(user)
            await session.commit()
            await session.refresh(user)
        
        return user

    async def change_password(
        self,
        user_id: int,
        current_password: str,
        new_password: str
    ) -> bool:
        """Change user password"""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Verify current password
        if not User.verify_password(current_password, user.hashed_password):
            return False
        
        # Hash new password and update
        async with await self._get_session() as session:
            user.hashed_password = User.get_password_hash(new_password)
            session.add(user)
            await session.commit()
        
        return True


# Global user database instance
user_db = UserDB()
